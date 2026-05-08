import wandb
import numpy as np
import torch
import collections
import tqdm
import zarr
import os
import random
import pickle
from termcolor import cprint
from diffusion_policy_3d.env import DexArtEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from diffusion_policy_3d.model.vision.articubot import PointNet2_super
from train_high_level import compute_weighted_displacement


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 goal_mode='None',
                 oracle_goal_zarr=None,
                 eef_points = 4,
                 high_level_ckpt='None',
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            self.base_env = DexArtEnv(
                task_name=task_name,
                use_test_set=is_test,
            )
            self.base_env.seed(4)
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(self.base_env),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        seed = 4
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env_train = env_fn(is_test=False)
        self.env_train.seed(seed)

        self.episode_train = n_train
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.goal_mode = goal_mode
        self.eef_points = eef_points

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)
        
        if self.goal_mode == 'pointcloud_oracle':
            self.demo_data = zarr.open(oracle_goal_zarr, mode='r')
            self.retrieve_goal()
        elif self.goal_mode == 'high_level':
            num_classes = self.eef_points * 3 + 1
            self.high_level_model = PointNet2_super(num_classes=num_classes, input_channel=3)
            self.high_level_model.load_state_dict(torch.load(high_level_ckpt))
        

    def retrieve_goal(self):
        episode_ends = self.demo_data["meta"]["episode_ends"]
        progress = self.demo_data["meta"]["progress"]
        imagin_robot = self.demo_data["data"]["imagin_robot"]
        demo_id = self.demo_data["meta"]["demo_id"]


        num_demos = len(episode_ends)
        goal_gripper_pcd = []
        for i in range(num_demos):        
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            last_idx = episode_ends[i] - 1
            
            progress_idx = progress[i]
            first_goal = imagin_robot[start_idx + progress_idx]
            first_goal = first_goal[..., :3] 
            second_goal = imagin_robot[last_idx]
            second_goal = second_goal[..., :3] 
            
            goal_gripper_pcd.append({
                "demo_id": demo_id[i],
                "first_goal": first_goal,
                "second_goal": second_goal,
            })
        self.goal_gripper_pcd = goal_gripper_pcd

    def get_obs(self):
        """Observation for saving"""
        state = self.first_obs
        obs_dict = {
            'observed_point_cloud': state['point_cloud'], # T, 1024, 6
        }
        return obs_dict
    

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []
        
        # demo_save_dir_success = os.path.join('data/demo_dp3_24', 'success_demo')
        # demo_save_dir_failure = os.path.join('data/demo_dp3_24', 'failure_demo')
        # os.makedirs(demo_save_dir_success, exist_ok=True)
        # os.makedirs(demo_save_dir_failure, exist_ok=True)

        init_seed = 0
        seed_step = 1399  # set some random seed interval
        success_id = 0

        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            
            # set seed
            if self.goal_mode == 'pointcloud_oracle':
                oracle_goal = self.goal_gripper_pcd[episode_id]
                episode_seed = init_seed + oracle_goal["demo_id"] * seed_step
            else:
                episode_seed = np.random.randint(0, 25536)
            
            self.base_env.seed(episode_seed)
            obs = env_train.reset()

            # self.first_obs = obs
            # first_obs = self.get_obs()
            # with open(os.path.join(demo_save_dir_success, f'first_obs_{oracle_goal["demo_id"]}.pkl'), 'wb') as f:
            #     pickle.dump(first_obs, f)

            policy.reset()

            done = False
            reward_sum = 0.
            progress = 0
            goal_pcd_arrays = []

            demo_data = []
            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    # print(obs_dict_input['imagin_robot'].shape)
                    
                    # print(obs_dict_input['imagin_robot'].shape)
                    # print(obs_dict_input['point_cloud'].shape)
                    # print(obs_dict_input['agent_pos'].shape)
                    # print(self.goal_gripper_pcd[0].shape)
                    # breakpoint()
                    
                    if self.goal_mode =='None':
                        obs_dict_input['goal_gripper_pcd'] = obs_dict['imagin_robot'].unsqueeze(0)
                    elif self.goal_mode == 'pointcloud_oracle':

                        shape = obs_dict_input['imagin_robot'].shape
                        batch_size, obs_step, _, _ = shape

                        if progress < 1e-5:
                            goal_pcd = torch.tensor(oracle_goal["first_goal"])
                        else:
                            goal_pcd = torch.tensor(oracle_goal["second_goal"])
                        goal_pcd = goal_pcd.unsqueeze(0).unsqueeze(1)  # (1,1,N,F)

                        # Add batch and obs_step dims and expand to match size
                        goal_pcd_arrays = goal_pcd.expand(batch_size, obs_step, *goal_pcd.shape[2:]) 
                        obs_dict_input['goal_gripper_pcd'] = goal_pcd_arrays

                    elif self.goal_mode == 'high_level':

                        assert self.high_level_model is not None
                        self.high_level_model = self.high_level_model.to(device)

                        # print(obs_dict['imagin_robot'].shape)
                        # print(obs_dict['point_cloud'].shape)
                        num_imagin_points = obs_dict['imagin_robot'].shape[1]
                        
                        if self.eef_points == 4:
                            chosen_four_point_idx = torch.tensor([16, 40, 64, 88])
                        elif self.eef_points == 12:    
                            chosen_four_point_idx = torch.tensor([4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92]) # One on each link
                        goal_pcd = torch.zeros((1, num_imagin_points, 3), dtype=torch.float32).to(device)
                        
                        pcd, imagined_pcd = obs_dict['point_cloud'][-1, :, :3], obs_dict['imagin_robot'][-1, :, :3]
                        high_level_obs = torch.cat([pcd, imagined_pcd], axis=0)[None].permute(0,2,1).to(device).float()
                        
                        with torch.no_grad():
                            pred = self.high_level_model(high_level_obs)
                        
                        if self.eef_points == 4:
                            goal_type = "4points"
                        elif self.eef_points == 12:
                            goal_type = "12points"
                        else:
                            raise NotImplementedError

                        pred_points = compute_weighted_displacement(high_level_obs, pred, goal_type)
                        goal_pcd[:, chosen_four_point_idx] = pred_points
                        # print(goal_pcd.shape)
                        shape = obs_dict_input['imagin_robot'].shape
                        batch_size, obs_step, _, _ = shape
                        goal_pcd_arrays = goal_pcd.unsqueeze(1).expand(batch_size, obs_step, *goal_pcd.shape[1:])
                        obs_dict_input['goal_gripper_pcd'] = goal_pcd_arrays
                        
                    else:
                        raise ValueError("Selected goal_mode not implemented.")

                    observed = {
                        'point_cloud': obs_dict_input['point_cloud'][0, -1],
                        'goal_gripper_pcd': obs_dict_input['goal_gripper_pcd'][0, -1],
                        'imagin_robot': obs_dict_input['imagin_robot'][0, -1],
                    }
                    demo_data.append(observed)

                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                
                # step env
                obs, reward, done, info = env_train.step(action)

                progress = obs["progress"][1]

                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())

            # if env_train.is_success():
            #     filename = f"episode_{episode_id}.pkl"
            #     with open(os.path.join(demo_save_dir_success, filename), 'wb') as f:
            #         pickle.dump(demo_data, f)
            # else:
            #     filename = f"episode_{episode_id}.pkl"
            #     with open(os.path.join(demo_save_dir_failure, filename), 'wb') as f:
            #         pickle.dump(demo_data, f)



        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train

        log_data['test_mean_score'] = SR_mean_train

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {SR_mean_train:.3f}", 'green')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data
