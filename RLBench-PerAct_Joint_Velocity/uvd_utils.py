import os 
import pickle
import numpy as np
from ee_utils import extract_key_actions
from pipeline_utils import motion_plan

# === Configurable Parameters ===
task_name = 'SweepToDustpanOfSize'
episode_dir = '/home/jaehyukchang/Final Data/SweepToDustpanOfSize/all_variations/episodes/'
num_of_episodes = 100
success_count = 0 
success_indices = []

for i in range(num_of_episodes): 
    print(f"\n{'='*20} ATTEMPTING DEMO {i} {'='*20}")
    print(f"Successful demos so far: {success_count}/{num_of_episodes}")
    
    episode_path = os.path.join(episode_dir, f'episode{i}')
    initial_poses = np.load(os.path.join(episode_path, 'initial_poses.npz'))
    
    with open(os.path.join(episode_path, 'variation_number.pkl'), 'rb') as f:
        variation = pickle.load(f)
    
    with open(os.path.join(episode_path, 'low_dim_obs.pkl'), 'rb') as f:
        demo = pickle.load(f)

    uvd_keypoints = np.load(os.path.join(episode_path, 'uvd_keypoints_vip.npy'))
    print(uvd_keypoints)
    actions = np.array([
        np.concatenate([obs.gripper_pose.flatten(), [obs.gripper_open]]) 
        for obs in demo
    ])

    keypoint_actions = extract_key_actions(actions, uvd_keypoints)
    print(f'len of actions: {len(keypoint_actions)}')

    video_dir = f'/home/jaehyukchang/Final Data/SweepToDustpanOfSize/all_variations/episodes/episode{i}/demo_videos'
    try:
        motion_plan_success, result = motion_plan(
            task_name=task_name,
            initial_poses=initial_poses,
            actions=keypoint_actions,
            variation=variation, 
            video_dir=video_dir,
            stage = 'motion_plan_uvd',
        )
    except Exception as e:
        print(f"Error while motion planning demo for task {task_name}: {e}")
    
    if motion_plan_success: 
        print('Motion Plan Successful!')
        success_count += 1
        success_indices.append(i)
    else: 
        print('Motion Plan Unsuccessful!')

print(f'success rate: {success_count/num_of_episodes}')
print(f'success cases: {success_indices}')