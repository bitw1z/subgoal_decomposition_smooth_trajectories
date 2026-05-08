import os
import csv
import glob
import random
import pickle
import numpy as np
import imageio.v2 as imageio

from PIL import Image
from rlbench import tasks
from rlbench.demo import Demo
from rlbench.backend import utils
from pyrep.const import ObjectType
from rlbench.backend.const import *
from pyrep.objects.shape import Shape
from scipy.signal import savgol_filter
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import *
from rlbench.action_modes.gripper_action_modes import Discrete

headless_state = True

def create_obs_config(image_size=[128, 128], renderer='opengl3'):
    """Create observation configuration matching RVT/PerAct training data format"""
    from pyrep.const import RenderMode

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # Set image sizes for all cameras (RVT/PerAct standard)
    for camera in ['right_shoulder_camera', 'left_shoulder_camera', 'overhead_camera', 
                   'wrist_camera', 'front_camera']:
        getattr(obs_config, camera).image_size = image_size
        getattr(obs_config, camera).depth_in_meters = False  # 0-1 scale
        getattr(obs_config, camera).masks_as_one_channel = False  # RGB encoding

    # Set render mode
    render_mode = RenderMode.OPENGL3 if renderer == 'opengl3' else RenderMode.OPENGL
    for camera in ['right_shoulder_camera', 'left_shoulder_camera', 'overhead_camera', 
                   'wrist_camera', 'front_camera']:
        getattr(obs_config, camera).render_mode = render_mode

    return obs_config

def create_action_mode():
    return MoveArmThenGripper(
        arm_action_mode=JointVelocity(), 
        gripper_action_mode=Discrete()
    )

def save_keypoints_txt(all_keypoints, filepath):
    with open(filepath, 'w') as f:
        for keypoints in all_keypoints:  # Iterate through the list
            f.write(f"Demo: {keypoints['demo_num']} | Variation: {keypoints['variation']}\n")
            # Add whatever other keypoint data you want to write
            f.write(f"Original keypoints: {keypoints['original_keypoints']}\n")
            f.write(f"Smoothed keypoints 01: {keypoints['smoothed_keypoints_01']}\n") 
            f.write(f"Smoothed keypoints best: {keypoints['smoothed_keypoints_best']}\n")
            f.write("-" * 50 + "\n")  # Separator between demos

def sync_demo_with_ee_actions(demo, ee_actions):
    assert len(demo) == len(ee_actions), (len(demo), len(ee_actions))

    for t, obs in enumerate(demo):
        # overwrite pose + gripper state
        obs.gripper_pose = ee_actions[t, :7].astype(np.float32)
        obs.gripper_open = float(ee_actions[t, -1])

    return demo

def save_demo(demo, episode_path, variation, descriptions, initial_poses, ee_actions, detection_threshold):
    """Save demo in RVT/PerAct compatible format (same as RLBench dataset_generator.py)"""

    # Create folder structure matching RVT/PerAct expectations
    camera_folders = {
        'left_shoulder': {
            'rgb': os.path.join(episode_path, LEFT_SHOULDER_RGB_FOLDER),
            'depth': os.path.join(episode_path, LEFT_SHOULDER_DEPTH_FOLDER),
            'mask': os.path.join(episode_path, LEFT_SHOULDER_MASK_FOLDER)
        },
        'right_shoulder': {
            'rgb': os.path.join(episode_path, RIGHT_SHOULDER_RGB_FOLDER),
            'depth': os.path.join(episode_path, RIGHT_SHOULDER_DEPTH_FOLDER),
            'mask': os.path.join(episode_path, RIGHT_SHOULDER_MASK_FOLDER)
        },
        'overhead': {
            'rgb': os.path.join(episode_path, OVERHEAD_RGB_FOLDER),
            'depth': os.path.join(episode_path, OVERHEAD_DEPTH_FOLDER),
            'mask': os.path.join(episode_path, OVERHEAD_MASK_FOLDER)
        },
        'wrist': {
            'rgb': os.path.join(episode_path, WRIST_RGB_FOLDER),
            'depth': os.path.join(episode_path, WRIST_DEPTH_FOLDER),
            'mask': os.path.join(episode_path, WRIST_MASK_FOLDER)
        },
        'front': {
            'rgb': os.path.join(episode_path, FRONT_RGB_FOLDER),
            'depth': os.path.join(episode_path, FRONT_DEPTH_FOLDER),
            'mask': os.path.join(episode_path, FRONT_MASK_FOLDER)
        }
    }

    # Create all directories
    for camera_data in camera_folders.values():
        for path in camera_data.values():
            os.makedirs(path, exist_ok=True)

    # Save images
    for i, obs in enumerate(demo):
        cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']

        for camera in cameras:
            # Get observation data
            rgb_data = getattr(obs, f'{camera}_rgb')
            depth_data = getattr(obs, f'{camera}_depth')
            mask_data = getattr(obs, f'{camera}_mask')

            # Save RGB
            Image.fromarray(rgb_data).save(
                os.path.join(camera_folders[camera]['rgb'], IMAGE_FORMAT % i))

            # Save depth (RVT/PerAct expects specific depth encoding)
            utils.float_array_to_rgb_image(depth_data, scale_factor=DEPTH_SCALE).save(
                os.path.join(camera_folders[camera]['depth'], IMAGE_FORMAT % i))

            # Save mask
            Image.fromarray((mask_data * 255).astype(np.uint8)).save(
                os.path.join(camera_folders[camera]['mask'], IMAGE_FORMAT % i))

        '''
        # Nullify image data to save memory
        for camera in cameras:
            setattr(obs, f'{camera}_rgb', None)
            setattr(obs, f'{camera}_depth', None)
            setattr(obs, f'{camera}_point_cloud', None)
            setattr(obs, f'{camera}_mask', None)
        '''

    # Synchronize demo observations with ee_actions
    demo = sync_demo_with_ee_actions(demo, ee_actions)

    # Save low-dimensional data (joint states, etc.)
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    # Save variation number 
    with open(os.path.join(episode_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)

    # Save variation_descriptions
    with open(os.path.join(episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
        pickle.dump(descriptions, f)
    
    # Save ee_actions
    np.save(os.path.join(episode_path, 'smooth_ee_actions.npy'), np.array(ee_actions))

    # Save initial poses
    np.savez(os.path.join(episode_path, 'initial_poses.npz'), **initial_poses)

    # Save detection_threshold
    np.save(os.path.join(episode_path, 'detection_threshold.npy'), detection_threshold)


def setup_episode_path(output_dir, task_name, episode_num):
    """Create RVT/PerAct compatible episode path structure"""
    variation_path = os.path.join(output_dir, 'all_variations')
    episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % episode_num)
    video_dir = os.path.join(episode_path, 'demo_videos')
    
    os.makedirs(episode_path, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    return episode_path, video_dir

def generate_and_record_demo(task_name, previous_variation, video_dir, image_size=[128, 128], renderer='opengl3'):
    """Generate and record original demo"""

    print(f"Generating original demo for {task_name}")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    # Initialize environment 
    env = Environment(
        action_mode=create_action_mode(),
        obs_config=create_obs_config(image_size, renderer),
        headless=headless_state  # Visual feedback during recording
    )
    env.launch()

    # Get task and record demonstration
    task_env = env.get_task(task_class)
    variation = (previous_variation + 1) % task_env.variation_count()
    task_env.set_variation(variation)
    descriptions, _ = task_env.reset() # Apply variation
    
    # Get all objects in the scene dynamically
    cube = Shape('cube')
    stick = Shape('stick')
    target0 = Shape('target0')
    distractor1 = Shape('distractor1')
    distractor2 = Shape('distractor2')
    distractor3 = Shape('distractor3')

    initial_poses = {
        'cube_pose': np.array(cube.get_pose()),
        'stick_pose': np.array(stick.get_pose()),
        'target0_pose': np.array(target0.get_pose()),
        'distractor1_pose': np.array(distractor1.get_pose()),
        'distractor2_pose': np.array(distractor2.get_pose()),
        'distractor3_pose': np.array(distractor3.get_pose()),
    }

    cube_pose = initial_poses['cube_pose']
    target0_pose = initial_poses['target0_pose']
    cube_x = cube_pose[0]
    target0_x = target0_pose[0]
    if target0_x > cube_x: 
        print('Not pull. skip this demo')
        env.shutdown()
        return None, None, None, None, None

    print("📹 Recording demonstration...")
    try: 
        demos = task_env.get_demos(amount=1, live_demos=True)
    except Exception as e:
        print(f"Unsuccessful demo is created. Skipping this demo!")
        env.shutdown()
        return None, None, None, None, None
        
    original_demo = demos[0]

    video_frames = [obs.front_rgb for obs in original_demo]

    raw_frames_dir = os.path.join(video_dir, 'raw_frames')
    os.makedirs(raw_frames_dir, exist_ok=True)

    for f in glob.glob(os.path.join(raw_frames_dir, '*')):
        os.remove(f)

    for i, frame in enumerate(video_frames):
        frame_path = os.path.join(video_dir, 'raw_frames', f"frame_{i:04d}.png")  
        imageio.imwrite(frame_path, frame)

    video_path = os.path.join(video_dir, 'original_demo.mp4')
    imageio.mimsave(video_path, video_frames, fps=30)

    actions = np.array([
        np.concatenate([obs.joint_velocities.flatten(), [obs.gripper_open]]) 
        for obs in original_demo
    ])


    print(f"✅ Recorded {len(actions)} action steps")

    env.shutdown()
    return original_demo, initial_poses, actions, variation, descriptions

def replay_demo(task_name, initial_poses, actions, variation, video_dir, stage, image_size=[128, 128], renderer='opengl3'):
    """Replay demo and verify the replay is successful"""
    
    print("🔄 Replaying with provided actions...")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    # Initialize environment
    env = Environment(
        action_mode=create_action_mode(),
        obs_config=create_obs_config(image_size, renderer),
        headless=headless_state # Visual feedback during recording
    )
    env.launch()

    task_env = env.get_task(task_class)
    task_env.set_variation(variation)
    task_env.reset()

    # Reset scene to initial state
    cube = Shape('cube')
    stick = Shape('stick')
    target0 = Shape('target0')
    distractor1 = Shape('distractor1')
    distractor2 = Shape('distractor2')
    distractor3 = Shape('distractor3')
    
    cube.set_pose(initial_poses['cube_pose'])
    stick.set_pose(initial_poses['stick_pose'])
    target0.set_pose(initial_poses['target0_pose'])
    distractor1.set_pose(initial_poses['distractor1_pose'])
    distractor2.set_pose(initial_poses['distractor2_pose'])
    distractor3.set_pose(initial_poses['distractor3_pose'])

    observations = []
    video_frames = []
    for action in actions: 
        obs, reward, terminate = task_env.step(action)
        observations.append(obs)
        video_frames.append(obs.front_rgb)

    env._pyrep.stop()

    success = (reward == 1)
    
    env.shutdown()
    new_demo = Demo(observations)

    if success: 
        video_path = os.path.join(video_dir, f'{stage}.mp4')
        imageio.mimsave(video_path, video_frames, fps=30)

    return success, new_demo

def motion_plan(task_name, initial_poses, actions, variation, video_dir, stage, image_size=[128, 128], renderer='opengl3'): 
    """Perform motion planning based on the keyframe end-effector poses"""

    print("🔄 Motion planning with provided actions...")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    custom_action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), # collision_checking=True for motion plan for Slide Block
        gripper_action_mode=Discrete(),
    )

    # Initialize environment
    env = Environment(
        action_mode=custom_action_mode,
        obs_config=create_obs_config(image_size, renderer),
        headless=headless_state # Visual feedback during recording
    )
    env.launch()

    task_env = env.get_task(task_class)
    task_env.set_variation(variation)
    task_env.reset()

    # Reset scene to initial state
    cube = Shape('cube')
    stick = Shape('stick')
    target0 = Shape('target0')
    distractor1 = Shape('distractor1')
    distractor2 = Shape('distractor2')
    distractor3 = Shape('distractor3')
    
    cube.set_pose(initial_poses['cube_pose'])
    stick.set_pose(initial_poses['stick_pose'])
    target0.set_pose(initial_poses['target0_pose'])
    distractor1.set_pose(initial_poses['distractor1_pose'])
    distractor2.set_pose(initial_poses['distractor2_pose'])
    distractor3.set_pose(initial_poses['distractor3_pose'])

    observations = []
    video_frames = []
    try: 
        for action in actions:
            obs, reward, terminate = task_env.step(action)
            observations.append(obs)
            video_frames.append(obs.front_rgb)
    except Exception as e: 
        env.shutdown()
        print(f'motionplan went wrong: {e}')
        return False, False 

    env._pyrep.stop()
    success = (reward == 1.0)
    
    env.shutdown()

    video_path = os.path.join(video_dir, f'{stage}.mp4')
    imageio.mimsave(video_path, video_frames, fps=3)
    
    return success, True