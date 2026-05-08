import os
import pickle
import random
import logging
import argparse
import numpy as np
import imageio.v2 as imageio

from PIL import Image
from typing import List
from rlbench import tasks
from rlbench.demo import Demo
from rlbench.backend import utils
from rlbench.backend.const import *
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import *
from rlbench.action_modes.gripper_action_modes import Discrete

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
        arm_action_mode=JointPosition(), 
        gripper_action_mode=Discrete()
    )

def save_demo(demo, episode_path, variation):
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

        # Nullify image data to save memory
        for camera in cameras:
            setattr(obs, f'{camera}_rgb', None)
            setattr(obs, f'{camera}_depth', None)
            setattr(obs, f'{camera}_point_cloud', None)
            setattr(obs, f'{camera}_mask', None)

    # Save low-dimensional data (joint states, etc.)
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    # Save variation number 
    with open(os.path.join(episode_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)

def setup_episode_path(output_dir, task_name, episode_num):
    """Create RVT/PerAct compatible episode path structure"""
    variation_path = os.path.join(output_dir, 'all_variations')
    episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % episode_num)

    os.makedirs(episode_path, exist_ok=True)
    return episode_path

def generate_and_record_demo(task_name, image_size=[128, 128], renderer='opengl3'):
    """Generate and record original demo"""

    print(f"Generating original demo for {task_name}")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    # Initialize environment 
    env = Environment(
        action_mode=create_action_mode(),
        obs_config=create_obs_config(image_size, renderer),
        headless=True  # Visual feedback during recording
    )
    env.launch()

    # Get task and record demonstration
    task_env = env.get_task(task_class)
    variation = random.randint(0, task_env.variation_count()-1) # Randomize variation
    task_env.set_variation(variation)
    task_env.reset() # Apply variation

    # Get all objects in the scene dynamically
    target0 = Shape('push_buttons_target0')
    target1 = Shape('push_buttons_target1')
    target2 = Shape('push_buttons_target2')

    initial_poses = {
        'target0_pose': np.array(target0.get_pose()),
        'target1_pose': np.array(target1.get_pose()),
        'target2_pose': np.array(target2.get_pose()),
    }

    print("📹 Recording demonstration...")
    demos = task_env.get_demos(amount=1, live_demos=True)
    original_demo = demos[0]

    # Extract actions from demo
    actions = []
    for step in original_demo:
        action = np.concatenate([step.joint_positions.flatten(), [step.gripper_open]])
        actions.append(action)
    actions = np.array(actions)
    print(f"✅ Recorded {len(actions)} action steps")

    env.shutdown()
   
    return original_demo, initial_poses, actions, variation

def replay_demo(task_name, initial_poses, actions, variation, image_size=[128, 128], renderer='opengl3'):
    """Replay demo and verify the replay is successful"""
    
    print("🔄 Replaying with provided actions...")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    # Initialize environment
    env = Environment(
        action_mode=create_action_mode(),
        obs_config=create_obs_config(image_size, renderer),
        headless=True # Visual feedback during recording
    )
    env.launch()

    task_env = env.get_task(task_class)
    task_env.set_variation(variation)
    task_env.reset()

    # Reset scene to initial state
    target0 = Shape('push_buttons_target0')
    target1 = Shape('push_buttons_target1')
    target2 = Shape('push_buttons_target2')

    target0_pose = initial_poses['target0_pose']
    target1_pose = initial_poses['target1_pose']
    target2_pose = initial_poses['target2_pose']

    target0.set_pose(target0_pose)
    target1.set_pose(target1_pose)
    target2.set_pose(target2_pose)

    observations = []
    for action in actions:
        obs, reward, terminate = task_env.step(action)
        observations.append(obs)

    env._pyrep.stop()
    success = (reward == 1.0)
    
    env.shutdown()
    new_demo = Demo(observations)

    return success, new_demo

def smooth_joint_positions(actions, window_length, polyorder): 
    """Apply Savitzky-Golay filter to smooth joint positions"""
    # Ensure window_length is odd and valid
    if window_length >= len(actions):
        window_length = len(actions) - 1 if len(actions) % 2 == 0 else len(actions) - 2
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length - 1)

    # Create smoothed velocities array (copy of original)
    joint_positions = actions[:, :7]  # First 7 columns are joint positions

    # Apply Savitzky-Golay filter
    smoothed_positions = savgol_filter(
        joint_positions, 
        window_length=window_length,
        polyorder=polyorder, 
        axis=0
    )

    smoothed_actions = actions.copy()
    smoothed_actions[:, :7] = smoothed_positions

    return smoothed_actions

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError

def motion_plan(task_name, initial_poses, actions, variation, image_size=[128, 128], renderer='opengl3'): 
    """Perform motion planning based on the keyframe end-effector poses"""

    print("🔄 Motion planning with provided actions...")
    # Get task class from the tasks module
    task_class = getattr(tasks, task_name)

    custom_action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=True),
        gripper_action_mode=Discrete(),
    )

    # Initialize environment
    env = Environment(
        action_mode=custom_action_mode,
        obs_config=create_obs_config(image_size, renderer),
        headless=True # Visual feedback during recording
    )
    env.launch()

    task_env = env.get_task(task_class)
    task_env.set_variation(variation)
    task_env.reset()

    # Reset scene to initial state
    target0 = Shape('push_buttons_target0')
    target1 = Shape('push_buttons_target1')
    target2 = Shape('push_buttons_target2')

    target0_pose = initial_poses['target0_pose']
    target1_pose = initial_poses['target1_pose']
    target2_pose = initial_poses['target2_pose']

    target0.set_pose(target0_pose)
    target1.set_pose(target1_pose)
    target2.set_pose(target2_pose)

    observations = []
    for action in actions:
        obs, reward, terminate = task_env.step(action)
        observations.append(obs)

    env._pyrep.stop()
    success = (reward == 1.0)
    
    env.shutdown()

    return success

def extract_ee_poses(demo, keypoints):
    keypoints = set(keypoints)

    keypoint_actions = []
    for idx, obs in enumerate(demo):
        if idx in keypoints:
            keypoint_action = np.concatenate([obs.gripper_pose.flatten(), [obs.gripper_open]])
            keypoint_actions.append(keypoint_action)
    
    return keypoint_actions

def run_single_demo(task_name, window_length, polyorder, episode_num):
    results = []
    keypoint_counts = {
        'original': 0,
        'smoothed_0.1': 0,
        'smoothed_0.25': 0
    }

    # Launch demo 
    try:
        original_demo, initial_poses, actions, variation = generate_and_record_demo(
            task_name=task_name, 
        )
    except Exception as e:
         print(f"Error while generating demo for task {task_name}: {e}")
         return [], []

    # Replay demo 
    replay_success, _ = replay_demo(
        task_name=task_name,
        initial_poses=initial_poses,
        actions=actions,
        variation=variation, 
    )
    if not replay_success:
        print("Replay unsuccessful")
        return [], []

    # Extract keypoints 
    original_keypoints = keypoint_discovery(original_demo, stopping_delta=0.1, method='heuristic')
    keypoint_counts['original'] = len(original_keypoints)
    
    # Extract ee_poses corresponding to those keypoints 
    ee_poses_original = extract_ee_poses(original_demo, original_keypoints)
    '''
    # Perform motion planning based on key ee_poses
    motion_plan_success = motion_plan(
        task_name=task_name,
        initial_poses=initial_poses,
        actions=ee_poses_original,
        variation=variation,
    )
    results.append({
        'variant': 'original',
        'threshold': 0.1,
        'Replay': replay_success,
        'MotionPlan': motion_plan_success,
        'keypoints': len(original_keypoints),
    })
    '''

    # Smooth original actions
    smoothed_actions = smooth_joint_positions(actions, window_length=window_length, polyorder=polyorder)
    replay_success, smoothed_demo = replay_demo(
        task_name=task_name,
        initial_poses=initial_poses,
        actions=smoothed_actions,
        variation=variation, 
    )

    if not replay_success:
        print("Smoothed replay unsuccessful")
        return [], []

    # Extract smoothed keypoints with threshold = 0.1
    detection_threshold = 0.1
    smoothed_keypoints = keypoint_discovery(smoothed_demo, stopping_delta=detection_threshold, method='heuristic')
    keypoint_counts['smoothed_0.1'] = len(smoothed_keypoints)
    ee_poses_smoothed = extract_ee_poses(smoothed_demo, smoothed_keypoints)

    # Motion plan with 0.1 smoothed ee_poses
    motion_plan_success = motion_plan(
        task_name=task_name,
        initial_poses=initial_poses,
        actions=ee_poses_smoothed,
        variation=variation,
    )

    results.append({
        'variant': 'smoothed (0.1)',
        'threshold': detection_threshold,
        'Replay': replay_success,
        'MotionPlan': motion_plan_success,
        'keypoints': len(smoothed_keypoints),
    })

    # Extract smoothed keypoints with threshold = 0.25
    detection_threshold = 0.25
    smoothed_keypoints = keypoint_discovery(smoothed_demo, stopping_delta=detection_threshold, method='heuristic')
    keypoint_counts['smoothed_0.25'] = len(smoothed_keypoints)
    ee_poses_smoothed = extract_ee_poses(smoothed_demo, smoothed_keypoints)

    # Motion plan with 0.1 smoothed ee_poses
    motion_plan_success = motion_plan(
        task_name=task_name,
        initial_poses=initial_poses,
        actions=ee_poses_smoothed,
        variation=variation,
    )

    results.append({
        'variant': 'smoothed (0.25)',
        'threshold': detection_threshold,
        'Replay': replay_success,
        'MotionPlan': motion_plan_success,
        'keypoints': len(smoothed_keypoints),
    })

    return results, keypoint_counts

def main():
    """Main function to generate smoothed demo data for RVT/PerAct"""
    parser = argparse.ArgumentParser(description='Generate smoothed demo in RVT/PerAct format')
    parser.add_argument('--task_name', type=str, required=True, 
                       help='RLBench task name')
    parser.add_argument('--window_length', type=int, required=True,
                       help='Savitzky-Golay filter window length')
    parser.add_argument('--polyorder', type=int, required=True,
                       help='Savitzky-Golay filter polynomial order')
    args = parser.parse_args()

    task_name = args.task_name
    window_length = args.window_length
    polyorder = args.polyorder

    trials = 0
    episode_num = 0
    num_of_episodes = 10 # number of episodes wanted to create     
    all_results = []
    all_keypoint_counts = []

    motion_plan_success = {
        'original': 0,
        'smoothed (0.1)': 0,
        'smoothed (0.25)': 0
    }

    while episode_num < num_of_episodes: 
        results, keypoint_counts = run_single_demo(task_name, window_length, polyorder, episode_num)
        if not results:
            trials += 1 
            continue
        all_results.append(results)
        all_keypoint_counts.append(keypoint_counts)
        print(results)
        for result in results: 
            if result['MotionPlan'] is True:
                motion_plan_success[result['variant']] += 1    
        episode_num += 1
        trials += 1
    
    # Calculate average keypoints
    avg_keypoints = {
        'original': sum(kc['original'] for kc in all_keypoint_counts) / num_of_episodes,
        'smoothed_0.1': sum(kc['smoothed_0.1'] for kc in all_keypoint_counts) / num_of_episodes,
        'smoothed_0.25': sum(kc['smoothed_0.25'] for kc in all_keypoint_counts) / num_of_episodes,
    }
    
    # Calculate success rates
    success_rates = {
        variant: (count / episode_num) * 100 
        for variant, count in motion_plan_success.items()
    }

    # Print final summary
    print("\n" + "="*50)
    print("FINAL STATISTICS SUMMARY")
    print("="*50)
    
    print(f"\nAverage Keypoints Detected (over {episode_num} demos):")
    print(f"  Original (0.1):       {avg_keypoints['original']:.2f}")
    print(f"  Smoothed (0.1):       {avg_keypoints['smoothed_0.1']:.2f}")
    print(f"  Smoothed (0.25):       {avg_keypoints['smoothed_0.25']:.2f}")

    print(f"\nMotion Planning Success Rates (over {episode_num} demos):")
    print(f"  Original:             {success_rates['original']:.1f}%")
    print(f"  Smoothed (0.1):       {success_rates['smoothed (0.1)']:.1f}%")
    print(f"  Smoothed (0.25):       {success_rates['smoothed (0.25)']:.1f}%")
    print(f"  Number for trials:     {trials}")

if __name__ == "__main__":
    main()

def smooth_ee_positions_partial(actions, window_length, polyorder, keypoints, radius): 
    """Apply Savitzky-Golay filter to smooth joint positions"""
    # Ensure window_length is odd and valid
    if window_length >= len(actions):
        window_length = len(actions) - 1 if len(actions) % 2 == 0 else len(actions) - 2
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length - 1)

    # Create smoothed velocities array (copy of original)
    smoothed_actions = actions.copy()
    joint_positions = actions[:, ]



    joint_positions = actions[:, :7]  # First 7 columns are joint positions
    t = len(actions)
    
    for keypoint in keypoints: 
        # Define local smoothing window around the keyframe 
        start = max(0, keypoint  - radius)
        end = min(t, keypoint + radius + 1)

        if end-start < window_length:
            print('Not enough points to smooth')
            continue 

        local_segment = joint_positions[start:end]
        smoothed_segment = savgol_filter(local_segment, window_length, polyorder, axis=0)
        smoothed_actions[start:end, :7] = smoothed_segment

    return smoothed_actions