import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ee_utils import * 
from demo_utils import *
from rlbench.backend.const import *
from peract_keypoint import keypoint_discovery

def original_demo_phase(task_name, demo_num, failure_log, output_dir, previous_variation): 
    """
    Launch Original Demo
    Extract End Effector Positions 
    Replay without any change 
    Motion Plan with Keypoints (threshold = 0.1)
    """
    result = {}
    episode_path, video_dir = setup_episode_path(output_dir, task_name, demo_num)
 
    # Launch demo 
    try:
        demo, initial_poses, actions, variation, descriptions = generate_and_record_demo(
            task_name=task_name, 
            previous_variation=previous_variation,
            video_dir=video_dir,
        )
    except Exception as e:
        print(f"Error while generating demo for task {task_name}: {e}")
        failure_log.append({
            'demo': demo_num,
            'stage': 'Generate Demo',
            'reason': 'Launching Demo Failed.'
        })
        return False, failure_log, None, None, None, None, None, None, None

    if not demo: 
        failure_log.append({
            'demo': demo_num,
            'stage': 'Generate Demo',
            'reason': 'Demo was not created correctly. Skip this demo.'
        })
        return False, failure_log, None, None, None, None, None, None, None
    # Replay demo 
    try:
        replay_success, _ = replay_demo(
            task_name=task_name,
            initial_poses=initial_poses,
            actions=actions,
            variation=variation, 
            video_dir=video_dir,
            stage='original_replay', 
        )
    except Exception as e:
        print(f"Error while replaying demo for task {task_name}: {e}")
        failure_log.append({
            'demo': demo_num,
            'stage': 'Original Replay',
            'reason': 'Launching Replay Failed'
        })
        return False, failure_log, None, None, None, None, None, None, None

    if not replay_success:
        print(f'Demo{demo_num}: Original replay failed. Skipping this demo.')
        failure_log.append({
            'demo': demo_num,
            'stage': 'Original Replay',
            'reason': 'Unsuccessful Replay'
        })
        return False, failure_log, None, None, None, None, None, None, None

     # Extract original demo keypoints
    keypoints = keypoint_discovery(demo, stopping_delta=0.1, method='heuristic') 
    ee_actions = extract_ee(demo, actions)
    keypoint_actions = extract_key_actions(ee_actions, keypoints)

    # Motion plan with original keypoints actions 
    try:
        motion_plan_success, result = motion_plan(
            task_name=task_name,
            initial_poses=initial_poses,
            actions=keypoint_actions,
            variation=variation, 
            video_dir=video_dir,
            stage = 'motion_plan_original_01',
        )
    except Exception as e:
        print(f"Error while motion planning demo for task {task_name}: {e}")
        failure_log.append({
            'demo': demo_num,
            'stage': 'Motion Plan Original',
            'reason': 'Unsuccessful Motion Planning'
        })
        return False, failure_log, None, None, None, None, None, None, None

    episode_path, video_dir = setup_episode_path(output_dir, task_name, demo_num)
    np.save(os.path.join(episode_path, 'original_ee_actions.npy'), np.array(ee_actions))

    return True, failure_log, actions, initial_poses, keypoints, motion_plan_success, variation, video_dir, descriptions

def find_best_smooth_factors(task_name, demo_num, failure_log, output_dir, video_dir, initial_poses, original_actions, variation, window_length, polyorder):
    """
    Find the best smoothing factors that work well per demo
    """

    while window_length > polyorder and window_length >= 3: 
        try:
            print(f'Trying window_length = {window_length} and polyorder = {polyorder}')
            smoothed_actions, window_length, polyorder = smooth_joint_velocities(original_actions, window_length, polyorder)
            smoothed_replay_success, smoothed_demo = replay_demo(
                task_name=task_name,
                initial_poses=initial_poses,
                actions=smoothed_actions,
                variation=variation, 
                video_dir=video_dir,
                stage='smoothed_replay', 
            )
            if smoothed_replay_success:
                print(f'Smoothed replay successful: window_length={window_length}, polyorder={polyorder}')
                smoothing_factors = {
                    'window_length': window_length, 
                    'polyorder': polyorder, 
                }
                return smoothed_demo, smoothed_actions, failure_log, smoothing_factors

        except Exception as e:
            print(f"Error with window_length={window_length}, polyorder={polyorder}: {e}")

        if polyorder == 2:
            polyorder = 3
        else:  # polyorder == 3
            polyorder = 2
            window_length -= 2

     # If we exit the loop, no parameters worked
    failure_log.append({
        'demo': demo_num,
        'stage': 'Smoothing',
        'reason': 'No valid smoothing parameters found'
    })
    
    return None, None, failure_log, None

def find_best_detection_threshold(task_name, demo_num, failure_log, output_dir, video_dir, initial_poses, smoothed_demo, smoothed_actions, variation, window_length, polyorder, descriptions):
    # Motion plan with smoothed keypoint actions every point
    smoothed_ee_actions = extract_ee(smoothed_demo, smoothed_actions)
    
    try:
        motion_plan_success_every, result = motion_plan(
            task_name=task_name,
            initial_poses=initial_poses,
            actions=smoothed_ee_actions,
            variation=variation, 
            video_dir=video_dir,
            stage = 'motion_plan_success_every',
        )
    except Exception as e:
        print(f"Error while motion planning demo for task {task_name}: {e}")
        failure_log.append({
            'demo': demo_num,
            'stage': 'Motion Plan Every',
            'reason': 'Unsuccessful Motion Planning'
        })
        return None, failure_log

    if not result: 
        print('Motion Plan went wrong. Skip this demo')
        failure_log.append({
            'demo': demo_num,
            'stage': 'Motion Plan Every',
            'reason': 'Unsuccessful Motion Planning'
        })
        return None, failure_log

    smoothed_keypoints_01 = keypoint_discovery(smoothed_demo, stopping_delta=0.10, method='heuristic') 
    keypoint_actions_smooth_01 = extract_key_actions(smoothed_ee_actions, smoothed_keypoints_01)
    
    # Motion plan with smoothed keypoint actions (threshold = 0.1)
    try:
        motion_plan_success_smoothed_01, result = motion_plan(
            task_name=task_name,
            initial_poses=initial_poses,
            actions=keypoint_actions_smooth_01,
            variation=variation, 
            video_dir=video_dir,
            stage = 'motion_plan_smoothed_01',
        )
    except Exception as e:
        print(f"Error while motion planning demo for task {task_name}: {e}")
        failure_log.append({
            'demo': demo_num,
            'stage': 'Motion Plan Smooth',
            'reason': 'Unsuccessful Motion Planning'
        })
        return None, failure_log
    
    if not result: 
        print('Motion Plan went wrong. Skip this demo')
        failure_log.append({
            'demo': demo_num,
            'stage': 'Motion Plan 0.1',
            'reason': 'Unsuccessful Motion Planning'
        })
        return None, failure_log

    if motion_plan_success_smoothed_01: 
        motion_plan_result = {
                    'detection_threshold': 0.1, 
                    'motion_plan_success_every': motion_plan_success_every,
                    'motion_plan_success_smoothed_01': motion_plan_success_smoothed_01, 
                    'smoothed_keypoints_01': smoothed_keypoints_01, 
                    'motion_plan_success_smoothed_best': motion_plan_success_smoothed_01, 
                    'smoothed_keypoints_best': smoothed_keypoints_01, 
                }
        episode_path, video_dir = setup_episode_path(output_dir, task_name, demo_num)
        save_demo(smoothed_demo, episode_path, variation, descriptions, initial_poses, smoothed_ee_actions, 0.1)
        return motion_plan_result, failure_log

    # Now find the best threshold that is successful for motion planning
    max_threshold = 1.0 # Upper limit
    detection_threshold  = 0.15 

    while detection_threshold <= max_threshold: 
        print(f'Trying detection threshold = {detection_threshold}')
        smoothed_keypoints = keypoint_discovery(smoothed_demo, stopping_delta=detection_threshold, method='heuristic')
        keypoint_actions_smooth = extract_key_actions(smoothed_ee_actions, smoothed_keypoints)

        try:
            motion_plan_success_smoothed, result = motion_plan(
                task_name=task_name,
                initial_poses=initial_poses,
                actions=keypoint_actions_smooth,
                variation=variation, 
                video_dir=video_dir,
                stage = 'motion_plan_smoothed_best',
            )

            if not result: 
                print('Motion Plan went wrong. Skip this demo')
                failure_log.append({
                    'demo': demo_num,
                    'stage': 'Motion Plan Threshold',
                    'reason': 'Unsuccessful Motion Planning'
                })
                return None, failure_log


            if motion_plan_success_smoothed:
                print(f'Motion Plan successful: detection threshold = {detection_threshold}')
                motion_plan_result = {
                    'detection_threshold': detection_threshold, 
                    'motion_plan_success_every': motion_plan_success_every,
                    'motion_plan_success_smoothed_01': motion_plan_success_smoothed_01, 
                    'smoothed_keypoints_01': smoothed_keypoints_01, 
                    'motion_plan_success_smoothed_best': motion_plan_success_smoothed, 
                    'smoothed_keypoints_best': smoothed_keypoints, 
                }
                episode_path, video_dir = setup_episode_path(output_dir, task_name, demo_num)
                save_demo(smoothed_demo, episode_path, variation, descriptions, initial_poses, smoothed_ee_actions, detection_threshold)
                return motion_plan_result, failure_log

        except Exception as e:
            print(f"Error while motion planning demo for task {task_name}: {e}")
        
        # Increment threshold for next iteration
        detection_threshold += 0.05
        detection_threshold = round(detection_threshold , 2)
    
    # If we exit the loop, no threshold worked
    failure_log.append({
        'demo': demo_num,
        'stage': 'Motion Plan Smooth',
        'reason': f'Motion Planning failed for all thresholds up to {max_threshold}'
    })

    return None, failure_log