import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ee_utils import * 
from demo_utils import *
from pipeline_utils import *
from rlbench.backend.const import *
from peract_keypoint import keypoint_discovery

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
    
    attempt_count = 0
    successful_demos = 0 
    demo_num = 50
    previous_variation = -1
    num_of_episodes = 50

    output_dir = f'./demo_data/{task_name}'
    os.makedirs(output_dir, exist_ok=True)

    failure_log = []
    all_results = []
    all_keypoints = []

    while successful_demos < num_of_episodes: 
        print(f"\n{'='*20} ATTEMPTING DEMO {attempt_count} {'='*20}")
        print(f"Successful demos so far: {successful_demos}/{num_of_episodes}")
        prev_failure_count = len(failure_log)

        success, failure_log, original_actions, initial_poses, original_keypoints, motion_plan_success_original, variation, video_dir, descriptions = original_demo_phase(
            task_name, demo_num, failure_log, output_dir, previous_variation
        )

        if len(failure_log) > prev_failure_count:
            attempt_count += 1
            continue
        
        smoothed_demo, smoothed_actions, failure_log, smoothing_factors = find_best_smooth_factors(
            task_name, demo_num, failure_log, output_dir, video_dir, initial_poses, original_actions, variation, window_length, polyorder
        ) 

        if len(failure_log) > prev_failure_count:
            attempt_count += 1
            continue

        motion_plan_result, failure_log = find_best_detection_threshold(
            task_name, demo_num, failure_log, output_dir, video_dir, initial_poses, smoothed_demo, smoothed_actions, variation, window_length, polyorder, descriptions
        )

        if len(failure_log) > prev_failure_count:
            attempt_count += 1
            continue

        result = {
            'demo_num': demo_num, 
            'variation': int(variation), 
            'polyorder': smoothing_factors['polyorder'], 
            'window_length': smoothing_factors['window_length'], 
            'detection_threshold': motion_plan_result['detection_threshold'],
            'motion_plan_success_original': motion_plan_success_original,         
            'motion_plan_success_every': motion_plan_result['motion_plan_success_every'],
            'motion_plan_success_smoothed_01': motion_plan_result['motion_plan_success_smoothed_01'],
            'motion_plan_success_smoothed_best': motion_plan_result['motion_plan_success_smoothed_best'],
        }
        all_results.append(result)
        keypoints = {
            'demo_num': demo_num, 
            'variation': int(variation), 
            'original_keypoints': original_keypoints, 
            'smoothed_keypoints_01': motion_plan_result['smoothed_keypoints_01'], 
            'smoothed_keypoints_best': motion_plan_result['smoothed_keypoints_best'], 
        }
        all_keypoints.append(keypoints)

        print(result)
        print(keypoints)

        attempt_count += 1
        successful_demos += 1 
        previous_variation += 1
        demo_num += 1

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(output_dir, 'failure_log.json'), 'w') as f:
        json.dump(failure_log, f, indent=2)

    save_keypoints_txt(all_keypoints, os.path.join(output_dir, 'all_keypoints.txt'))

    motion_plan_success_original_avg = 0 
    motion_plan_success_every_avg = 0 
    motion_plan_success_smoothed_01_avg = 0 
    motion_plan_success_smoothed_best_avg = 0 

    for result in all_results: 
        motion_plan_success_original_avg += result['motion_plan_success_original']
        motion_plan_success_every_avg += result['motion_plan_success_every']
        motion_plan_success_smoothed_01_avg += result['motion_plan_success_smoothed_01']
        motion_plan_success_smoothed_best_avg += result['motion_plan_success_smoothed_best']

    original_keypoints_avg = 0 
    smoothed_keypoints_01_avg = 0 
    smoothed_keypoints_best_avg = 0 

    for keypoint in all_keypoints: 
        original_keypoints_avg += len(keypoint['original_keypoints'])
        smoothed_keypoints_01_avg += len(keypoint['smoothed_keypoints_01'])
        smoothed_keypoints_best_avg += len(keypoint['smoothed_keypoints_best'])

    final_result = {
        'mp_original': motion_plan_success_original_avg/num_of_episodes,
        'mp_every': motion_plan_success_every_avg/num_of_episodes,
        'mp_smooth_01': motion_plan_success_smoothed_01_avg/num_of_episodes,
        'mp_smooth_best': motion_plan_success_smoothed_best_avg/num_of_episodes,
        'keypoints_original': original_keypoints_avg/num_of_episodes, 
        'smoothed_keypoints_01_avg': smoothed_keypoints_01_avg/num_of_episodes, 
        'smoothed_keypoints_best_avg': smoothed_keypoints_best_avg/num_of_episodes, 
    }

    with open(os.path.join(output_dir, 'final_result.json'), 'w') as f:
        json.dump(final_result, f, indent=2)

if __name__ == "__main__":
    main()