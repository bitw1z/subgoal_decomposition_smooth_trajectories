
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth_ee_positions(actions, window_length, polyorder): 
    """Apply Savitzky-Golay filter to smooth end-effector positions"""
    # Ensure window_length is odd and valid
    if window_length >= len(actions):
        window_length = len(actions) - 1 if len(actions) % 2 == 0 else len(actions) - 2
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length - 1)

    # Create smoothed actions array (copy of original)
    smoothed_actions = actions.copy()
    ee_positions = actions[:, :3]

    smoothed_positions = savgol_filter(ee_positions, window_length, polyorder, axis=0)
    smoothed_actions[:, :3] = smoothed_positions

    return smoothed_actions

def smooth_joint_velocities(actions, window_length, polyorder): 
    """Apply Savitzky-Golay filter to smooth end-effector positions"""
    # Ensure window_length is odd and valid
    if window_length >= len(actions):
        window_length = len(actions) - 1 if len(actions) % 2 == 0 else len(actions) - 2

    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length - 1)

    # Create smoothed actions array (copy of original)
    joint_velocities = actions[:, :7]  # First 7 columns are joint velocities    

    # Apply Savitzky-Golay filter
    smoothed_velocities = savgol_filter(
        joint_velocities, 
        window_length=window_length,
        polyorder=polyorder, 
        axis=0
    )

    smoothed_actions = actions.copy()
    smoothed_actions[:, :7] = smoothed_velocities

    return smoothed_actions, window_length, polyorder

def extract_key_actions(actions, keypoints): 
    """Extract actions at specified keypoint indices"""
    keypoints = np.asarray(keypoints)

    if np.any(keypoints >= len(actions)) or np.any(keypoints < 0):
        raise ValueError(f"Invalid keypoints: {keypoints}. "
                         f"Action length is {len(actions)}.")

    return actions[keypoints]

def visualize_velocity(demo, demo_name):
    """Visualize joint velocities of every joint over time."""
    velocities = [obs.joint_velocities for obs in demo]  # List of arrays

    velocities = np.array(velocities)  # Convert to shape (T, num_joints)
    timesteps = np.arange(len(velocities))
    num_joints = velocities.shape[1]

    plt.figure(figsize=(12, 6))
    for joint_idx in range(num_joints):
        plt.plot(timesteps, velocities[:, joint_idx], label=f'Joint {joint_idx + 1}')

    plt.title('Joint Velocities Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(loc='upper right', ncol=4)
    plt.grid(True)
    plt.ylim(-0.7, 0.7)
    plt.tight_layout()
    plt.savefig(f'./{demo_name}', dpi=300)
    plt.show()

def extract_ee(demo, actions): 
    ee_actions = []
    for idx, obs in enumerate(demo):
        pose = obs.gripper_pose.flatten()        # shape (7,)
        gripper = np.array([actions[idx][-1]])   # shape (1,)
        action = np.concatenate([pose, gripper]) # shape (8,)
        ee_actions.append(action)

    return np.array(ee_actions)