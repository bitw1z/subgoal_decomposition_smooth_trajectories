import os
import glob
import time
import pickle
import argparse
import random
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from pyrep.objects.shape import Shape
from rlbench.tasks import PushButtons
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from pyrep.objects.proximity_sensor import ProximitySensor

task_name = 'PushButtons'

# Set up action mode
action_mode = MoveArmThenGripper(
   arm_action_mode=JointVelocity(), 
   gripper_action_mode=Discrete()
)

# Set up observation configuration
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.gripper_pose = True
obs_config.joint_velocities = True
obs_config.joint_positions = True
obs_config.front_camera.rgb = True
obs_config.gripper_open = True

# Initialize environment
env = Environment(action_mode,
                  obs_config=obs_config,
                  headless=False,
                 )
env.launch() 

# Initialize task environment
task_env = env.get_task(PushButtons)
variation_idx = 2
task_env.set_variation(variation_idx) 
task_env.reset()  

demos = task_env.get_demos(amount=1, live_demos=True)
demo = demos[0]

task_name = 'PushButtons'
save_dir = f'./demo_data/{task_name}/visualizations'
os.makedirs(save_dir, exist_ok=True)

joint_velocities = np.array([obs.joint_velocities for obs in demo])       # Shape: (T, 7)
joint_positions = np.array([obs.joint_positions for obs in demo])         # Shape: (T, 7)
ee_positions = np.array([obs.gripper_pose[:3] for obs in demo])  
video_frames = [obs.front_rgb for obs in demo]

timesteps = np.arange(len(demo))

# ========== 1. Joint Velocities ==========
plt.figure(figsize=(10, 6))
for i in range(7):
    plt.plot(timesteps, joint_velocities[:, i], label=f'Joint {i+1}')
plt.title('Joint Velocities Over Time')
plt.xlabel('Timestep')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'joint_velocities.png'), dpi=300)
plt.close()

# ========== 2. End-Effector Positions ==========
plt.figure(figsize=(10, 6))
axes = ['x', 'y', 'z']
for i in range(3):
    plt.plot(timesteps, ee_positions[:, i], label=f'{axes[i]}')
plt.title('End-Effector Position Over Time')
plt.xlabel('Timestep')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ee_position.png'), dpi=300)
plt.close()

# ========== 3. Joint Positions ==========
plt.figure(figsize=(10, 6))
for i in range(7):
    plt.plot(timesteps, joint_positions[:, i], label=f'Joint {i+1}')
plt.title('Joint Positions Over Time')
plt.xlabel('Timestep')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'joint_positions.png'), dpi=300)
plt.close()

print(f"[✓] Saved visualizations to: {save_dir}")

video_path = os.path.join(save_dir, f'video.mp4')
imageio.mimsave(video_path, video_frames, fps=30)

env.shutdown()