import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rdp import rdp

# --- Load trajectory (first 3 columns = X,Y,Z) ---
arr = np.load('/home/jaehyukchang/RLBench-peract (JointVelocity)/demo_data/SweepToDustpanOfSize/all_variations/episodes/episode0/original_ee_actions.npy')

pos = arr[:, :3].astype(float)

'''
# Apply RDP (epsilon controls simplification tolerance)
epsilon = 0.03  # increase for more simplification
pos_simplified = np.array(rdp(pos.tolist(), epsilon=epsilon))

x, y, z = pos_simplified[:, 0], pos_simplified[:, 1], pos_simplified[:, 2]
'''
x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

# --- Keypoints as timestep indices (choose one of the two lines) ---
key_idx = [48, 57, 78, 106]          # if you have them in Python
# key_idx = np.load('keypoints.npy')      # if saved as a 1D numpy array

# (Optional) safety clamp to valid range
key_idx = np.array(key_idx, dtype=int)
key_idx = key_idx[(key_idx >= 0) & (key_idx < len(x))]

# --- Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='EE Trajectory')
ax.scatter(x[0], y[0], z[0], s=60, c='green', label='Start')
ax.scatter(x[-1], y[-1], z[-1], s=60, c='red',   label='End')

# Keypoints on the curve
ax.scatter(x[key_idx], y[key_idx], z[key_idx], s=60, c='orange', label='Keypoints')

ax.set_title('End-Effector Trajectory (3D)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend(); ax.grid(True)

# Make the 3D box cubic (nice proportions)
try: ax.set_box_aspect([1,1,1])
except: pass

plt.show()
