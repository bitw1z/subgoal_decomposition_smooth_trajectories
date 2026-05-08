# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES.
# Dataset construction for "Goal-conditioned dp3"

import os
import clip
import json
import torch
import pickle
import numpy as np
from typing import List

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
import torch.distributed as dist

from rvt.utils.peract_utils import IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.utils import extract_obs

# --- RVT utilities (ONLY for preprocessing + PC extraction) ---
import rvt.utils.peract_utils as peract_utils
import rvt.utils.rvt_utils as rvt_utils

from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_debug_goal_pcd(scene_pc, cur_gripper_pcd, goal_gripper_pcd, title="debug"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # scene (light, many points)
    ax.scatter(
        scene_pc[:, 0], scene_pc[:, 1], scene_pc[:, 2],
        s=1, alpha=0.1
    )

    # current gripper (blue)
    ax.scatter(
        cur_gripper_pcd[:, 0],
        cur_gripper_pcd[:, 1],
        cur_gripper_pcd[:, 2],
        s=80
    )

    # goal gripper (red)
    ax.scatter(
        goal_gripper_pcd[:, 0],
        goal_gripper_pcd[:, 1],
        goal_gripper_pcd[:, 2],
        s=80
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # equal aspect ratio (IMPORTANT)
    max_range = (scene_pc.max(axis=0) - scene_pc.min(axis=0)).max() / 2.0
    mid = scene_pc.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()   # <-- this blocks execution

def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    replay_size=3e5,
):
    """
    Creates a replay buffer suitable for DP3.
    """

    N_OBS_STEPS = 2

    lang_emb_dim = 512
    max_token_seq_len = 77  

    observation_elements = []

    observation_elements.extend([
        # low_dim_state can be removed if unused, or left as-is
        ObservationElement("low_dim_state", (N_OBS_STEPS, 4), np.float32),

        # IMPORTANT: these must become histories
        ObservationElement("gripper_pose", (N_OBS_STEPS, 7), np.float32),
        ObservationElement("gripper_open", (N_OBS_STEPS, 1), np.float32),
        ObservationElement("goal_gripper_pose", (7,), np.float32),
        ObservationElement("goal_gripper_open", (1,), np.float32),

    ])

    observation_elements.extend([
        ReplayElement(
            "lang_goal_embs",
            (max_token_seq_len, lang_emb_dim),
            np.float32,
        ),
        ReplayElement(
            "lang_goal",
            (1,),
            object,
        ),
    ])

    # --------------------------------------------------------
    # Visual observations (unchanged from RVT)
    # --------------------------------------------------------
    for cname in cameras:
        observation_elements.extend(
            [
                ObservationElement(f"{cname}_rgb", (N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32),
                ObservationElement(f"{cname}_depth", (N_OBS_STEPS, 1, IMAGE_SIZE, IMAGE_SIZE), np.float32),
                ObservationElement(f"{cname}_point_cloud", (N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32),
                ObservationElement(f"{cname}_camera_extrinsics", (N_OBS_STEPS, 4, 4), np.float32),
                ObservationElement(f"{cname}_camera_intrinsics", (N_OBS_STEPS, 3, 3), np.float32),
            ]
        )

    # --------------------------------------------------------
    # Minimal metadata
    # --------------------------------------------------------
    extra_replay_elements = [
        ReplayElement("episode_idx", (), int),
    ]

    replay_buffer = UniformReplayBuffer(
        disk_saving=disk_saving,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(16, 8),       # action sequence: (H=16, 8)
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
    )

    return replay_buffer


# ------------------------------------------------------------
# Continuous EE action extraction
# ------------------------------------------------------------
def _get_continuous_action(obs_tp1: Observation):
    """
    Build the continuous DP3 action target from the next observation (t+1).

    We supervise DP3 with an 8D end-effector command:
        [x, y, z, qx, qy, qz, qw, gripper]

    Notes:
    - We use the *next* gripper pose (obs_{t+1}) as the target action a_t.
    - Quaternions have a sign ambiguity (q and -q represent the same rotation).
      We canonicalize by enforcing qw >= 0 to avoid discontinuities in regression.
    """
    quat = obs_tp1.gripper_pose[3:]
    if quat[-1] < 0:
        quat = -quat
    grip = float(obs_tp1.gripper_open)
    return np.concatenate([obs_tp1.gripper_pose[:3], quat, np.array([grip])])


def correct_obs_format(obs):
    """
    Removes unused mask fields to avoid RLBench inconsistencies.
    """
    for cam in ['overhead']:
        setattr(obs, f'{cam}_rgb', None)
        setattr(obs, f'{cam}_depth', None)
        setattr(obs, f'{cam}_point_cloud', None)
        setattr(obs, f'{cam}_mask', None)

    for cam in ['left_shoulder', 'right_shoulder', 'front', 'wrist']:
        setattr(obs, f'{cam}_mask', None)

    return obs

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

class RunningStats:
    """
    Tracks per-dimension count / sum / sumsq / min / max.
    Assumes input x has feature dimension on the last axis.
    All leading dimensions are flattened into samples.
    """
    def __init__(self, name=""):
        self.name = name
        self.count = 0
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim >= 1, f"{self.name}: expected ndim >= 1, got {x.ndim}"
        feat_dim = x.shape[-1]
        x_flat = x.reshape(-1, feat_dim)  # (N, D)

        if x_flat.shape[0] == 0:
            return

        batch_sum = x_flat.sum(axis=0)
        batch_sumsq = (x_flat ** 2).sum(axis=0)
        batch_min = x_flat.min(axis=0)
        batch_max = x_flat.max(axis=0)
        batch_count = x_flat.shape[0]

        if self.sum is None:
            self.sum = batch_sum
            self.sumsq = batch_sumsq
            self.min = batch_min
            self.max = batch_max
        else:
            self.sum += batch_sum
            self.sumsq += batch_sumsq
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)

        self.count += batch_count

    def as_dict(self, eps: float = 1e-6):
        if self.count == 0:
            raise RuntimeError(f"{self.name}: no samples collected.")

        mean = self.sum / self.count
        var = self.sumsq / self.count - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std = np.maximum(std, eps)

        out = {
            "count": int(self.count),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
        }
        return out


class DP3StatsAccumulator:
    """
    Collect stats for the exact DP3 inputs:
      - agent_pos: (T_obs, 8)
      - point_cloud: (T_obs, N, 3)
      - lang: (T_obs, 512)
      - action: (H, 8)
    """
    def __init__(self):
        self.stats = {
            "agent_pos": RunningStats("agent_pos"),
            "point_cloud": RunningStats("point_cloud"),
            "gripper_pcd": RunningStats("gripper_pcd"),
            "goal_gripper_pcd": RunningStats("goal_gripper_pcd"),
            "lang": RunningStats("lang"),
            "action": RunningStats("action"),
        }

    def update(self, agent_pos, point_cloud, gripper_pcd, goal_gripper_pcd, lang, action):
        self.stats["agent_pos"].update(agent_pos)
        self.stats["point_cloud"].update(point_cloud)
        self.stats["gripper_pcd"].update(gripper_pcd)
        self.stats["goal_gripper_pcd"].update(goal_gripper_pcd)
        self.stats["lang"].update(lang)
        self.stats["action"].update(action)

    def state_dict(self):
        return {k: v.as_dict() for k, v in self.stats.items()}

def quat_to_rotmat_np(q):
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def make_gripper_template_4_np():
    left_finger = np.array([0.0, -0.0405, 0.0800], dtype=np.float32)
    right_finger = np.array([0.0,  0.0405, 0.0800], dtype=np.float32)
    wrist = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center_contact = 0.5 * (left_finger + right_finger)
    template = np.stack([left_finger, right_finger, wrist, center_contact], axis=0)
    assert template.shape == (4, 3)
    return template

def build_gripper_pcd_from_pose_np(pose_7, open_1, template_4):
    """
    pose_7: (N,7)  -> [x, y, z, qx, qy, qz, qw]
    open_1: (N,1)  -> gripper openness in [0,1]
    template_4: (4,3) in gripper-local frame

    Point order:
        0: left finger
        1: right finger
        2: wrist
        3: center contact

    returns:
        (N,4,3) in world frame
    """

    assert pose_7.ndim == 2 and pose_7.shape[1] == 7, pose_7.shape
    assert open_1.ndim == 2 and open_1.shape[1] == 1, open_1.shape
    assert template_4.shape == (4, 3), template_4.shape

    N = pose_7.shape[0]

    # ----------------------------------------
    # Extract translation and rotation
    # ----------------------------------------
    t = pose_7[:, :3]           # (N,3)
    q = pose_7[:, 3:7]          # (N,4)
    R = quat_to_rotmat_np(q)    # (N,3,3)

    # ----------------------------------------
    # Copy template into batch
    # ----------------------------------------
    p = np.repeat(template_4[None, :, :], repeats=N, axis=0).copy()   # (N,4,3)

    # ----------------------------------------
    # Apply gripper opening (LOCAL frame)
    # ----------------------------------------
    open_amt = np.clip(open_1[:, 0], 0.0, 1.0)    # (N,)

    # same convention as before: template centered at open=0.5
    delta = 0.021 * (open_amt - 0.5)              # (N,)

    # move fingers along local y-axis
    p[:, 0, 1] -= delta   # left finger
    p[:, 1, 1] += delta   # right finger

    # recompute center contact (important!)
    p[:, 3, :] = 0.5 * (p[:, 0, :] + p[:, 1, :])

    # ----------------------------------------
    # Transform to world frame
    # p_world = R * p_local + t
    # ----------------------------------------
    p_world = np.einsum("bij,bpj->bpi", R, p) + t[:, None, :]

    return p_world.astype(np.float32)

def farthest_point_sampling_np(xyz, n_samples):
    """
    xyz: (N,3)
    returns: (n_samples,)
    """
    N = xyz.shape[0]
    centroids = np.zeros((n_samples,), dtype=np.int64)
    distances = np.full((N,), 1e10, dtype=np.float32)
    farthest = 0

    for i in range(n_samples):
        centroids[i] = farthest
        centroid_xyz = xyz[farthest:farthest+1]
        dist = np.sum((xyz - centroid_xyz) ** 2, axis=-1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return centroids

def _build_goal_conditioned_dp3_inputs_for_stats(
    obs_dict,
    cameras,
    goal_gripper_pose,
    goal_gripper_open,
    template_4,
):
    """
    Build the exact goal-conditioned DP3 inputs for stats:
      - point_cloud: (2, 1024, 3)
      - goal_gripper_pcd: (2, 4, 3)

    This mirrors the agent preprocessing path:
      obs_bt -> peract_utils._preprocess_inputs -> rvt_utils.get_pc_img_feat
      -> FPS scene -> concat current gripper pcd
    """
    T_obs = obs_dict["gripper_pose"].shape[0]
    assert T_obs == 2, T_obs

    # -------------------------------------------------
    # Current gripper history
    # -------------------------------------------------
    gp_hist = np.asarray(obs_dict["gripper_pose"], dtype=np.float32)   # (2,7)
    go_hist = np.asarray(obs_dict["gripper_open"], dtype=np.float32)   # (2,1)

    cur_gripper_pcd = build_gripper_pcd_from_pose_np(
        gp_hist, go_hist, template_4
    )   # (2,4,3)

    # -------------------------------------------------
    # Scene point cloud per timestep using SAME path as agent
    # -------------------------------------------------
    scene_pc_t = []

    for t in range(T_obs):
        cam_points = []

        for cam in cameras:
            pc = np.asarray(obs_dict[f"{cam}_point_cloud"][t], dtype=np.float32)  # (3,H,W)
            assert pc.ndim == 3 and pc.shape[0] == 3, f"{cam}_point_cloud has shape {pc.shape}"

            pc_flat = np.transpose(pc, (1, 2, 0)).reshape(-1, 3)  # (H*W, 3)
            cam_points.append(pc_flat)

        pc_cat = np.concatenate(cam_points, axis=0)  # (num_cams * H * W, 3)
        scene_pc_t.append(pc_cat.astype(np.float32))

    scene_pc = np.stack(scene_pc_t, axis=0)  # (2, N, 3)

    # -------------------------------------------------
    # FPS scene per timestep
    # -------------------------------------------------
    scene_1024 = []
    for t in range(T_obs):
        idx = farthest_point_sampling_np(scene_pc[t], 1024)
        scene_1024.append(scene_pc[t][idx])
    scene_1024 = np.stack(scene_1024, axis=0).astype(np.float32)   # (2,1024,3)

    # -------------------------------------------------
    # Goal gripper pcd (static goal, repeated across T_obs)
    # -------------------------------------------------
    goal_pose = np.asarray(goal_gripper_pose, dtype=np.float32)[None, :]   # (1,7)
    goal_open = np.asarray(goal_gripper_open, dtype=np.float32)[None, :]   # (1,1)

    goal_pcd = build_gripper_pcd_from_pose_np(
        goal_pose, goal_open, template_4
    )[0]   # (4,3)

    goal_gripper_pcd = np.repeat(
        goal_pcd[None, :, :], repeats=T_obs, axis=0
    ).astype(np.float32)   # (2,4,3)

    return scene_1024, cur_gripper_pcd, goal_gripper_pcd, scene_pc

def _save_dp3_stats(stats_accumulator, stats_path):
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats = stats_accumulator.state_dict()
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # optional lightweight human-readable summary
    summary_path = stats_path.replace(".pkl", "_summary.json")
    summary = {}
    for k, v in stats.items():
        rng = v["max"] - v["min"]
        summary[k] = {
            "count": v["count"],
            "dim": int(v["mean"].shape[0]),
            "min_min": float(v["min"].min()),
            "max_max": float(v["max"].max()),
            "mean_abs_mean": float(np.mean(np.abs(v["mean"]))),
            "min_std": float(v["std"].min()),
            "max_std": float(v["std"].max()),
            "num_near_zero_range_dims": int(np.sum(rng < 1e-6)),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[STATS] Saved DP3 stats to: {stats_path}")
    print(f"[STATS] Saved DP3 stats summary to: {summary_path}")

# ------------------------------------------------------------
# Fill replay with dense action sequences
# ------------------------------------------------------------
def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    cameras: List[str],
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
    action_horizon: int = 16,
    agent: str = "our",
    collect_stats: bool = False,
    stats_save_path: str = None,
):

    if replay._disk_saving and os.path.exists(task_replay_storage_folder):
        print("[Info] Replay exists. Loading from disk.")
        replay.recover_from_disk(task, task_replay_storage_folder)

        if collect_stats:
            if stats_save_path is not None and os.path.exists(stats_save_path):
                print(f"[Info] Stats file already exists: {stats_save_path}")
            else:
                print("[Warning] Replay exists but stats file was not found/generated in this run.")
        return

    else:
        if replay._disk_saving:
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    print("Filling Goal-conditioned DP3 replay (dense EE actions + future keypoint goals)...")
    H = action_horizon
    DEBUG_OBS = True

    stats_accumulator = DP3StatsAccumulator() if (collect_stats and agent == "dp3") else None
    template_4 = make_gripper_template_4_np() if stats_accumulator is not None else None

    for d_idx in range(start_idx, start_idx + num_demos):
        print(f"Filling demo {d_idx}")
        demo = get_stored_demo(data_path=data_path, index=d_idx)
        T = len(demo)

        episode_keypoints = keypoint_discovery(demo, method="rdp", episode_idx=d_idx)

        # Ensure final timestep is always a goal
        if len(episode_keypoints) == 0 or episode_keypoints[-1] != T - 1:
            episode_keypoints.append(T - 1)

        next_kp_idx = 0

        var_desc_file = os.path.join(
            data_path, episode_folder % d_idx, variation_desriptions_pkl
        )
        with open(var_desc_file, "rb") as f:
            descs = pickle.load(f)
        desc = descs[0] if isinstance(descs, (list, tuple)) else str(descs)

        if clip_model is not None:
            tokens = clip.tokenize([desc]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            with torch.no_grad():
                _, lang_embs = _clip_encode_text(clip_model, token_tensor)
            lang_goal_embs_np = lang_embs[0].float().detach().cpu().numpy()  # (77,512)
        else:
            lang_goal_embs_np = np.zeros((77, 512), dtype=np.float32)

        for t in range(T - 1):
            while (
                next_kp_idx < len(episode_keypoints)
                and t >= episode_keypoints[next_kp_idx]
            ):
                next_kp_idx += 1

            if next_kp_idx >= len(episode_keypoints):
                break

            # ---- pick two consecutive observations ----
            obs_tm1 = demo[t - 1] if t > 0 else demo[t]
            obs_t = demo[t]

            # ---- build main observation dict from obs_t ----
            obs_obj_t = correct_obs_format(obs_t)
            obs_dict = extract_obs(
                obs=obs_obj_t,
                cameras=CAMERAS,
                t=t,
                prev_action=None,
                channels_last=False,
                episode_length=T,
            )

            # ---- get RLBench low_dim_state for t-1 and t ----
            obs_obj_tm1 = correct_obs_format(obs_tm1)
            obs_dict_tm1 = extract_obs(
                obs=obs_obj_tm1,
                cameras=CAMERAS,
                t=t - 1 if t > 0 else 0,
                prev_action=None,
                channels_last=False,
                episode_length=T,
            )

            ld_tm1 = obs_dict_tm1["low_dim_state"].astype(np.float32)
            ld_t = obs_dict["low_dim_state"].astype(np.float32)

            # stack into (2,4)
            obs_dict["low_dim_state"] = np.stack([ld_tm1, ld_t], axis=0)

            gp_tm1 = obs_obj_tm1.gripper_pose.astype(np.float32)
            gp_t = obs_obj_t.gripper_pose.astype(np.float32)

            # enforce qw >= 0 consistently
            if gp_tm1[6] < 0:
                gp_tm1[3:7] = -gp_tm1[3:7]
            if gp_t[6] < 0:
                gp_t[3:7] = -gp_t[3:7]

            go_tm1 = np.array([obs_obj_tm1.gripper_open], dtype=np.float32)
            go_t = np.array([obs_obj_t.gripper_open], dtype=np.float32)

            gp_hist = np.stack([gp_tm1, gp_t], axis=0)   # (2,7)
            go_hist = np.stack([go_tm1, go_t], axis=0)   # (2,1)

            obs_dict["gripper_pose"] = gp_hist
            obs_dict["gripper_open"] = go_hist

            goal_obs = demo[episode_keypoints[next_kp_idx]]

            goal_gp = goal_obs.gripper_pose.astype(np.float32)
            goal_go = np.array([goal_obs.gripper_open], dtype=np.float32)

            if goal_gp[6] < 0:
                goal_gp[3:7] = -goal_gp[3:7]

            obs_dict["goal_gripper_pose"] = goal_gp
            obs_dict["goal_gripper_open"] = goal_go

            for cam in CAMERAS:
                for suffix in [
                    "rgb",
                    "depth",
                    "point_cloud",
                    "camera_extrinsics",
                    "camera_intrinsics",
                ]:
                    key = f"{cam}_{suffix}"
                    obs_dict[key] = np.stack(
                        [
                            obs_dict_tm1[key].astype(np.float32),
                            obs_dict[key].astype(np.float32),
                        ],
                        axis=0,
                    )

            # ------------------------------------------------
            # Action sequence
            # ------------------------------------------------
            action_seq = []
            for k in range(t, t + H):
                target_idx = k + 1
                if target_idx >= T:
                    target_idx = T - 1  # repeat final action (padding)

                a = _get_continuous_action(demo[target_idx])
                action_seq.append(a)

            action_seq = np.array(action_seq, dtype=np.float32)  # (H,8)

            terminal = (t == T - 2)
            reward = float(terminal)

            if agent == "dp3":
                obs_dict.pop("ignore_collisions", None)

            obs_dict["lang_goal_embs"] = lang_goal_embs_np
            obs_dict["lang_goal"] = np.array([desc], dtype=object)

            if DEBUG_OBS and d_idx == start_idx and t == 1:
                cam = CAMERAS[0]
                rgb = obs_dict[f"{cam}_rgb"]
                pc = obs_dict[f"{cam}_point_cloud"]
                print(f"[DEBUG] {cam}_rgb stacked shape: {rgb.shape}")
                print(f"[DEBUG] {cam}_point_cloud stacked shape: {pc.shape}")

                rgb_diff = np.mean(np.abs(rgb[1] - rgb[0]))
                pc_diff = np.mean(np.abs(pc[1] - pc[0]))
                print(f"[DEBUG] {cam} mean|rgb(t)-rgb(t-1)| = {rgb_diff:.3e}")
                print(f"[DEBUG] {cam} mean|pc(t)-pc(t-1)|  = {pc_diff:.3e}")

            if DEBUG_OBS and d_idx == start_idx and t == 0:
                print("\n" + "=" * 80)
                print("[DEBUG] Value sanity checks (non-zero / non-constant)")
                print("=" * 80)

                print("\n" + "=" * 80)
                print("[DEBUG] obs_dict contents before replay.add()")
                print("=" * 80)

                for k, v in obs_dict.items():
                    if hasattr(v, "shape"):
                        print(f"{k:25s} shape={v.shape} dtype={v.dtype}")
                    else:
                        print(f"{k:25s} type={type(v)} value={v}")

                print(f"action_seq               shape={action_seq.shape} dtype={action_seq.dtype}")
                print("=" * 80 + "\n")

                def check_array(name, x):
                    x = np.asarray(x)
                    print(
                        f"{name:25s} "
                        f"min={x.min(): .3e} "
                        f"max={x.max(): .3e} "
                        f"mean={x.mean(): .3e} "
                        f"std={x.std(): .3e}"
                    )
                    assert not np.allclose(x, 0), f"{name} is all zeros"
                    assert x.std() > 0, f"{name} has zero variance"

                check_array("lang_goal_embs", obs_dict["lang_goal_embs"])
                check_array("low_dim_state", obs_dict["low_dim_state"])
                check_array("gripper_pose", obs_dict["gripper_pose"])

                cam = CAMERAS[0]
                check_array(f"{cam}_rgb", obs_dict[f"{cam}_rgb"])
                check_array(f"{cam}_depth", obs_dict[f"{cam}_depth"])
                check_array("action_seq", action_seq)

                print("[DEBUG] ✔ All checked fields contain non-trivial values")
                print("=" * 80 + "\n")

            assert action_seq.shape == (H, 8)

            def _assert_finite(name, x):
                x = np.asarray(x)
                if not np.isfinite(x).all():
                    bad = np.argwhere(~np.isfinite(x))
                    idx = tuple(bad[0])
                    print(
                        f"\n[REPLAY DATA BAD] demo={d_idx} t={t} key={name} "
                        f"shape={x.shape} dtype={x.dtype}"
                    )
                    print(f"[REPLAY DATA BAD] first bad index={idx} value={x[idx]}")
                    raise RuntimeError("Non-finite (NaN/Inf) found while building replay.")

            # replay-stored arrays
            _assert_finite("low_dim_state", obs_dict["low_dim_state"])
            _assert_finite("gripper_pose", obs_dict["gripper_pose"])
            _assert_finite("gripper_open", obs_dict["gripper_open"])
            _assert_finite("lang_goal_embs", obs_dict["lang_goal_embs"])
            _assert_finite("action_seq", action_seq)
            _assert_finite("goal_gripper_pose", obs_dict["goal_gripper_pose"])
            _assert_finite("goal_gripper_open", obs_dict["goal_gripper_open"])

            for cam in CAMERAS:
                _assert_finite(f"{cam}_rgb", obs_dict[f"{cam}_rgb"])
                _assert_finite(f"{cam}_depth", obs_dict[f"{cam}_depth"])
                _assert_finite(f"{cam}_point_cloud", obs_dict[f"{cam}_point_cloud"])
                _assert_finite(f"{cam}_camera_extrinsics", obs_dict[f"{cam}_camera_extrinsics"])
                _assert_finite(f"{cam}_camera_intrinsics", obs_dict[f"{cam}_camera_intrinsics"])

            # ------------------------------------------------
            # Collect DP3-ready stats
            # ------------------------------------------------
            if stats_accumulator is not None:
                agent_pos = np.concatenate([gp_hist, go_hist], axis=-1).astype(np.float32)  # (2,8)

                point_cloud, gripper_pcd, goal_gripper_pcd, full_scene_pc = _build_goal_conditioned_dp3_inputs_for_stats(
                    obs_dict=obs_dict,
                    cameras=CAMERAS,
                    goal_gripper_pose=obs_dict["goal_gripper_pose"],
                    goal_gripper_open=obs_dict["goal_gripper_open"],
                    template_4=template_4,
                )

                '''
                # ---- DEBUG VISUALIZATION HERE ----
                if d_idx == start_idx and t == 0:
                    for obs_step in range(2):
                        show_debug_goal_pcd(
                            scene_pc=point_cloud[obs_step],
                            cur_gripper_pcd=gripper_pcd[obs_step],
                            goal_gripper_pcd=goal_gripper_pcd[obs_step],
                            title=f"demo={d_idx}, t={t}, obs_step={obs_step}"
                        )

                    exit()   # <-- VERY IMPORTANT: stop after first visualization
                '''

                lang_vec = lang_goal_embs_np.mean(axis=0).astype(np.float32)  # (512,)
                lang = np.repeat(lang_vec[None, :], repeats=2, axis=0).astype(np.float32)  # (2,512)

                _assert_finite("agent_pos(stats)", agent_pos)
                _assert_finite("point_cloud(stats)", point_cloud)
                _assert_finite("goal_gripper_pcd(stats)", goal_gripper_pcd)
                _assert_finite("lang(stats)", lang)
                _assert_finite("gripper_pcd(stats)", gripper_pcd)

                stats_accumulator.update(
                    agent_pos=agent_pos,
                    point_cloud=point_cloud,
                    gripper_pcd=gripper_pcd,
                    goal_gripper_pcd=goal_gripper_pcd,
                    lang=lang,
                    action=action_seq,
                )

            replay.add(
                task,
                task_replay_storage_folder,
                action_seq,
                reward,
                terminal,
                timeout=False,
                episode_idx=d_idx,
                **obs_dict,
            )

    if stats_accumulator is not None and stats_save_path is not None:
        _save_dp3_stats(stats_accumulator, stats_save_path)

    print("DP3 replay filled successfully.")
    