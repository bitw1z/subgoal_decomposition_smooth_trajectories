# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES.
# Dataset construction for "Non-goal-conditioned dp3"

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
            "lang": RunningStats("lang"),
            "action": RunningStats("action"),
        }

    def update(self, agent_pos, point_cloud, lang, action):
        self.stats["agent_pos"].update(agent_pos)
        self.stats["point_cloud"].update(point_cloud)
        self.stats["lang"].update(lang)
        self.stats["action"].update(action)

    def state_dict(self):
        return {k: v.as_dict() for k, v in self.stats.items()}


def _build_dp3_point_cloud_from_obs_dict(obs_dict, cameras):
    """
    Convert replay-style per-camera point clouds:
      each cam: (T_obs=2, 3, H, W)
    into DP3-style:
      (T_obs=2, N, 3), where N is concatenated across cameras.
    """
    pcs = []
    for cam in cameras:
        key = f"{cam}_point_cloud"
        p = np.asarray(obs_dict[key], dtype=np.float32)   # (2,3,H,W)
        assert p.ndim == 4 and p.shape[1] == 3, (key, p.shape)

        T, C, H, W = p.shape
        p = np.transpose(p, (0, 2, 3, 1)).reshape(T, H * W, 3)  # (2, H*W, 3)
        pcs.append(p)

    pc = np.concatenate(pcs, axis=1)  # (2, N, 3)
    return pc


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

    print("Filling Non-goal conditioned DP3 replay (dense EE actions)...")
    H = action_horizon
    DEBUG_OBS = True

    stats_accumulator = DP3StatsAccumulator() if (collect_stats and agent == "dp3") else None

    for d_idx in range(start_idx, start_idx + num_demos):
        print(f"Filling demo {d_idx}")
        demo = get_stored_demo(data_path=data_path, index=d_idx)
        T = len(demo)

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
                point_cloud = _build_dp3_point_cloud_from_obs_dict(obs_dict, CAMERAS).astype(np.float32)  # (2,N,3)

                # match training-time reduction:
                # lang_goal_embs: (77,512) -> mean over token dim -> (512,)
                lang_vec = lang_goal_embs_np.mean(axis=0).astype(np.float32)  # (512,)
                lang = np.repeat(lang_vec[None, :], repeats=2, axis=0).astype(np.float32)  # (2,512)

                _assert_finite("agent_pos(stats)", agent_pos)
                _assert_finite("point_cloud(stats)", point_cloud)
                _assert_finite("lang(stats)", lang)

                stats_accumulator.update(
                    agent_pos=agent_pos,
                    point_cloud=point_cloud,
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
