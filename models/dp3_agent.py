import torch
import clip
import numpy as np
import torch.optim as optim
from collections import deque
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, List, Dict

# --- YARR ---
from yarr.agents.agent import ActResult, Agent

# --- RVT utilities (ONLY for preprocessing + PC extraction) ---
import rvt.utils.peract_utils as peract_utils
import rvt.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils
from rvt.utils.dataset import _clip_encode_text

from torch.optim.lr_scheduler import CosineAnnealingLR
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

# --- DP3 ---
from diffusion_policy_3d.policy.dp3 import DP3
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import pickle
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

# --- constants ---
from rvt.utils.peract_utils import LOW_DIM_SIZE

DEFAULT_CAMERAS = ['left_shoulder', 'right_shoulder', 'wrist', 'front']

def manage_loss_log(agent, loss_log: Dict[str, float], reset_log: bool):
    if (not hasattr(agent, "loss_log")) or reset_log:
        agent.loss_log = {}
    for k, v in loss_log.items():
        if k in agent.loss_log:
            agent.loss_log[k].append(v)
        else:
            agent.loss_log[k] = [v]


class ConfigDict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value

def _prepare_low_dim_obs_chunk(
    current_state: np.ndarray,
    history: Optional[List[np.ndarray]],
    n_obs_steps: int,
    default_dim: int
) -> np.ndarray:
    if history is None:
        history = [np.zeros(default_dim, dtype=np.float32)] * (n_obs_steps - 1)
    history = history[-(n_obs_steps - 1):]
    history.append(current_state)
    return np.stack(history, axis=0)


class dp3_agent(Agent):

    def __init__(
        self,
        policy_cfg: dict,
        cameras: list = DEFAULT_CAMERAS,
        cos_dec_max_step: int = 160000,
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        lr_warmup_steps: int = 2000,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        amp: bool = False,
        **kwargs,
    ):
        super().__init__()

        H = int(policy_cfg.get("horizon", 16))
        N_ACTION_STEPS = int(policy_cfg.get("n_action_steps", H))
        N_OBS_STEPS = int(policy_cfg.get("n_obs_steps", 2))
        policy_cfg["horizon"] = H
        policy_cfg["n_action_steps"] = N_ACTION_STEPS
        policy_cfg["n_obs_steps"] = N_OBS_STEPS

        self.n_obs_steps = 2
        self._proprio_history: Optional[List[np.ndarray]] = None

        encoder_output_dim = policy_cfg.get("encoder_output_dim", 64)
        pc_enc_cfg = policy_cfg.get("pointcloud_encoder_cfg", None)
        if pc_enc_cfg is None:
            pc_enc_cfg = ConfigDict(
                in_channels=3,
                out_channels=encoder_output_dim,
                use_layernorm=True,
                final_norm="layernorm",
                normal_channel=False,
            )
        elif isinstance(pc_enc_cfg, dict):
            pc_enc_cfg = ConfigDict(**pc_enc_cfg)

        policy_cfg["pointcloud_encoder_cfg"] = pc_enc_cfg

        noise_cfg = policy_cfg.pop("noise_scheduler_cfg", {})
        target = noise_cfg.get("_target_", "")

        if "DDIMScheduler" in target:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=noise_cfg.get("num_train_timesteps", 1000),
                beta_start=noise_cfg.get("beta_start", 0.0001),
                beta_end=noise_cfg.get("beta_end", 0.02),
            )
        else:
            beta_schedule = noise_cfg.get("beta_schedule", "linear")
            if beta_schedule == "squaredcos_v2":
                beta_schedule = "linear"
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=noise_cfg.get("num_train_timesteps", 1000),
                beta_schedule=beta_schedule,
            )

        NUM_SCENE_PCD = 1024
        NUM_HAND_PCD = 4
        ACTION_DIM = 8

        shape_meta = {
            "obs": {
                "agent_pos": {"shape": (8,), "type": "low_dim"},
                "point_cloud": {"shape": (NUM_SCENE_PCD, 3), "type": "point_cloud"},
                "gripper_pcd": {"shape": (NUM_HAND_PCD, 3), "type": "point_cloud"},
                "goal_gripper_pcd": {"shape": (NUM_HAND_PCD, 3), "type": "point_cloud"},
                "lang": {"shape": (512,), "type": "low_dim"},
            },
            "action": {"shape": (ACTION_DIM,), "type": "action"},  
        }

        print("\n========= SHAPE META (DP3Agent) =========")
        for k, v in shape_meta["obs"].items():
            print(f"obs[{k}]  shape={v['shape']}")
        print(f"action shape={shape_meta['action']['shape']}")
        print("=========================================\n")

        # ---- extract architecture-critical args ----
        pointnet_type = policy_cfg.pop("pointnet_type")            # "pointnet"
        pointcloud_encoder_cfg = policy_cfg.pop("pointcloud_encoder_cfg")

        self.policy = DP3(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,

            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,

            **policy_cfg,
        )

        self.cameras = cameras
        self.normalizer = self.policy.normalizer

        self.__optimizer = None
        self.__lr_sched = None
        self._device = None
        self._scaler = GradScaler(enabled=amp)
        self._amp = amp

        self._optimizer_type = optimizer_type
        self._lr = lr
        self._min_lr = min_lr
        self._lr_warmup_steps = lr_warmup_steps
        self._weight_decay = weight_decay
        self._cos_dec_max_step = cos_dec_max_step

        self._normalizer_fitted = False

        self._action_buffer = []

        self.debug_act = True          # turn off later
        self._debug_printed = False

        self.exec_horizon = 15   # standard DP3: execute 8 out of predicted 16
            

    def _dbg(self, name, x):
        if torch.is_tensor(x):
            finite = torch.isfinite(x).all().item() if x.is_floating_point() else True
            print(
                f"[DP3 ACT DBG] {name}: "
                f"shape={tuple(x.shape)} "
                f"dtype={x.dtype} "
                f"device={x.device} "
                f"finite={finite} "
                + (
                    f"min={x.min().item():.3g} max={x.max().item():.3g}"
                    if x.is_floating_point()
                    else ""
                )
            )
        else:
            print(f"[DP3 ACT DBG] {name}: {type(x)}")

    def load_clip(self):
        self.clip_model, _ = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        if hasattr(self, "clip_model"):
            del self.clip_model
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()

    def build(self, training: bool, device: str):
        self._device = device
        self.policy.to(device)

        self.gripper_template_4 = dp3_agent.make_gripper_template_4(
            device=torch.device(device),
            dtype=torch.float32
        )

        OptimizerClass = optim.AdamW if self._optimizer_type == "adamw" else optim.Adam
        self._optimizer = OptimizerClass(
            self.policy.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        cosine_sched = CosineAnnealingLR(
            self._optimizer,
            T_max=self._cos_dec_max_step,
            eta_min=self._min_lr,
        )

        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1.0,
            total_epoch=self._lr_warmup_steps,
            after_scheduler=cosine_sched,
        )

    @staticmethod
    def make_gripper_template_4(device, dtype=torch.float32):
        """
        Deterministic canonical Panda gripper keypoints (4, 3)
        in gripper-local frame.

        Point order:
        0: left finger contact point
        1: right finger contact point
        2: wrist / gripper base point
        3: center contact point (midpoint between left/right)
        """

        # Approximate local-frame coordinates for Panda gripper.
        # These should be adjusted once you confirm the exact gripper frame convention.
        #
        # Assumption here:
        # - z points forward from gripper base toward fingertip/contact region
        # - y separates left/right fingers
        # - x is lateral thickness direction
        #
        # The exact values are less important than semantic consistency.

        left_finger = torch.tensor([0.0, -0.0405, 0.0800], device=device, dtype=dtype)
        right_finger = torch.tensor([0.0,  0.0405, 0.0800], device=device, dtype=dtype)
        wrist = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

        center_contact = 0.5 * (left_finger + right_finger)

        template = torch.stack(
            [left_finger, right_finger, wrist, center_contact], dim=0
        )

        assert template.shape == (4, 3)
        return template

    @staticmethod
    def quat_to_rotmat(q):
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        x, y, z, w = q.unbind(-1)

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))
        return R

    @staticmethod
    def build_gripper_pcd_from_pose(pose_7, open_1, template_4):
        """
        pose_7: (B, 7) = [x, y, z, qx, qy, qz, qw]
        open_1: (B, 1) gripper openness in [0, 1]
        template_4: (4, 3) in gripper-local frame

        Point order in template_4:
        0: left finger point
        1: right finger point
        2: wrist point
        3: center contact point

        returns:
        p_world: (B, 4, 3) in world frame
        """
        assert pose_7.dim() == 2 and pose_7.shape[-1] == 7, pose_7.shape
        assert open_1.dim() == 2 and open_1.shape[-1] == 1, open_1.shape
        assert template_4.shape == (4, 3), template_4.shape

        B = pose_7.shape[0]

        # translation + rotation
        t = pose_7[:, :3]               # (B, 3)
        q = pose_7[:, 3:7]              # (B, 4)
        R = dp3_agent.quat_to_rotmat(q) # (B, 3, 3)

        # start from canonical local template
        p = template_4.unsqueeze(0).expand(B, -1, -1).clone()   # (B, 4, 3)

        # ------------------------------------------------------------------
        # Adjust finger opening in LOCAL frame.
        #
        # Assumption:
        # - point 0 = left finger
        # - point 1 = right finger
        # - point 2 = wrist
        # - point 3 = center contact
        #
        # We move left/right finger points along local y-axis.
        # Then recompute the center contact as the midpoint.
        # ------------------------------------------------------------------
        open_amt = torch.clamp(open_1[:, 0], 0.0, 1.0)   # (B,)

        # If template_4 was defined at open=0.5, this keeps that as zero offset.
        # 0.021 is the approximate full finger travel scale you used before.
        delta = 0.021 * (open_amt - 0.5)                  # (B,)
        delta = delta.view(B)                             # (B,)

        # left finger (index 0), right finger (index 1)
        # choose signs so that increasing open_amt moves them farther apart
        p[:, 0, 1] -= delta   # left finger
        p[:, 1, 1] += delta   # right finger

        # recompute center contact as midpoint of left/right finger points
        p[:, 3, :] = 0.5 * (p[:, 0, :] + p[:, 1, :])

        # world transform: p_world = R * p_local + t
        p_world = torch.einsum("bij,bpj->bpi", R, p) + t.unsqueeze(1)   # (B, 4, 3)
        return p_world
    
    @staticmethod
    def farthest_point_sampling(xyz: torch.Tensor, n_samples: int):
        """
        xyz: (B, N, 3)
        returns: indices (B, n_samples)
        """
        B, N, _ = xyz.shape
        device = xyz.device

        centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device)

        farthest = torch.zeros(B, dtype=torch.long, device=device)

        batch_indices = torch.arange(B, device=device)

        for i in range(n_samples):
            centroids[:, i] = farthest
            centroid_xyz = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)

            dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
            mask = dist < distances
            distances[mask] = dist[mask]

            farthest = torch.max(distances, dim=1)[1]

        return centroids

    def _translate_rvt_batch_to_dp3_batch(self, replay_sample, is_training):
        device = next(self.policy.parameters()).device
        T = self.n_obs_steps

        action = replay_sample["action"].float().to(device)
        B = action.shape[0]

        # -------------------------------------------------
        # current gripper history: expect (B,1,T,7)/(B,T,7) and (B,1,T,1)/(B,T,1)
        # -------------------------------------------------
        gp = replay_sample["gripper_pose"].float().to(device)   # (B,1,T,7) or (B,T,7) or rare (B,7)
        go = replay_sample["gripper_open"].float().to(device)   # (B,1,T,1) or (B,T,1) or rare (B,1)

        # remove YARR singleton
        if gp.dim() == 4 and gp.shape[1] == 1:
            gp = gp[:, 0]   # (B,T,7)
            go = go[:, 0]   # (B,T,1)

        # rare fallback: single state -> repeat across T
        if gp.dim() == 2:
            gp = gp.unsqueeze(1).repeat(1, T, 1)   # (B,T,7)
            go = go.unsqueeze(1).repeat(1, T, 1)   # (B,T,1)

        assert gp.dim() == 3 and gp.shape[1] == T and gp.shape[2] == 7, gp.shape
        assert go.dim() == 3 and go.shape[1] == T and go.shape[2] == 1, go.shape

        # canonicalize current quaternion signs for all timesteps
        current_quat = gp[..., 3:7]                 # (B,T,4)
        current_mask = current_quat[..., 3] < 0     # qw < 0
        gp[..., 3:7][current_mask] *= -1

        # low-dim state
        agent_pos = torch.cat([gp, go], dim=-1)     # (B,T,8)

        expected = (B, T, 8)
        got = tuple(agent_pos.shape)
        assert got == expected, (got, expected)

        # -------------------------------------------------
        # goal gripper state: usually static, repeat across T later
        # -------------------------------------------------
        goal_pose = replay_sample["goal_gripper_pose"].float().to(device)   # (B,7) or (B,1,7)
        goal_open = replay_sample["goal_gripper_open"].float().to(device)   # (B,1) or (B,1,1)

        if goal_pose.dim() == 3 and goal_pose.shape[1] == 1:
            goal_pose = goal_pose[:, 0]   # (B,7)
            goal_open = goal_open[:, 0]   # (B,1)

        assert goal_pose.dim() == 2 and goal_pose.shape[1] == 7, goal_pose.shape
        assert goal_open.dim() == 2 and goal_open.shape[1] == 1, goal_open.shape

        # canonicalize goal quaternion signs
        goal_quat = goal_pose[:, 3:7]
        goal_mask = goal_quat[:, 3] < 0
        goal_pose[goal_mask, 3:7] *= -1

        # -------------------------------------------------
        # build current gripper pcd for each timestep
        # helper expects (N,7), (N,1), so flatten B*T first
        # -------------------------------------------------
        gp_flat = gp.reshape(B * T, 7)
        go_flat = go.reshape(B * T, 1)

        cur_gripper_pcd = self.build_gripper_pcd_from_pose(
            gp_flat, go_flat, self.gripper_template_4
        )
        cur_gripper_pcd = cur_gripper_pcd.reshape(B, T, 4, 3)

        # -------------------------------------------------
        # build goal gripper pcd once, then repeat across T
        # -------------------------------------------------

        goal_gripper_pcd = self.build_gripper_pcd_from_pose(
            goal_pose, goal_open, self.gripper_template_4
        )
        goal_gripper_pcd = goal_gripper_pcd.unsqueeze(1).repeat(1, T, 1, 1)   

        # -------------------------------------------------
        # build dense scene point cloud for each timestep
        # We construct one observation dict per (batch_idx, timestep)
        # and run the same RVT preprocessing used in goal-conditioned DP3.
        # -------------------------------------------------
        meta_keys = {
            "action",
            "low_dim_state", "low_dim_state_tp1",
            "gripper_pose", "gripper_pose_tp1",
            "gripper_open", "gripper_open_tp1",
            "goal_gripper_pose", "goal_gripper_open",
            "episode_idx", "reward", "terminal", "timeout",
            "indices", "tasks",
            "lang_goal_embs", "lang_goal_tokens",
        }

                # -------------------------------------------------
        # build dense scene point cloud directly from replay
        # Each camera point cloud is stored as:
        #   (B,1,T,3,H,W) or (B,T,3,H,W)
        # We convert each timestep to (num_cams * H * W, 3)
        # -------------------------------------------------
        scene_pc_bt = []

        for b in range(B):
            scene_pc_t = []

            for t in range(T):
                cam_points = []

                for cam in self.cameras:
                    pc = replay_sample[f"{cam}_point_cloud"].float().to(device)

                    # Remove YARR singleton if present
                    if pc.dim() == 6 and pc.shape[1] == 1:
                        pc = pc[:, 0]   # (B,T,3,H,W)

                    assert pc.dim() == 5, (cam, pc.shape)
                    assert pc.shape[1] == T, (cam, pc.shape)

                    pc_bt = pc[b, t]   # (3,H,W)
                    assert pc_bt.dim() == 3 and pc_bt.shape[0] == 3, (cam, pc_bt.shape)

                    # (3,H,W) -> (H,W,3) -> (H*W,3)
                    pc_flat = pc_bt.permute(1, 2, 0).reshape(-1, 3)
                    cam_points.append(pc_flat)

                pc_cat = torch.cat(cam_points, dim=0)   # (num_cams * H * W, 3)
                scene_pc_t.append(pc_cat)

            scene_pc_t = torch.stack(scene_pc_t, dim=0)   # (T,N,3)
            scene_pc_bt.append(scene_pc_t)

        scene_pc = torch.stack(scene_pc_bt, dim=0).float()   # (B,T,N,3)

        if is_training and not hasattr(self, "_debug_scene_pc"):
            self._debug_scene_pc = scene_pc.detach().cpu()

        # -------------------------------------------------
        # FPS downsample each (B,T) scene cloud to 1024 points
        # -------------------------------------------------
        scene_pc_flat = scene_pc.reshape(B * T, scene_pc.shape[2], 3)   # (B*T,65536,3)

        fps_idx = self.farthest_point_sampling(scene_pc_flat, 1024)   # (B*T,1024)
        scene_1024_flat = torch.gather(
            scene_pc_flat, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
        )   # (B*T,1024,3)

        scene_1024 = scene_1024_flat.reshape(B, T, 1024, 3)   # (B,T,1024,3)

        # -------------------------------------------------
        # language: static instruction, repeat across T
        # -------------------------------------------------
        lang = replay_sample["lang_goal_embs"].float().to(device)   # (B,1,77,512) or (B,77,512)
        if lang.dim() == 4 and lang.shape[1] == 1:
            lang = lang[:, 0]   # (B,77,512)

        lang_vec = lang.mean(dim=1)                                 # (B,512)
        lang_vec = lang_vec.unsqueeze(1).repeat(1, T, 1)            # (B,T,512)

        return {
            "obs": {
                "agent_pos": agent_pos,                 # (B,T,8)
                "point_cloud": scene_1024,              # (B,T,1024,3)
                "gripper_pcd": cur_gripper_pcd,         # (B,T,4,3)
                "goal_gripper_pcd": goal_gripper_pcd,   # (B,T,4,3)
                "lang": lang_vec,                       # (B,T,512)
            },
            "action": action,                           # (B,H,8)
        }
    
    def load_normalizer_from_stats(
        self,
        stats_path: str,
        mode: str = "limits",
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
        lang_std_floor: float = 1e-2,
        verbose: bool = True,
    ):
        """
        Load replay-time statistics from dp3_norm_stats.pkl and build a fitted
        LinearNormalizer for the exact DP3 keys:
            obs.agent_pos
            obs.point_cloud
            obs.goal_gripper_pcd
            obs.lang
            action

        Expected stats file format for each key:
            {
                "count": int,
                "mean": np.ndarray,   # shape (D,)
                "std": np.ndarray,    # shape (D,)
                "min": np.ndarray,    # shape (D,)
                "max": np.ndarray,    # shape (D,)
            }
        """
        assert mode in ["limits", "gaussian"], mode

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        required_keys = ["agent_pos", "point_cloud", "gripper_pcd", "goal_gripper_pcd", "lang", "action"]
        for key in required_keys:
            if key not in stats:
                raise KeyError(f"Missing '{key}' in stats file: {stats_path}")

        def _to_torch_1d(x, name):
            if isinstance(x, torch.Tensor):
                t = x.detach().clone().float()
            else:
                t = torch.as_tensor(x, dtype=torch.float32)
            t = t.flatten()
            if t.ndim != 1:
                raise ValueError(f"{name} must flatten to 1D, got shape {tuple(t.shape)}")
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains NaN/Inf")
            return t

        def _build_single_field_from_stats(field_name: str, field_stats: dict):
            x_min = _to_torch_1d(field_stats["min"], f"{field_name}.min")
            x_max = _to_torch_1d(field_stats["max"], f"{field_name}.max")
            x_mean = _to_torch_1d(field_stats["mean"], f"{field_name}.mean")
            x_std = _to_torch_1d(field_stats["std"], f"{field_name}.std")

            # Extra protection for language dimensions with tiny variance
            if field_name == "lang":
                x_std = torch.clamp(x_std, min=lang_std_floor)

            if mode == "limits":
                input_range = x_max - x_min
                ignore_dim = input_range < range_eps

                safe_range = input_range.clone()
                safe_range[ignore_dim] = (output_max - output_min)

                scale = (output_max - output_min) / safe_range
                offset = output_min - scale * x_min

                # Same constant-dimension handling as LinearNormalizer._fit(..., mode="limits")
                offset[ignore_dim] = (output_max + output_min) / 2.0 - x_min[ignore_dim]

            else:  # gaussian
                safe_std = x_std.clone()
                safe_std = torch.clamp(safe_std, min=range_eps)

                scale = 1.0 / safe_std
                offset = -x_mean * scale

            input_stats_dict = {
                "min": x_min,
                "max": x_max,
                "mean": x_mean,
                "std": x_std,
            }

            single = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=input_stats_dict,
            )

            if verbose:
                rng = x_max - x_min
                print(
                    f"[NORMALIZER LOAD] {field_name:12s} "
                    f"dim={x_mean.numel():4d} "
                    f"min_std={x_std.min().item():.3e} "
                    f"max_std={x_std.max().item():.3e} "
                    f"min_range={rng.min().item():.3e} "
                    f"max_range={rng.max().item():.3e} "
                    f"ignored={(rng < range_eps).sum().item()}"
                )

            return single

        # Build a full LinearNormalizer with the exact keys DP3 expects
        normalizer = LinearNormalizer()
        normalizer["agent_pos"] = _build_single_field_from_stats("agent_pos", stats["agent_pos"])
        normalizer["point_cloud"] = _build_single_field_from_stats("point_cloud", stats["point_cloud"])
        normalizer["gripper_pcd"] = _build_single_field_from_stats("gripper_pcd", stats["gripper_pcd"])
        normalizer["goal_gripper_pcd"] = _build_single_field_from_stats(
            "goal_gripper_pcd", stats["goal_gripper_pcd"]
        )
        normalizer["lang"] = _build_single_field_from_stats("lang", stats["lang"])
        normalizer["action"] = _build_single_field_from_stats("action", stats["action"])

        # Install into policy
        self.policy.set_normalizer(normalizer)
        self._normalizer_fitted = True
        self.policy.to(self._device)

        if verbose:
            print(f"[NORMALIZER LOAD] Loaded from: {stats_path}")
            print(f"[NORMALIZER LOAD] mode={mode} output_range=({output_min}, {output_max})")
            print(f"[NORMALIZER LOAD] ready={self._normalizer_fitted}")

    def update(self, replay_sample, backprop, reset_log, eval_log, step):
        assert self._normalizer_fitted, "Normalizer must be loaded before update()."

        if step == 0:
            print("\n[DP3Agent] Replay sample keys:")
            for k, v in replay_sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k:30s} {tuple(v.shape)}")
        
        
        dp3_batch = self._translate_rvt_batch_to_dp3_batch(
            replay_sample, is_training=(step == 0)
        )

        if step == 0:
            print("\n[DP3Agent] DP3 batch shapes:")
            print("  obs.agent_pos   :", dp3_batch["obs"]["agent_pos"].shape)
            print("  obs.point_cloud :", dp3_batch["obs"]["point_cloud"].shape)
            print("  obs.gripper_pcd :", dp3_batch["obs"]["gripper_pcd"].shape)
            print("  obs.goal_gripper_pcd :", dp3_batch["obs"]["goal_gripper_pcd"].shape)
            print("  action          :", dp3_batch["action"].shape)
            print("  obs.lang        :", dp3_batch["obs"]["lang"].shape)
            print()

        if step == 0:
            nobs = self.policy.normalizer.normalize(dp3_batch["obs"])
            naction = self.policy.normalizer["action"].normalize(dp3_batch["action"])

            def _rng(name, x):
                print(
                    f"[NORM DBG] {name:16s} "
                    f"min={x.min().item():.3f} "
                    f"max={x.max().item():.3f} "
                    f"mean={x.mean().item():.3f}"
                )

            _rng("agent_pos", nobs["agent_pos"])
            _rng("point_cloud", nobs["point_cloud"])
            _rng("goal_gripper_pcd", nobs["goal_gripper_pcd"])
            _rng("lang", nobs["lang"])
            _rng("action", naction)
            _rng("gripper_pcd", nobs["gripper_pcd"])

        if step == 0:
            print("[NORMALIZER] ready:", self._normalizer_fitted)

        with autocast(enabled=self._amp):
            loss, loss_dict = self.policy.compute_loss(dp3_batch)

        # --- STOP IMMEDIATELY if loss is NaN/Inf (before optimizer step) ---
        if not torch.isfinite(loss).item():
            print("\n[NaN-TRIPWIRE] Non-finite loss detected.")
            print(f"  step={step}  loss={loss.item()}")

            def _stat(name, x):
                if torch.is_tensor(x) and x.is_floating_point():
                    fin = torch.isfinite(x)
                    print(f"  {name:16s} finite={fin.all().item()} "
                        f"min={x[fin].min().item() if fin.any() else float('nan'):.3g} "
                        f"max={x[fin].max().item() if fin.any() else float('nan'):.3g}")

            _stat("agent_pos",   dp3_batch["obs"]["agent_pos"])
            _stat("point_cloud", dp3_batch["obs"]["point_cloud"])
            _stat("gripper_pcd", dp3_batch["obs"]["gripper_pcd"])
            _stat("goal_gripper_pcd", dp3_batch["obs"]["goal_gripper_pcd"])
            _stat("lang",        dp3_batch["obs"]["lang"])
            _stat("action",      dp3_batch["action"])

            raise RuntimeError("Stopping to avoid poisoning weights with NaNs.")

        self._loss = float(loss.item()) 

        if backprop:
            self._optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._lr_sched.step()

        log = {k: float(v) for k, v in loss_dict.items()}
        log["total_loss"] = float(loss.item())
        log["lr"] = self._optimizer.param_groups[0]["lr"]

        manage_loss_log(self, log, reset_log)
        return log

    @torch.no_grad()
    def act(self, step, observation, deterministic=True, pred_distri=False):
        raise NotImplementedError(
            "Standalone act() is disabled for goal-conditioned DP3. "
            "Use the hierarchical wrapper act() instead."
        )
        
    def reset(self):
        self._proprio_history = None
        self._action_buffer = []
        self._eval_agentpos_hist = deque(maxlen=self.n_obs_steps)
        self._eval_pc_hist = deque(maxlen=self.n_obs_steps)

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    @property
    def _network(self):
        return self.policy

    @property
    def _optimizer(self):
        return self.__optimizer

    @_optimizer.setter
    def _optimizer(self, v):
        self.__optimizer = v

    @property
    def _lr_sched(self):
        return self.__lr_sched

    @_lr_sched.setter
    def _lr_sched(self, v):
        self.__lr_sched = v

    def parameters(self):
        return self.policy.parameters()

    def update_summaries(self, **kwargs):
        return {"loss": getattr(self, "_loss", None)}

    def act_summaries(self, **kwargs):
        return {}

    def save_weights(self, path):
        torch.save(self._network.state_dict(), path)

    def load_weights(self, path):
        self._network.load_state_dict(torch.load(path))