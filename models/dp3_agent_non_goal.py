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

        ACTION_DIM = 8

        shape_meta = {
            "obs": {
                "agent_pos": {"shape": (8,), "type": "low_dim"},
                "point_cloud": {"shape": (65536, 3), "type": "point_cloud"},
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

    def _stack_pc_from_replay(self, replay_sample, device):
        # Each cam: (B,1,2,3,128,128) -> (B,2,16384,3)
        pcs = []
        for cam in self.cameras:
            key = f"{cam}_point_cloud"
            p = replay_sample[key].float().to(device)

            # strip YARR singleton
            if p.dim() == 6 and p.shape[1] == 1:
                p = p[:, 0]  # (B,2,3,H,W)

            B, T, C, H, W = p.shape  # T should be 2
            p = p.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, 3)  # (B,2,16384,3)
            pcs.append(p)

        pc = torch.cat(pcs, dim=2)  # concat points across cameras -> (B,2,65536,3)
        return pc

    def _translate_rvt_batch_to_dp3_batch(self, replay_sample, is_training):
        device = next(self.policy.parameters()).device

        action = replay_sample["action"].float().to(device)
        
        # --- gripper history ---
        gp = replay_sample["gripper_pose"].float().to(device)   # (B,1,T,7) or (B,T,7)
        go = replay_sample["gripper_open"].float().to(device)   # (B,1,T,1) or (B,T,1)

        # remove YARR singleton dimension
        if gp.dim() == 4 and gp.shape[1] == 1:
            gp = gp[:, 0]   # (B,T,7)
            go = go[:, 0]   # (B,T,1)

        # if it somehow comes as (B,7) / (B,1), expand to time (rare, but safe)
        if gp.dim() == 2:
            gp = gp.unsqueeze(1).repeat(1, self.n_obs_steps, 1)   # (B,T,7)
            go = go.unsqueeze(1).repeat(1, self.n_obs_steps, 1)   # (B,T,1)

        agent_pos = torch.cat([gp, go], dim=-1)   # (B,T,8)

        expected = (action.shape[0], self.n_obs_steps, 8)
        got = tuple(agent_pos.shape)
        assert got == expected, (got, expected)

        pc_list = []
        meta_keys = {
            "action",
            "low_dim_state", "low_dim_state_tp1",
            "gripper_pose", "gripper_pose_tp1",
            "gripper_open", "gripper_open_tp1",
            "episode_idx", "reward", "terminal", "timeout",
            "indices", "tasks"
        }

        pc_tensor = self._stack_pc_from_replay(replay_sample, device)  # (B,2,65536,3)

        lang = replay_sample["lang_goal_embs"].float().to(device)  # (B,1,77,512) or (B,77,512)
        if lang.dim() == 4 and lang.shape[1] == 1:
            lang = lang[:, 0]  # (B,77,512)

        lang_vec = lang.mean(dim=1)  # (B,512)
        lang_vec = lang_vec.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # (B,T,512)


        return {
            "obs": {
                "agent_pos": agent_pos,
                "point_cloud": pc_tensor,
                "lang": lang_vec,
            },
            "action": action,
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

        required_keys = ["agent_pos", "point_cloud", "lang", "action"]
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
            print("  action          :", dp3_batch["action"].shape)
            print("  obs.lang        :", dp3_batch["obs"]["lang"].shape)
            print()

        if step == 0:
            nobs = self.policy.normalizer.normalize(dp3_batch["obs"])
            naction = self.policy.normalizer["action"].normalize(dp3_batch["action"])

            def _rng(name, x):
                print(f"[NORM DBG] {name:12s} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f}")

            _rng("agent_pos", nobs["agent_pos"])
            _rng("point_cloud", nobs["point_cloud"])
            _rng("lang", nobs["lang"])
            _rng("action", naction)

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
        device = self._device
        Tobs = self.n_obs_steps
 
        gp = observation["gripper_pose"].to(device)
        go = observation["gripper_open"].to(device)

        # handle possible shapes
        if gp.dim() == 3 and gp.shape[1] == 1:
            gp = gp[:, 0]  # (1, 7)
        if go.dim() == 3 and go.shape[1] == 1:
            go = go[:, 0]  # (1, 1)

        # optional: enforce qw >= 0 to match training convention
        quat = gp[:, 3:7]
        mask = quat[:, 3] < 0
        gp[mask, 3:7] = -gp[mask, 3:7]

        agent_pos_t = torch.cat([gp, go], dim=-1).unsqueeze(1)  # (1, 1, 8)
        
        pc_keys = [
            "left_shoulder_point_cloud",
            "right_shoulder_point_cloud",
            "wrist_point_cloud",
            "front_point_cloud",
        ]

        pc_list = []
        for k in pc_keys:
            p = observation[k].to(device)   # (1, 1, 3, 128, 128)
            p = p[:, -1]                    # (1, 3, 128, 128)
            B, C, H, W = p.shape
            p = p.permute(0, 2, 3, 1).reshape(B, H * W, 3)
            pc_list.append(p)

        pc_t = torch.cat(pc_list, dim=1)      # (1, 65536, 3)
        pc_t = pc_t.unsqueeze(1)

        # update history
        self._eval_agentpos_hist.append(agent_pos_t)
        self._eval_pc_hist.append(pc_t)

        # pad at episode start
        while len(self._eval_agentpos_hist) < Tobs:
            self._eval_agentpos_hist.appendleft(self._eval_agentpos_hist[0])
        while len(self._eval_pc_hist) < Tobs:
            self._eval_pc_hist.appendleft(self._eval_pc_hist[0])

        if len(self._action_buffer) > 0:
            a = self._action_buffer.pop(0)
            return ActResult(a)

        # Build stacked policy input (including history)
        agent_pos = torch.cat(list(self._eval_agentpos_hist), dim=1)  # (B, Tobs, 8)
        pc = torch.cat(list(self._eval_pc_hist), dim=1)              # (B, Tobs, N, 3)

        lang_tokens = observation["lang_goal_tokens"].long().to(device)
        if lang_tokens is None:
            raise KeyError("Expected 'lang_goal_tokens' in observation when add_lang=True")

        lang_tokens = lang_tokens.long().to(device)

        _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_tokens[0])
        lang_goal_embs = lang_goal_embs.float() 

        if lang_goal_embs.dim() == 2:
            lang_goal_embs = lang_goal_embs.unsqueeze(0)  # (1,77,512)

        lang_vec = lang_goal_embs.mean(dim=1)            # (1, 512)
        lang_vec = lang_vec.unsqueeze(1).repeat(1, Tobs, 1)  # (1, Tobs, 512)

        # ------------------------
        # DP3 policy input
        # ------------------------
        policy_input = {
            "agent_pos": agent_pos,
            "point_cloud": pc,
            "lang": lang_vec,
        }

        # ================= DEBUG: ACT INPUT =================
        if self.debug_act and not self._debug_printed:
            print("\n========== DP3 ACT DEBUG ==========")
            print("[DP3 ACT DBG] observation keys:", list(observation.keys()))

            self._dbg("agent_pos", agent_pos)
            self._dbg("point_cloud", pc)
            self._dbg("lang_vec", lang_vec)

            print("[DP3 ACT DBG] policy_input keys:", list(policy_input.keys()))
            print("===================================\n")

            self._debug_printed = True

        # ===================================================

        out = self.policy.predict_action(policy_input) 

        if self.debug_act and self._debug_printed:
            action_out = out["action"]
            print(
                f"[DP3 ACT DBG] out['action'] shape={tuple(action_out.shape)} "
                f"(expected: [B, T, 8])"
            )

        action_seq = out["action"][0].detach().cpu().numpy()

        for i in range(len(action_seq)):
            # Discretize gripper
            action_seq[i, 7] = 1.0 if action_seq[i, 7] > 0.5 else 0.0

            # Normalize quaternion
            quat = action_seq[i, 3:7]
            norm = np.linalg.norm(quat)
            if norm < 1e-6:
                quat = np.array([0, 0, 0, 1.0])
            else:
                quat = quat / norm
            action_seq[i, 3:7] = quat

        K = getattr(self, "exec_horizon", 8)
        action_seq = action_seq[:K]

        self._action_buffer = list(action_seq)
        a = self._action_buffer.pop(0)

        print("[STEP0 ACTION]", a)
        print("  pos:", a[:3], "||pos||:", float(np.linalg.norm(a[:3])))
        print("  quat:", a[3:7], "||quat||:", float(np.linalg.norm(a[3:7])))
        print("  grip:", a[7])

        return ActResult(a)
        
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