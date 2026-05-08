import torch
import os
import numpy as np
from typing import List, Optional

# --- YARR ---
from yarr.agents.agent import Agent, ActResult

# --- RVT ---
import rvt.models.rvt_agent as rvt_agent

# --- DP3 ---
from rvt.models.dp3_agent import dp3_agent
from collections import deque
from rvt.utils.dataset import _clip_encode_text

class HierarchicalRVT2DP3Agent(Agent):
    """
    Hierarchical evaluation agent:
      High-level  : RVT-2 (goal prediction)
      Low-level   : goal-conditioned DP3 (action synthesis)
    """

    def __init__(
        self,
        high_agent: rvt_agent.RVTAgent,
        low_agent: dp3_agent,
        cameras: List[str],
        exec_horizon: int = 16,
    ):
        super().__init__()

        # -----------------------------
        # Sub-agents (already trained)
        # -----------------------------
        self.high = high_agent   # RVTAgent
        self.low  = low_agent    # dp3_agent instance

        self.cameras = cameras

        # -----------------------------
        # Hierarchy timing
        # -----------------------------
        self.exec_horizon = exec_horizon

        # -----------------------------
        # Persistent internal state
        # -----------------------------
        self.current_goal_pose = None        # (1, 7)
        self.current_goal_open = None        # (1, 1)
        self.current_goal_pcd  = None        # (1, 4, 3)

        self._action_queue: List[np.ndarray] = []

        # -----------------------------
        # Runtime bookkeeping
        # -----------------------------
        self._device = None
        self._episode_started = False

        self._rvt_goal_history = []     # list of dicts: {step, goal_pose(7,), goal_open(1,)}
        self._dp3_action_history = []   # list of dicts: {step, action(8,)}
        self._episode_id = 0

        self._agent_pos_history = deque(maxlen=2)
        self._scene_pcd_history = deque(maxlen=2)
        
    def build(self, training: bool, device: str):
        """
        Called once before rollout.
        """

        self._device = device

        # High-level agent (RVT-2)
        self.high.build(training=False, device=device)
        self.high.eval()

        if hasattr(self.high, "load_clip"):
            self.high.load_clip()

        # Low-level agent (DP3)
        self.low.build(training=False, device=device)
        self.low.eval()

        # Reset hierarchy state
        self.reset()

    def eval(self):
        self.high.eval()
        self.low.eval()

    def train(self):
        self.high.train()
        self.low.train()    

    def _plot_episode_trajectory(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if len(self._rvt_goal_history) == 0 or len(self._dp3_action_history) == 0:
            print("No data to plot.")
            return

        # --- collect RVT goals ---
        rvt_goals = np.array([
            h["goal_pose"][:3] for h in self._rvt_goal_history
        ])  # (N,3)

        # --- collect DP3 executed positions ---
        dp3_actions = np.array([
            h["action"][:3] for h in self._dp3_action_history
        ])  # (T,3)

        # --- plot ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # DP3 trajectory (dense)
        ax.plot(
            dp3_actions[:, 0],
            dp3_actions[:, 1],
            dp3_actions[:, 2],
            label="DP3 trajectory",
            linewidth=2
        )

        # RVT goals (sparse)
        ax.scatter(
            rvt_goals[:, 0],
            rvt_goals[:, 1],
            rvt_goals[:, 2],
            c='red',
            label="RVT goals",
            s=60
        )

        # connect RVT goals
        ax.plot(
            rvt_goals[:, 0],
            rvt_goals[:, 1],
            rvt_goals[:, 2],
            linestyle='--'
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Episode {self._episode_id} trajectory")

        ax.legend()
        plt.tight_layout()

        plt.show()

    @torch.no_grad()
    def act(self, step, observation, deterministic=True, **kwargs):

        # print all observation keys once per episode
        if not hasattr(self, "_printed_obs_keys"):
            self._printed_obs_keys = True
            print("==== OBSERVATION KEYS ====")
            for k in sorted(observation.keys()):
                try:
                    v = observation[k]
                    shape = v.shape if hasattr(v, "shape") else type(v)
                    print(f"{k}: {shape}")
                except Exception as e:
                    print(f"{k}: <error reading> {e}")
            print("==========================")

        device = self._device

        # ---------------------------------------------------------
        # 0. Episode start
        # ---------------------------------------------------------
        if not self._episode_started:
            self._episode_started = True
            self.current_goal_pose = None
            self.current_goal_open = None
            self.current_goal_pcd  = None
            self._action_queue.clear()

        gp = observation["gripper_pose"].to(device=device, dtype=torch.float32)
        go = observation["gripper_open"].to(device=device, dtype=torch.float32)

        if gp.dim() == 3 and gp.shape[1] == 1:
            gp = gp[:, 0]
        if go.dim() == 3 and go.shape[1] == 1:
            go = go[:, 0]

        # gp: (1,7), go: (1,1)
        quat = gp[:, 3:7]
        mask = quat[:, 3] < 0
        gp[mask, 3:7] = -gp[mask, 3:7]

        agent_pos_now = torch.cat([gp, go], dim=-1)   # (1,8)

        # maintain history
        self._agent_pos_history.append(agent_pos_now)
        while len(self._agent_pos_history) < 2:
            self._agent_pos_history.append(agent_pos_now.clone())

        pc_keys = [
            "left_shoulder_point_cloud",
            "right_shoulder_point_cloud",
            "wrist_point_cloud",
            "front_point_cloud",
        ]

        cam_points = []
        for k in pc_keys:
            p = observation[k].to(device=device, dtype=torch.float32)

            # expected current eval shape: (B,1,3,H,W) or (B,3,H,W)
            if p.dim() == 5 and p.shape[1] == 1:
                p = p[:, 0]   # (B,3,H,W)
            elif p.dim() == 4:
                pass          # already (B,3,H,W)
            else:
                raise RuntimeError(f"{k} unexpected shape: {p.shape}")

            B, C, H, W = p.shape
            p = p.permute(0, 2, 3, 1).reshape(B, H * W, 3)
            cam_points.append(p)

        scene_pc_now = torch.cat(cam_points, dim=1)   # (B,N,3)

        fps_idx = dp3_agent.farthest_point_sampling(scene_pc_now, 1024)
        scene_1024_now = torch.gather(
            scene_pc_now, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
        )   # (B,1024,3)

        self._scene_pcd_history.append(scene_1024_now)
        while len(self._scene_pcd_history) < 2:
            self._scene_pcd_history.append(scene_1024_now.clone())

        if len(self._action_queue) == 0:

            # ---- query RVT-2 ----
            high_res: ActResult = self.high.act(
                step=step,
                observation=observation,
                deterministic=True,
            )

            high_action = torch.tensor(
                high_res.action,
                device=device,
                dtype=torch.float32
            ).unsqueeze(0)  # (1, 8)

            goal_pose = high_action[:, :7]
            goal_open = high_action[:, 7:8]

            # ---- quaternion sign convention ----
            quat = goal_pose[:, 3:7]
            mask = quat[:, 3] < 0
            goal_pose[mask, 3:7] = -goal_pose[mask, 3:7]

            print("===== RVT PREDICTS GOAL!! =====")
            print("RVT goal:", goal_pose)
            print("============================")

            # ---- build goal gripper point cloud ----
            self.current_goal_pcd = self.low.build_gripper_pcd_from_pose(
                goal_pose,
                goal_open,
                self.low.gripper_template_4.to(device),
            )

            self.current_goal_pose = goal_pose
            self.current_goal_open = goal_open
            self._action_queue.clear()

            self._rvt_goal_history.append({
                "step": int(step),
                "goal_pose": goal_pose[0].detach().cpu().numpy().copy(),  # (7,)
                "goal_open": goal_open[0].detach().cpu().numpy().copy(),  # (1,)
            })

            cur_pos = gp[:, :3]
            goal_pos = goal_pose[:, :3]

            dist = torch.norm(cur_pos - goal_pos, dim=-1)

            
            print("===== RVT REPLAN DEBUG =====")
            print("current pos:", cur_pos[0].detach().cpu().numpy())
            print("goal pos   :", goal_pos[0].detach().cpu().numpy())
            print("distance   :", dist.item())
            print("============================")

            agent_pos_hist = torch.stack(list(self._agent_pos_history), dim=1)   # (B,2,8)

            scene_1024_hist = torch.stack(list(self._scene_pcd_history), dim=1)   # (B,2,1024,3)

            # current gripper pcd history
            gp_hist = agent_pos_hist[..., :7]     # (B,2,7)
            go_hist = agent_pos_hist[..., 7:8]    # (B,2,1)

            B, T, _ = gp_hist.shape
            gp_flat = gp_hist.reshape(B * T, 7)
            go_flat = go_hist.reshape(B * T, 1)

            cur_gripper_pcd = self.low.build_gripper_pcd_from_pose(
                gp_flat,
                go_flat,
                self.low.gripper_template_4.to(device),
            )

            cur_gripper_pcd_hist = cur_gripper_pcd.reshape(B, T, 4, 3)
            
            goal_gripper_pcd_hist = self.current_goal_pcd.unsqueeze(1).repeat(1, 2, 1, 1)

            lang_tokens = observation["lang_goal_tokens"].to(device=device)[:, 0].long()

            with torch.no_grad():
                _, lang_embs = _clip_encode_text(self.high.clip_model, lang_tokens)

            lang_vec = lang_embs.mean(dim=1).float()               # (1,512)
            lang_hist = lang_vec.unsqueeze(1).repeat(1, 2, 1)      # (1,2,512)

            policy_input = {
                "agent_pos": agent_pos_hist,              # (1, 2, 8)
                "point_cloud": scene_1024_hist,           # (1, 2, 1024, 3)
                "gripper_pcd": cur_gripper_pcd_hist,      # (1, 2, 4, 3)
                "goal_gripper_pcd": goal_gripper_pcd_hist,# (1, 2, 4, 3)
                "lang": lang_hist,                        # (1, 2, 512)
            }

            out = self.low.policy.predict_action(policy_input)
            action_seq = out["action"][0].detach().cpu().numpy()

            # ---- post-process actions ----
            for i in range(len(action_seq)):
                action_seq[i, 7] = 1.0 if action_seq[i, 7] > 0.5 else 0.0
                q = action_seq[i, 3:7]
                n = np.linalg.norm(q)
                action_seq[i, 3:7] = q / n if n > 1e-6 else np.array([0, 0, 0, 1.0])

            self._action_queue = list(action_seq[:self.exec_horizon])

        # ---------------------------------------------------------
        # 3. Execute ONE DP3 action
        # ---------------------------------------------------------
        action = self._action_queue.pop(0)
        print("DP3 action:", action)

        self._dp3_action_history.append({
            "step": int(step),
            "action": np.array(action, dtype=np.float32).copy(),  # (8,)
        })

        return ActResult(action)

    def reset(self):
        self.current_goal_pose = None
        self.current_goal_open = None
        self.current_goal_pcd  = None
        self._action_queue.clear()
        self._episode_started = False

        # logging
        self._rvt_goal_history.clear()
        self._dp3_action_history.clear()
        self._episode_id += 1

        self._agent_pos_history.clear()
        self._scene_pcd_history.clear()

    def update(self, *args, **kwargs):
        raise RuntimeError("Hierarchical agent does not support training.")

    def update_summaries(self, **kwargs):
        return {}

    def act_summaries(self, **kwargs):
        return {}

    def save_weights(self, path):
        # optional: save both sub-agents
        self.high.save_weights(path + "_high.pth")
        self.low.save_weights(path + "_low.pth")

    def load_weights(self, path):
        # optional: load both sub-agents
        self.high.load_weights(path + "_high.pth")
        self.low.load_weights(path + "_low.pth")