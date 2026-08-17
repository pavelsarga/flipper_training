"""Inference module for D3QN-trained flipper policies (AT-D3QN / ICM-D3QN, Pan et al. 2023).

No Isaac Sim required at runtime — pure torch reimplementation of ATD3QNEncoder /
ICMD3QNEncoder (rl_modules/atd3qn/atd3qn_observation.py, rl_modules/icmd3qn/icmd3qn_observation.py),
loading the deployed dueling-D3QN checkpoint directly (bare encoder state_dict, no wrapper).

Both baselines are structurally identical apart from:
  - STATE_DIM: 2 (ATD3QN: front/rear flipper angle) vs 3 (ICMD3QN: + chassis pitch)
  - flipper normalisation: (x + limit) / (2*limit) -> [0,1]  (ATD3QN)
                            x / limit                -> ~[-1,1]  (ICMD3QN)

Observation (15 terrain-band means + STATE_DIM), matching
{ATD3QN,ICMD3QN}Module.get_observations()/calc_scanned_height_maps() exactly:
  - terrain: (45,21) heightmap, centred on (robot_z - track_wheel_radius), row-grouped
    mean-pool by 3 over the FULL width -> 15 values (NOT FTR's flatten-and-local-mean).
  - flipper state: front/rear angle in FTR's logical (sign-flipped-front) convention,
    i.e. thetas([FL,FR,RL,RR]) * [-1,-1,1,1], taking indices [0,2] (front,rear) — the
    same convention FtrWheelArticulation.get_all_flipper_positions produces, which is
    exactly what env.flipper_positions holds under sync_flipper_control.

Action: argmax(Q) over 9 discrete {front,rear} x {-1,0,+1} combinations -> a fixed
±15 deg (pi/12) step ADDED to the live measured flipper angle (accumulate-onto-current,
matching ftr_env.py's flipper_control_mode: increment), then clamped to MARV's real
asymmetric physical limits (front [-90 deg, +80 deg], rear [-80 deg, +90 deg] — verified
against ftr_env.py's marv_asym clamp branch for sync_flipper_control + not
only_front_flipper: low=[-front_up,-back_down], high=[front_down,back_up]).

Returned in the SAME 6-D [v, w, fl, fr, rl, rr] shape as FtrPolicyInferenceModule, but
as ABSOLUTE RADIAN JOINT TARGETS (fl=fr=front target, rl=rr=rear target), not [-1,1]
policy output — the caller (flipper_policy_node.py) publishes these directly to
/flippers_cmd_pos/*, not through the PPO-family's velocity-conversion path.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

from marv_rl_training.utils.logutils import get_terminal_logger


class _DuelingQNet(nn.Module):
    """Bare reimplementation of ATD3QNEncoder / ICMD3QNEncoder — identical architecture,
    only STATE_DIM (and therefore fusion_mlp's input width) differs."""

    def __init__(self, hm_dim: int, state_dim: int, hm_hidden: int, hm_out: int, fusion_hidden: int, num_actions: int):
        super().__init__()
        self.hm_dim = hm_dim
        self.terrain_mlp = nn.Sequential(
            nn.Linear(hm_dim, hm_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hm_hidden, hm_out),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hm_out + state_dim, fusion_hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.value_head = nn.Linear(fusion_hidden, 1)
        self.advantage_head = nn.Linear(fusion_hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[..., : self.hm_dim]
        e = x[..., self.hm_dim:]
        h_prime = self.terrain_mlp(h)
        s_prime = self.fusion_mlp(torch.cat([h_prime, e], dim=-1))
        value = self.value_head(s_prime)
        advantage = self.advantage_head(s_prime)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class D3QNPolicyInferenceModule:
    """Inference module for atd3qn/icmd3qn checkpoints (train_d3qn.py / Isaac Sim).

    Usage::

        module = D3QNPolicyInferenceModule(
            config_path="logs/run/config.yaml",
            policy_weights_path="logs/run/weights/policy_final.pth",
            device="cpu",
        )
        action = module.infer_action(
            heightmap=heightmap_np,   # (H, W) float32, any size (resampled to 45x21)
            robot_z=robot_z_m,        # float, robot base height in the map's frame
            thetas=flipper_angles_np, # (4,) float32 [FL, FR, RL, RR] radians, raw joints
            quat=quaternion_ros_np,   # (4,) float32 [x, y, z, w] — only used if pitch needed
        )
        # action: (6,) float32 [v, w, fl, fr, rl, rr] — v/w constants, fl/fr/rl/rr are
        # ABSOLUTE radian joint targets (fan out from the 2-DOF front/rear pair).
    """

    _HM_ROWS = 45
    _HM_COLS = 21
    _HM_DIM = 15  # after row-grouped mean-pool by 3

    def __init__(
        self,
        config_path: str | Path,
        policy_weights_path: str | Path,
        device: str = "cpu",
        **_ignored,  # accepts/ignores vecnorm_weights_path etc. for call-site compatibility
    ):
        self.logger = get_terminal_logger("D3QNPolicyInferenceModule")
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)

        overrides = self.cfg.env_cfg_overrides
        module_name = overrides.get("module_name")
        if module_name not in ("atd3qn", "icmd3qn"):
            raise ValueError(f"D3QNPolicyInferenceModule requires module_name atd3qn/icmd3qn, got {module_name!r}")
        self.is_icm = module_name == "icmd3qn"
        self.state_dim = 3 if self.is_icm else 2

        popts = self.cfg.get("policy_opts", {})
        self.track_vel = float(popts.get("track_vel", 0.6))
        hm_hidden = int(popts.get("hm_hidden", 32))
        hm_out = int(popts.get("hm_out", 4))
        fusion_hidden = int(popts.get("fusion_hidden", 16))

        self.track_wheel_radius = float(overrides.get("render_radius", 0.1165))
        # Observation-normalisation joint limit — deg2rad(flipper_pos_max_deg), default
        # 90.0 (neither atd3qn nor icmd3qn config overrides it).
        self.obs_joint_limit = float(np.deg2rad(overrides.get("flipper_pos_max_deg", 90.0)))
        # Physical clamp — asymmetric MARV limits (degrees), matching ftr_env.py's
        # marv_asym branch for sync_flipper_control + not only_front_flipper:
        #   low = [-front_up, -back_down], high = [front_down, back_up]
        front_up = float(overrides.get("marv_flipper_front_up_deg", 90.0))
        front_down = float(overrides.get("marv_flipper_front_down_deg", 80.0))
        back_up = float(overrides.get("marv_flipper_back_up_deg", 90.0))
        back_down = float(overrides.get("marv_flipper_back_down_deg", 80.0))
        self.front_low = -np.deg2rad(front_up)
        self.front_high = np.deg2rad(front_down)
        self.rear_low = -np.deg2rad(back_down)
        self.rear_high = np.deg2rad(back_up)
        self.step_rad = float(np.deg2rad(overrides.get("flipper_angle_increment_deg", 15.0)))

        self.q_net = _DuelingQNet(
            hm_dim=self._HM_DIM,
            state_dim=self.state_dim,
            hm_hidden=hm_hidden,
            hm_out=hm_out,
            fusion_hidden=fusion_hidden,
            num_actions=9,
        ).to(self.device)
        state_dict = torch.load(str(policy_weights_path), map_location=self.device)
        self.q_net.load_state_dict(state_dict, strict=True)
        self.q_net.eval()

        self.logger.info(
            f"D3QNPolicyInferenceModule ready — module={module_name}, device={device}, "
            f"track_vel={self.track_vel}, front=[{np.rad2deg(self.front_low):.1f},{np.rad2deg(self.front_high):.1f}]deg, "
            f"rear=[{np.rad2deg(self.rear_low):.1f},{np.rad2deg(self.rear_high):.1f}]deg, step={np.rad2deg(self.step_rad):.1f}deg"
        )

        # Debug/viz snapshot for callers with an RViz-style debug view (e.g.
        # marv_flipper_control_research's flipper_policy_node, which draws
        # /policy_heightmap_extent + highlights /policy_heightmap_image from this) --
        # this module owns the crop, so it's the only place that can record it.
        self.last_policy_heightmap: np.ndarray | None = None

    def reset(self):
        pass  # stateless — every tick re-derives the target from the live measured angle

    def _terrain_bands(self, heightmap_45x21: np.ndarray, robot_z: float) -> np.ndarray:
        hm = heightmap_45x21 - (robot_z - self.track_wheel_radius)
        # row-grouped mean over full width -> (15,)
        return hm.reshape(self._HM_DIM, 3, self._HM_COLS).mean(axis=(1, 2)).astype(np.float32)

    def infer_action(
        self,
        heightmap: np.ndarray,
        thetas: np.ndarray,
        robot_z: float = 0.0,
        quat: np.ndarray | None = None,
        heightmap_extent=None,
        **_ignored,  # accepts/ignores goal_vec_local, xd_local, omega_local
    ) -> np.ndarray:
        from marv_rl_training.training.ftr_heightmap_window import ftr_heightmap_window

        hm = ftr_heightmap_window(heightmap, heightmap_extent)
        self.last_policy_heightmap = hm
        bands = self._terrain_bands(hm, robot_z)

        # FTR logical [front, rear] convention: thetas*[-1,-1,1,1], indices [0, 2].
        thetas = np.asarray(thetas, dtype=np.float32)
        front_logical = -thetas[0]
        rear_logical = thetas[2]

        if self.is_icm:
            pitch = 0.0
            if quat is not None:
                qx, qy, qz, qw = (float(v) for v in quat)
                pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
            state = np.array([
                front_logical / self.obs_joint_limit,
                rear_logical / self.obs_joint_limit,
                pitch / (math.pi / 3.0),
            ], dtype=np.float32)
        else:
            state = np.array([
                (front_logical + self.obs_joint_limit) / (2.0 * self.obs_joint_limit),
                (rear_logical + self.obs_joint_limit) / (2.0 * self.obs_joint_limit),
            ], dtype=np.float32)

        obs = np.concatenate([bands, state]).astype(np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            q = self.q_net(obs_t)
        idx = int(q.argmax(dim=-1).item())
        i = idx // 3 - 1  # front: {-1, 0, 1}
        j = idx % 3 - 1   # rear:  {-1, 0, 1}

        front_target = float(np.clip(front_logical + i * self.step_rad, self.front_low, self.front_high))
        rear_target = float(np.clip(rear_logical + j * self.step_rad, self.rear_low, self.rear_high))

        # Back to raw Gazebo joint convention (inverse of the front sign-flip) for FL/FR,
        # rear unchanged for RL/RR.
        fl = fr = -front_target
        rl = rr = rear_target

        return np.array([self.track_vel, 0.0, fl, fr, rl, rr], dtype=np.float32)
