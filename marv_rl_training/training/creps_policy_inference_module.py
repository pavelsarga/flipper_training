"""Inference module for the CREPS flipper controller (Pecka, Salansky, Zimmermann &
Svoboda 2016, "Autonomous Flipper Control with Safety Constraints").

Not a neural network — a deterministic 6-parameter LINEAR controller
(rl_modules/creps/creps_policy.py's CREPSLowerLevelPolicy), fixed omega loaded straight
from the trained checkpoint's `omega_mean`. Pure Python/NumPy, no torch needed.

    front = omega1 + omega2*pitch + omega3*height_ahead
    rear  = omega4 + omega5*pitch + omega6*height_ahead

matching rl_modules/creps/creps_module.py's observation (pitch, height_ahead) exactly:
  - pitch: raw radians from the base orientation.
  - height_ahead: a single heightmap cell ~0.20 m ahead of the robot (row
    height_ahead_row, default 18 of 45; center column), minus the robot's own ground
    reference (robot_z - track_wheel_radius) — the SAME reference the terrain-band
    reduction and FtrPolicyInferenceModule's heightmap centering both use.

front/rear are each clamped to [-1, 1] then mapped onto CREPS's symmetric ±90 deg limit
(ftr_env.py's position-control map: target = low + unit*(high-low), which for
low=-high=-pi/2 degenerates to target = pi/2 * clamp(raw, -1, 1)) — absolute radian
joint targets, fanned out [front,front,rear,rear] (sync_flipper_control).
"""

import math

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from marv_rl_training.training.ftr_heightmap_window import HM_ROWS, HM_COLS, HM_RES, ftr_heightmap_window
from marv_rl_training.utils.logutils import get_terminal_logger


class CREPSPolicyInferenceModule:
    """Inference module for the creps checkpoint (train_creps.py / Isaac Sim).

    Usage::

        module = CREPSPolicyInferenceModule(
            config_path="logs/run/config.yaml",
            policy_weights_path="logs/run/weights/creps_state_final.pth",
        )
        action = module.infer_action(
            heightmap=heightmap_np,   # (H, W) float32, any size (resampled to 45x21)
            robot_z=robot_z_m,        # float, robot base height in the map's frame
            quat=quaternion_ros_np,   # (4,) float32 [x, y, z, w]
        )
        # action: (6,) float32 [v, w, fl, fr, rl, rr] — v is the fixed forward speed,
        # w=0, fl=fr/rl=rr are ABSOLUTE radian joint targets.
    """

    def __init__(
        self,
        config_path: str | Path,
        policy_weights_path: str | Path,
        device: str = "cpu",
        **_ignored,
    ):
        self.logger = get_terminal_logger("CREPSPolicyInferenceModule")
        self.cfg = OmegaConf.load(config_path)
        overrides = self.cfg.env_cfg_overrides
        if overrides.get("module_name") != "creps":
            raise ValueError(f"CREPSPolicyInferenceModule requires module_name=creps, got {overrides.get('module_name')!r}")

        self.fixed_forward_vel = float(overrides.get("fixed_forward_vel", 0.4))
        self.track_wheel_radius = float(overrides.get("render_radius", 0.1165))

        # Symmetric limits — all four marv_flipper_*_deg fields equal 90.0 in the
        # deployed creps config; general position-control map: target = low+unit*(high-low).
        front_up = float(overrides.get("marv_flipper_front_up_deg", 90.0))
        front_down = float(overrides.get("marv_flipper_front_down_deg", 90.0))
        back_up = float(overrides.get("marv_flipper_back_up_deg", 90.0))
        back_down = float(overrides.get("marv_flipper_back_down_deg", 90.0))
        self.front_low, self.front_high = -np.deg2rad(front_up), np.deg2rad(front_down)
        self.rear_low, self.rear_high = -np.deg2rad(back_down), np.deg2rad(back_up)

        # height_ahead_row default (9) matches rl_modules/creps/creps_module.yaml;
        # not exposed via env_cfg_overrides, so hardcode the deployed value. Row 9 =
        # front flipper tip's flat-pose reach (0.6375m, from marv_sim.urdf) ahead of
        # base_link -- sampling right at the tip, not past it -- see
        # creps_module.yaml's comment for the full derivation/history (this constant
        # was 18, then 13, then 5, before settling on 9).
        self.height_ahead_row = 9

        ckpt = np.load(str(policy_weights_path), allow_pickle=True) if str(policy_weights_path).endswith(".npy") else None
        if ckpt is None:
            import torch
            state = torch.load(str(policy_weights_path), map_location="cpu", weights_only=False)
            omega = state["omega_mean"]
        else:
            omega = ckpt.item()["omega_mean"]
        self.omega = np.asarray(omega, dtype=np.float64).reshape(6)

        self.logger.info(
            f"CREPSPolicyInferenceModule ready — omega={self.omega.tolist()}, "
            f"fixed_forward_vel={self.fixed_forward_vel}"
        )
        # Debug/viz snapshot for callers with an RViz-style debug view (e.g.
        # marv_flipper_control_research's flipper_policy_node, which draws
        # /policy_heightmap_extent + /creps_debug_markers from these). None until
        # the first successful infer_action() call.
        self.last_policy_heightmap: np.ndarray | None = None
        self.last_diag: dict | None = None
        self._tick = 0  # throttles the diagnostic log below

    def reset(self):
        pass  # stateless

    def infer_action(
        self,
        heightmap: np.ndarray,
        robot_z: float = 0.0,
        quat: np.ndarray | None = None,
        heightmap_extent=None,
        **_ignored,
    ) -> np.ndarray:
        hm = ftr_heightmap_window(heightmap, heightmap_extent)
        self.last_policy_heightmap = hm
        col = HM_COLS // 2
        ground_ahead = float(hm[self.height_ahead_row, col])
        height_ahead = ground_ahead - (robot_z - self.track_wheel_radius)

        pitch = 0.0
        if quat is not None:
            qx, qy, qz, qw = (float(v) for v in quat)
            pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))

        front_raw = self.omega[0] + self.omega[1] * pitch + self.omega[2] * height_ahead
        rear_raw = self.omega[3] + self.omega[4] * pitch + self.omega[5] * height_ahead

        front_unit = np.clip(front_raw, -1.0, 1.0)
        rear_unit = np.clip(rear_raw, -1.0, 1.0)
        front_target = self.front_low + (front_unit + 1.0) * 0.5 * (self.front_high - self.front_low)
        rear_target = self.rear_low + (rear_unit + 1.0) * 0.5 * (self.rear_high - self.rear_low)

        fl = fr = float(front_target)
        rl = rr = float(rear_target)

        # x_ahead: along-track distance (m, base_link frame) of the ONE cell CREPS
        # actually reads -- row 0 = +1.10m front, center row 22 = robot, 0.05m/cell.
        self.last_diag = {
            "row": self.height_ahead_row, "col": col,
            "x_ahead": float((HM_ROWS // 2 - self.height_ahead_row) * HM_RES),
            "ground_ahead": ground_ahead, "height_ahead": height_ahead, "pitch": float(pitch),
            "front_raw": float(front_raw), "rear_raw": float(rear_raw),
            "front_target_rad": float(front_target), "rear_target_rad": float(rear_target),
        }
        self._tick += 1
        if self._tick % 10 == 0:
            self.logger.info(
                f"CREPS diag: pitch={math.degrees(pitch):.1f}deg height_ahead={height_ahead:.3f}m "
                f"-> front_raw={front_raw:.2f} rear_raw={rear_raw:.2f} (clip to [-1,1] before mapping) "
                f"target_deg=[front={math.degrees(front_target):.1f}, rear={math.degrees(rear_target):.1f}]"
            )

        return np.array([self.fixed_forward_vel, 0.0, fl, fr, rl, rr], dtype=np.float32)
