"""Inference module for FTR-trained policies (trained with train_ftr.py / Isaac Sim).

No Isaac Sim required at runtime — the 966-D observation is assembled from raw sensor
data following CrossingEnv._get_observations() exactly.

Observation layout (966-D):
  [0:945]   heightmap (45×21, mean-centred; resampled if input size differs)
  [945:947] roll, pitch (÷π)
  [947:950] linear velocity in robot frame (÷hmap_diag ≈ 2.483 m)
  [950:953] angular velocity in robot frame (÷π)
  [953:957] flipper positions normalised to [0, 1]
  [957:960] goal vector in robot frame (÷hmap_diag)
  [960:966] previous action [v, w, fl, fr, rl, rr]

Actions returned (6-D, raw policy output in [-1, 1]):
  [0]   track linear velocity
  [1]   track angular velocity
  [2:6] flipper velocity commands (incremental, degrees/step)
"""

import math

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.envs import VecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type

import flipper_training  # noqa: F401 — registers OmegaConf resolvers
from flipper_training.environment.ftr_env_adapter import OBS_KEY
from flipper_training.observations.ftr_flat_obs import FtrFlatObservation
from flipper_training.utils.logutils import get_terminal_logger


class _MinimalFtrEnv:
    """Minimal env stub so policy/encoder can be constructed without Isaac Sim."""

    def __init__(self, num_actions: int, device: torch.device, encoder_opts: dict):
        self.batch_size = torch.Size([1])
        self.device = device
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(1, num_actions),
            device=device,
            dtype=torch.float32,
        )
        self.observations = [FtrFlatObservation(env=self, encoder_opts=encoder_opts)]


class FtrPolicyInferenceModule:
    """Inference module for policies trained with train_ftr.py (FTR-Benchmark / Isaac Sim).

    Usage::

        module = FtrPolicyInferenceModule(
            config_path="logs/run/config.yaml",
            policy_weights_path="logs/run/weights/policy_final.pth",
            vecnorm_weights_path="logs/run/weights/vecnorm_final.pth",
            device="cuda:0",
        )
        action = module.infer_action(
            heightmap=heightmap_np,         # (H, W) float32
            heightmap_extent=None,          # unused (FTR uses fixed 2.25×1.05 m map)
            goal_vec_local=goal_np,         # (3,) float32, robot frame
            xd_local=linear_vel_np,         # (3,) float32
            omega_local=angular_vel_np,     # (3,) float32
            thetas=flipper_angles_np,       # (4,) float32
            quat=quaternion_ros_np,         # (4,) float32 [x,y,z,w]
        )
        # action: (6,) float32 [v, w, fl, fr, rl, rr]
    """

    # FTR map: 45 rows × 21 cols, 0.05 m/cell → 2.25 m × 1.05 m
    _HM_ROWS = 45
    _HM_COLS = 21
    _HM_RES = 0.05
    _HM_DIAG = math.sqrt((_HM_ROWS * _HM_RES) ** 2 + (_HM_COLS * _HM_RES) ** 2)  # ≈ 2.483 m

    def __init__(
        self,
        config_path: str | Path,
        policy_weights_path: str | Path,
        vecnorm_weights_path: str | Path | None = None,
        device: str = "cpu",
        num_actions: int = 6,
    ):
        """
        Args:
            config_path: Path to the training config.yaml saved alongside the run.
            policy_weights_path: Path to policy_final.pth (or any checkpoint).
            vecnorm_weights_path: Path to vecnorm_final.pth. Normalisation is skipped
                if not provided (not recommended for deployment).
            device: Inference device ("cpu" or "cuda:N").
            num_actions: Action space dimension. Default 6 = [v, w, fl, fr, rl, rr]
                for MARV with 4 independent flippers. Set to 4 if sync_flipper_control
                was enabled during training (= [v, w, front_flip, rear_flip]).
        """
        self.logger = get_terminal_logger("FtrPolicyInferenceModule")
        self.device = torch.device(device)

        self.cfg = OmegaConf.load(config_path)

        self.joint_limit = float(
            np.deg2rad(self.cfg.env_cfg_overrides.get("flipper_pos_max_deg", 60.0))
        )
        self.num_actions = num_actions

        # Build minimal fake env so the policy constructor can read action_spec + encoders
        fake_env = _MinimalFtrEnv(num_actions, self.device, self.cfg.ftr_obs_encoder_opts)

        # Build actor_value_wrapper (encoder + MLP are both baked into the actor)
        policy_cfg = self.cfg.policy_config(**self.cfg.policy_opts)
        actor_value_wrapper, _, _ = policy_cfg.create(
            env=fake_env,
            weights_path=str(policy_weights_path),
            device=self.device,
        )
        self.actor = actor_value_wrapper.get_policy_operator().eval()

        # Standalone VecNorm (no TransformedEnv wrapper needed)
        vecnorm_keys = [o.name for o in fake_env.observations if o.supports_vecnorm]
        self.vecnorm = VecNorm(in_keys=vecnorm_keys, **self.cfg.vecnorm_opts)
        if vecnorm_weights_path:
            self.vecnorm.load_state_dict(
                torch.load(str(vecnorm_weights_path), map_location=self.device), strict=False
            )
            self.logger.info(f"Loaded VecNorm from {vecnorm_weights_path}")
        else:
            self.logger.warning("No vecnorm_weights_path provided — running without normalisation.")
        self.vecnorm.eval()

        # Previous action buffer initialised to zeros (as in CrossingEnv._reset_idx)
        self._prev_action = torch.zeros(num_actions, dtype=torch.float32, device=self.device)

        self.logger.info(
            f"FtrPolicyInferenceModule ready — device={device}, num_actions={num_actions}, "
            f"joint_limit={np.rad2deg(self.joint_limit):.1f}°"
        )

    def reset(self):
        """Reset the previous-action buffer (call between episodes)."""
        self._prev_action.zero_()

    def _build_obs(
        self,
        heightmap: np.ndarray,
        goal_vec_local: np.ndarray,
        xd_local: np.ndarray,
        omega_local: np.ndarray,
        thetas: np.ndarray,
        quat: np.ndarray,
    ) -> torch.Tensor:
        """Assemble the 966-D FTR observation from raw sensor values."""
        import cv2

        # Heightmap: resample to 45×21 if needed, then mean-centre
        hm = heightmap.astype(np.float32)
        if hm.shape != (self._HM_ROWS, self._HM_COLS):
            hm = cv2.resize(hm, (self._HM_COLS, self._HM_ROWS), interpolation=cv2.INTER_LINEAR)
        hm_flat = (hm - hm.mean()).flatten()  # [945]

        # Roll, pitch from ROS quaternion convention (x, y, z, w)
        qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
        roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))

        obs = np.concatenate([
            hm_flat,                                                                           # [945]
            np.array([roll / np.pi, pitch / np.pi], dtype=np.float32),                        # [2]
            np.asarray(xd_local,     dtype=np.float32) / self._HM_DIAG,                       # [3]
            np.asarray(omega_local,  dtype=np.float32) / np.pi,                               # [3]
            (np.asarray(thetas, dtype=np.float32) + self.joint_limit) / (2.0 * self.joint_limit),  # [4]
            np.asarray(goal_vec_local, dtype=np.float32) / self._HM_DIAG,                     # [3]
            self._prev_action.cpu().numpy(),                                                   # [6]
        ])  # total: 966
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 966]

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent: list | None,
        goal_vec_local: np.ndarray,
        xd_local: np.ndarray,
        omega_local: np.ndarray,
        thetas: np.ndarray,
        quat: np.ndarray,
    ) -> np.ndarray:
        """Run one inference step.

        Args:
            heightmap: 2-D elevation map (any size; resampled internally to 45×21).
            heightmap_extent: Unused — FTR uses a fixed 2.25×1.05 m map. Kept for
                API compatibility with PPOPolicyInferenceModule.
            goal_vec_local: (3,) goal vector in robot frame (m).
            xd_local: (3,) linear velocity in robot frame (m/s).
            omega_local: (3,) angular velocity in robot frame (rad/s).
            thetas: (4,) flipper joint angles [FL, FR, RL, RR] in radians.
            quat: (4,) orientation quaternion in ROS convention [x, y, z, w].

        Returns:
            (6,) float32 array [v, w, fl, fr, rl, rr] — raw policy output in [-1, 1].
        """
        obs = self._build_obs(heightmap, goal_vec_local, xd_local, omega_local, thetas, quat)
        td = TensorDict({OBS_KEY: obs}, batch_size=[1], device=self.device)
        td = self.vecnorm(td)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = self.actor(td)
        action = td["action"].squeeze(0).detach()
        self._prev_action = action.to(self.device)
        return action.cpu().numpy()
