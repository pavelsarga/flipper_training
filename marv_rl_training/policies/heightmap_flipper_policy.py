"""Deterministic heightmap-based flipper policy for evaluation.

Divides the robot-centric heightmap into front and rear regions (with a configurable
dead-band strip directly under the robot excluded), then further into left/right
quadrants.  Each quadrant drives the corresponding flipper proportionally:

  positive terrain signal → flipper UP   (over a bump)
  negative terrain signal → flipper DOWN (over a dip)

The terrain signal for a quadrant is the mean of the N most-extreme heights
(top-N max if the quadrant mean is positive, bottom-N min if negative), which
suppresses single-pixel outliers while still reacting to significant obstacles.

Flipper velocity sign convention (matches FTR-benchmark CrossingEnv):
  Front flippers: positive velocity = rotate UP, negative = rotate DOWN
  Rear  flippers: negative velocity = rotate UP, positive = rotate DOWN

Action layout: [v, w, fl, fr, rl, rr]  — all in [-1, 1]

Heightmap layout (45 rows × 21 cols):
  row 0   = front (+x, ahead of robot)
  row 44  = rear  (-x, behind robot)
  col 0   = left  (-y)
  col 20  = right (+y)

Sync mode (sync_flippers=True):
  Uses the full-width front and rear halves instead of per-side quadrants.
  All front flippers receive the same signal; same for all rear flippers.
  This is independent of the env's sync_flipper_control setting.

For evaluation only — does not require a critic or optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from flipper_training.policies import PolicyConfig


class _HeightmapActorModule(nn.Module):
    """Reads current_frame_height_maps from the live env and produces flipper actions."""

    def __init__(
        self,
        n_robots: int,
        n_flipper_actions: int,
        ftr_gym_env,                    # reference to the unwrapped gymnasium env
        linear_speed: float,
        angular_speed: float,
        n_extremes: int,
        gain: float,
        deadzone: float,
        return_gain: float,
        sync_flippers: bool,
        middle_band_frac: float,
    ) -> None:
        super().__init__()
        self._n_robots = n_robots
        self._n_flipper_actions = n_flipper_actions
        self._ftr_gym_env = ftr_gym_env
        self._action_dim = 2 + n_flipper_actions
        self._linear_speed = linear_speed
        self._angular_speed = angular_speed
        self._n_extremes = n_extremes
        self._gain = gain
        self._deadzone = deadzone
        self._return_gain = return_gain
        self._sync_flippers = sync_flippers
        self._middle_band_frac = middle_band_frac

    def _terrain_signal(self, region: torch.Tensor) -> torch.Tensor:
        """Compute per-robot terrain signal as the mean height of a sub-region.

        Args:
            region: [N, H_sub, W_sub] mean-subtracted heights.

        Returns:
            [N] signal — positive = above average terrain, negative = below.
        """
        return region.reshape(region.shape[0], -1).mean(dim=-1)

    def _flipper_vel(self, signal: torch.Tensor, pos: torch.Tensor, max_angle: float) -> torch.Tensor:
        """Choose terrain response or return-to-zero based on deadzone.

        Args:
            signal:    [N] terrain signal for this flipper.
            pos:       [N] current flipper angle (radians).
            max_angle: Joint limit in radians (for normalising return velocity).

        Returns:
            [N] velocity command in [-1, 1].
        """
        in_deadzone = signal.abs() < self._deadzone
        terrain_vel = torch.clamp(self._gain * signal, -1.0, 1.0)
        return_vel  = torch.clamp(-self._return_gain * pos / max_angle, -1.0, 1.0)
        return torch.where(in_deadzone, return_vel, terrain_vel)

    def forward(self) -> torch.Tensor:
        unwrapped = self._ftr_gym_env.unwrapped
        hmap_raw = unwrapped.current_frame_height_maps      # [N, 45, 21]
        N, H, W = hmap_raw.shape
        device = hmap_raw.device

        # Mean-subtract per robot so that 0 = average terrain level
        hmap = hmap_raw - hmap_raw.mean(dim=[-2, -1], keepdim=True)  # [N, 45, 21]

        # Current flipper positions [N, n_flippers] in radians — used for return-to-zero
        flipper_pos = unwrapped.flipper_positions.to(device)    # [N, 4]
        cfg = unwrapped.cfg
        max_angle = float(__import__("numpy").deg2rad(cfg.flipper_pos_max_deg)) if cfg.flipper_pos_max_deg else (3.14159 / 2.0)

        # Row slices — exclude middle band directly under the robot
        band_rows = max(2, int(H * self._middle_band_frac))
        half_band = band_rows // 2
        front_end  = H // 2 - half_band     # e.g. 20 for H=45, frac=0.10
        rear_start = H // 2 + half_band     # e.g. 24

        action = torch.zeros(N, self._action_dim, dtype=torch.float32, device=device)
        action[:, 0] = self._linear_speed
        action[:, 1] = self._angular_speed

        n_front = self._n_flipper_actions // 2
        n_rear  = self._n_flipper_actions - n_front

        if self._sync_flippers:
            sig_front = self._terrain_signal(hmap[:, :front_end, :])
            sig_rear  = self._terrain_signal(hmap[:, rear_start:, :])

            # Use mean position of each pair for return-to-zero
            pos_front = flipper_pos[:, :n_front].mean(dim=-1)
            pos_rear  = flipper_pos[:, n_front:].mean(dim=-1)

            front_vel = self._flipper_vel(sig_front, pos_front, max_angle)
            rear_vel  = self._flipper_vel(sig_rear,  pos_rear,  max_angle)

            action[:, 2:2 + n_front] = front_vel.unsqueeze(-1).expand(-1, n_front)
            action[:, 2 + n_front:]  = rear_vel.unsqueeze(-1).expand(-1, n_rear)
        else:
            # Per-quadrant — col 0..W//2-1 = left, col W//2+1..W-1 = right
            left_end    = W // 2
            right_start = W - W // 2   # = 11 for W=21

            sig_fl = self._terrain_signal(hmap[:, :front_end,  :left_end   ])
            sig_fr = self._terrain_signal(hmap[:, :front_end,  right_start:])
            sig_rl = self._terrain_signal(hmap[:, rear_start:, :left_end   ])
            sig_rr = self._terrain_signal(hmap[:, rear_start:, right_start:])

            action[:, 2] = self._flipper_vel(sig_fl, flipper_pos[:, 0], max_angle)
            action[:, 3] = self._flipper_vel(sig_fr, flipper_pos[:, 1], max_angle)
            action[:, 4] = self._flipper_vel(sig_rl, flipper_pos[:, 2], max_angle)
            action[:, 5] = self._flipper_vel(sig_rr, flipper_pos[:, 3], max_angle)

        return action


class _HeightmapWrapper:
    """Thin wrapper satisfying the ActorCriticWrapper interface used in eval_ftr.py."""

    def __init__(self, actor_tdm: TensorDictModule) -> None:
        self._actor_tdm = actor_tdm

    def get_policy_operator(self) -> TensorDictModule:
        return self._actor_tdm

    def eval(self) -> "_HeightmapWrapper":
        self._actor_tdm.eval()
        return self

    def parameters(self):
        return iter([])


@dataclass
class HeightmapFlipperPolicyConfig(PolicyConfig):
    """Deterministic heightmap-based flipper policy.

    Args:
        linear_speed: Forward speed forwarded to action[:,0].  Applied before any
            ``--const_linear_vel`` override in the eval script.
        angular_speed: Angular velocity forwarded to action[:,1].
        n_extremes: Number of extreme heights to average when computing the terrain
            signal for a quadrant.  Using the mean of the N most-extreme values
            rather than the global max/min suppresses single-pixel outliers.
        gain: Proportional gain mapping terrain signal → flipper velocity command.
            A value of 1.0 means a terrain anomaly equal to 1.0 (in normalised
            heightmap units) produces a ±1.0 velocity command (saturated).
        deadzone: Minimum absolute terrain signal required to trigger terrain-based
            control. When |signal| < deadzone the flipper instead uses a P-controller
            to return to 0 (horizontal). Prevents reacting to flat-ground noise.
        return_gain: Proportional gain for the return-to-zero controller applied
            inside the deadzone. Output = -return_gain * angle / max_angle.
        sync_flippers: If True, use full front/rear halves instead of per-side
            quadrants. All front flippers receive the same signal; same for rear.
        middle_band_frac: Fraction of heightmap rows to exclude as a dead-band
            directly under the robot (default 0.10 = 10 % of 45 rows ≈ 4 rows).
    """

    linear_speed: float = 0.8
    angular_speed: float = 0.0
    n_extremes: int = 10
    gain: float = 1.0
    deadzone: float = 0.05
    return_gain: float = 0.5
    sync_flippers: bool = False
    middle_band_frac: float = 0.10

    def create(self, env, **kwargs):
        action_spec = env.action_spec
        n_robots: int = action_spec.shape[0]
        action_dim: int = action_spec.shape[-1]
        n_flipper_actions: int = action_dim - 2

        # Access the underlying gymnasium env so we can read raw heightmaps each step
        ftr_gym_env = env.ftr_env

        device = getattr(env, "device", torch.device("cpu"))

        module = _HeightmapActorModule(
            n_robots=n_robots,
            n_flipper_actions=n_flipper_actions,
            ftr_gym_env=ftr_gym_env,
            linear_speed=self.linear_speed,
            angular_speed=self.angular_speed,
            n_extremes=self.n_extremes,
            gain=self.gain,
            deadzone=self.deadzone,
            return_gain=self.return_gain,
            sync_flippers=self.sync_flippers,
            middle_band_frac=self.middle_band_frac,
        ).to(device)

        actor_tdm = TensorDictModule(module, in_keys=[], out_keys=["action"])
        return _HeightmapWrapper(actor_tdm), [], []
