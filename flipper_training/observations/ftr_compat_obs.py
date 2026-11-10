"""
FTR-compatible flat observation for the flipper_training physics engine.

Produces a (B, 968) flat vector that mirrors the 966-D observation built by
FTR CrossingEnv._get_observations(), with two extra dimensions for the
additional prev_action entries (8-D in flipper_training vs 6-D in FTR):

  945  heightmap (45×21), mean-subtracted, relative to robot Z
    2  roll, pitch  (÷π)
    3  linear velocity in body frame  (÷hmap_diag)
    3  angular velocity in body frame  (÷π)
    4  flipper joint positions normalised to [0, 1]
    3  goal vector in body frame  (÷hmap_diag)
    8  previous action  [4 track vels, 4 flipper vels]
──────
  968  total

The encoder used is FtrCNNFlatEncoder with state_dim=23 (all non-heightmap dims).

The observation tracks prev_action as an internal buffer.  Robots that were
just reset (step_count == 0) have their prev_action zeroed automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.observations import Observation, ObservationEncoder
from flipper_training.observations.ftr_flat_obs import FtrCNNFlatEncoder
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import (
    quaternion_to_roll,
    quaternion_to_pitch,
    planar_rot_from_q,
    inverse_quaternion,
    rotate_vector_by_quaternion,
)

# Physical extent of the FTR perception grid (metres).
HM_ROWS = 45        # forward/backward  (2.25 m)
HM_COLS = 21        # lateral           (1.05 m)
HM_CELL = 0.05      # metres per cell
HM_LEN_X = HM_ROWS * HM_CELL   # 2.25 m
HM_LEN_Y = HM_COLS * HM_CELL   # 1.05 m
HM_DIAG = math.sqrt(HM_LEN_X**2 + HM_LEN_Y**2)   # ≈ 2.483 m

# Observation dimensions.
HM_DIM   = HM_ROWS * HM_COLS   # 945
STATE_DIM = 23                  # 2+3+3+4+3+8 (non-heightmap)
OBS_DIM   = HM_DIM + STATE_DIM  # 968


@dataclass
class FtrCompatObservation(Observation):
    """FTR-compatible flat observation using the flipper_training physics engine.

    Produces OBS_DIM=968 values per robot: 945-D mean-subtracted heightmap
    (45×21 @ 0.05 m/cell) concatenated with the same state features as FTR's
    CrossingEnv, plus an 8-D prev_action.

    Args:
        flipper_pos_max_deg: Maximum flipper angle in degrees used to normalise
            joint positions to [0, 1].  Should match env_cfg_overrides.flipper_pos_max_deg.
        hmap_noise_std: Standard deviation of Gaussian noise added to the heightmap
            (metres).  0.0 = no noise.
        obs_noise_std: Standard deviation of Gaussian noise added to state features
            (roll/pitch/vel/joints/goal).  0.0 = no noise.
    """

    supports_vecnorm: bool = True  # instance-level override (ClassVar not picklable in dataclass)

    flipper_pos_max_deg: float = 60.0
    hmap_noise_std: float = 0.1
    obs_noise_std: float = 0.05   # modest noise on non-heightmap state features

    # Internal state — not part of the dataclass constructor.
    _prev_action: torch.Tensor = field(default=None, init=False, repr=False, compare=False)
    _percep_grid: torch.Tensor = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        n = self.env.n_robots
        dev = self.env.device

        # Build the 45×21 local perception grid (robot-centric XY, metres).
        # Forward (x) runs from -HM_LEN_X/2 to +HM_LEN_X/2.
        # Lateral (y) runs from -HM_LEN_Y/2 to +HM_LEN_Y/2.
        x_space = torch.linspace(-HM_LEN_X / 2, HM_LEN_X / 2, HM_ROWS, device=dev)
        y_space = torch.linspace(-HM_LEN_Y / 2, HM_LEN_Y / 2, HM_COLS, device=dev)
        px, py = torch.meshgrid(x_space, y_space, indexing="ij")  # (45, 21)
        grid_pts = torch.stack([px, py], dim=-1).reshape(-1, 2)    # (945, 2)
        # Repeat for all robots: (B, 945, 2)
        self._percep_grid = grid_pts.unsqueeze(0).expand(n, -1, -1).clone()

        # Previous action buffer (initialised to zeros).
        # flipper_training action space is always 8-D: [4 track vels, 4 flipper vels].
        self._prev_action = torch.zeros(n, 8, device=dev, dtype=self.env.out_dtype)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        B = curr_state.x.shape[0]
        dev = curr_state.x.device

        # Zero prev_action for newly-reset robots (step_count==0 ↔ just reset).
        reset_mask = self.env.step_count == 0
        if reset_mask.any():
            self._prev_action[reset_mask] = 0.0

        # ── Heightmap (45×21) ───────────────────────────────────────────────
        # Rotate local grid points by robot yaw and translate by robot XY.
        R_yaw = planar_rot_from_q(curr_state.q)  # (B, 2, 2)
        global_pts = (
            torch.bmm(self._percep_grid, R_yaw.transpose(1, 2))  # (B, 945, 2)
            + curr_state.x[:, :2].unsqueeze(1)
        )
        # Interpolate terrain heights at global points → (B, 945, 1).
        z_vals = interpolate_grid(
            self.env.terrain_cfg.z_grid,
            global_pts,
            self.env.terrain_cfg.max_coord,
        )  # (B, 945, 1)
        hm = z_vals.squeeze(-1)  # (B, 945)

        # Make relative to robot Z.
        hm = hm - curr_state.x[:, 2:3]  # (B, 945)

        # Mean-subtraction (FTR normalisation).
        hm = hm - hm.mean(dim=-1, keepdim=True)

        # Optional noise.
        if self.hmap_noise_std > 0.0:
            hm = hm + torch.randn_like(hm) * self.hmap_noise_std

        # ── State features ──────────────────────────────────────────────────
        inv_q = inverse_quaternion(curr_state.q)  # (B, 4)

        # Roll and pitch (÷π).
        roll  = quaternion_to_roll(curr_state.q).unsqueeze(-1)  / math.pi  # (B,1)
        pitch = quaternion_to_pitch(curr_state.q).unsqueeze(-1) / math.pi  # (B,1)

        # Linear velocity in body frame (÷hmap_diag).
        xd_body = rotate_vector_by_quaternion(
            curr_state.xd.unsqueeze(1), inv_q
        ).squeeze(1) / HM_DIAG  # (B,3)

        # Angular velocity in body frame (÷π).
        omega_body = rotate_vector_by_quaternion(
            curr_state.omega.unsqueeze(1), inv_q
        ).squeeze(1) / math.pi  # (B,3)

        # Flipper joint positions → [0, 1].
        fp_max = math.radians(self.flipper_pos_max_deg)
        thetas_norm = (curr_state.thetas + fp_max) / (2.0 * fp_max)  # (B,4)
        thetas_norm = thetas_norm.clamp(0.0, 1.0)

        # Goal vector in body frame (÷hmap_diag).
        goal_world = self.env.goal.x - curr_state.x  # (B,3)
        goal_body  = rotate_vector_by_quaternion(
            goal_world.unsqueeze(1), inv_q
        ).squeeze(1) / HM_DIAG  # (B,3)

        # Optional noise on state features.
        if self.obs_noise_std > 0.0:
            roll       = roll       + torch.randn_like(roll)       * self.obs_noise_std
            pitch      = pitch      + torch.randn_like(pitch)      * self.obs_noise_std
            xd_body    = xd_body    + torch.randn_like(xd_body)    * self.obs_noise_std
            omega_body = omega_body + torch.randn_like(omega_body) * self.obs_noise_std
            thetas_norm = thetas_norm + torch.randn_like(thetas_norm) * (self.obs_noise_std * 0.2)
            goal_body  = goal_body  + torch.randn_like(goal_body)  * self.obs_noise_std

        # ── Concatenate ─────────────────────────────────────────────────────
        obs = torch.cat([
            hm,                          # 945
            roll,                        #   1
            pitch,                       #   1
            xd_body,                     #   3
            omega_body,                  #   3
            thetas_norm,                 #   4
            goal_body,                   #   3
            self._prev_action,           #   8  (from previous step)
        ], dim=-1).to(self.env.out_dtype)  # (B, 968)

        # NaN/Inf guard.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Update prev_action for next step.
        self._prev_action = action.detach().to(self.env.out_dtype)

        return obs

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, OBS_DIM),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> ObservationEncoder:
        opts = dict(self.encoder_opts or {})
        return FtrCNNFlatEncoder(
            state_dim=STATE_DIM,
            **opts,
        )
