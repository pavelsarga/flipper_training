"""
FTR-compatible reward for the flipper_training physics engine.

Mirrors CrossingEnv._get_rewards() from the FTR-benchmark, supporting all
reward terms present in ftr_config_new_v5.yaml:

  • Velocity-modulated potential shaping: coef * (prev_dist - gamma * curr_dist) * clamp(v_norm³, 0, 1)
  • Per-step penalty (with optional linear/exponential scheduler)
  • Movement (action) bonus (with optional scheduler)
  • Joint-velocity variance penalty
  • Joint-angle variance penalty
  • Roll / roll-rate penalty
  • Pitch / pitch-rate penalty
  • Clearance penalty (body height above ground)
  • Shock penalty (linear acceleration magnitude)
  • Terminal bonuses / penalties (goal, failure, timeout)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_rewards import Reward
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import (
    inverse_quaternion,
    rotate_vector_by_quaternion,
    quaternion_to_roll,
    quaternion_to_pitch,
)


@dataclass
class _SchedulerCfg:
    """Internal helper — carries one coefficient scheduler's parameters."""
    type: Literal["linear", "exponential"] = "linear"
    start_factor: float = 1.0
    end_factor: float = 1.0
    total_iters: int = 1

    def factor(self, iteration: int) -> float:
        if self.total_iters <= 0:
            return self.end_factor
        t = min(iteration, self.total_iters) / self.total_iters
        if self.type == "linear":
            return self.start_factor + (self.end_factor - self.start_factor) * t
        elif self.type == "exponential":
            if self.start_factor <= 0 or self.end_factor <= 0:
                raise ValueError("Exponential scheduler requires both start_factor and end_factor > 0")
            return self.start_factor * (self.end_factor / self.start_factor) ** t
        raise ValueError(f"Unknown scheduler type: {self.type!r}")


@dataclass
class FtrCompatReward(Reward):
    """FTR CrossingEnv reward re-implemented for the flipper_training physics engine.

    All coefficients match the ftr_config_new_v5.yaml defaults.  Setting any
    optional coefficient to None disables that reward component.

    Args:
        goal_reached_reward: One-time bonus when the robot reaches the goal.
        failed_reward: One-time penalty on rollover / out-of-range failure.
        timeout_penalty: Optional one-time penalty on episode timeout (None = disabled).
        shaping_coef: Coefficient for potential-based distance shaping.
        shaping_gamma: Discount in the shaping potential (should match GAE gamma).
        step_penalty: Fixed per-step penalty.  Annealed by step_penalty_scheduler if set.
        track_vel_max: Maximum track velocity (m/s) used to normalise v_norm.
        action_bonus_coef: Coefficient for the movement bonus.  None = disabled.
            Annealed by action_bonus_coef_scheduler if set.
        lin_to_flipper_action_ratio: Weight of the linear-velocity bonus vs flipper bonus.
        lin_action_ratio: Blend of commanded vs measured forward velocity for the bonus.
        flipper_pos_max_deg: Used to normalise flipper actions in the action bonus.
        joint_vel_variance_coef: Penalises variance of flipper velocity commands.  None = off.
        joint_angle_variance_coef: Penalises asymmetric front/rear flipper positions.  None = off.
        roll_coef: Penalises roll beyond 15° dead-zone.  None = off.
        roll_rate_coef: Penalises roll angular rate.  None = off.
        pitch_coef: Penalises pitch beyond 7.5° dead-zone.  None = off.
        pitch_rate_coef: Penalises pitch angular rate.  None = off.
        clearance_coef: Penalises body dragging close to the ground.  None = off.
        wheel_radius: Approximate track-wheel radius for clearance computation (metres).
        shock_coef: Penalises large linear accelerations.  None = off.
        shock_scale: Sigmoid inflection point for shock penalty (m/s²).
        step_penalty_scheduler: Optional dict with keys type/start_factor/end_factor/total_iters.
        action_bonus_coef_scheduler: Same format as step_penalty_scheduler.
    """

    # ── Terminal ───────────────────────────────────────────────────────────
    goal_reached_reward: float = 120.0
    failed_reward: float = -30.0
    timeout_penalty: float | None = None

    # ── Potential shaping ──────────────────────────────────────────────────
    shaping_coef: float = 40.0
    shaping_gamma: float = 0.999
    step_penalty: float = -0.15
    track_vel_max: float = 0.95

    # ── Movement bonus ─────────────────────────────────────────────────────
    action_bonus_coef: float | None = 0.8
    lin_to_flipper_action_ratio: float = 0.6
    lin_action_ratio: float = 0.7
    flipper_pos_max_deg: float = 60.0

    # ── Smoothness ─────────────────────────────────────────────────────────
    joint_vel_variance_coef: float | None = None
    joint_angle_variance_coef: float | None = 0.1

    # ── Stability ──────────────────────────────────────────────────────────
    roll_coef: float | None = 1.0
    roll_rate_coef: float | None = None
    pitch_coef: float | None = 1.5
    pitch_rate_coef: float | None = None
    clearance_coef: float | None = 0.05
    wheel_radius: float = 0.1165   # MARV track-wheel radius (m)
    shock_coef: float | None = None
    shock_scale: float = 10.0

    # ── Schedulers (None = disabled) ───────────────────────────────────────
    step_penalty_scheduler: dict | None = None
    action_bonus_coef_scheduler: dict | None = None

    # ── Internal state (not passed to __init__) ────────────────────────────
    _prev_positions: torch.Tensor = field(default=None, init=False, repr=False, compare=False)
    _prev_lin_vel: torch.Tensor = field(default=None, init=False, repr=False, compare=False)
    _step_penalty_sched: _SchedulerCfg | None = field(default=None, init=False, repr=False, compare=False)
    _action_bonus_sched: _SchedulerCfg | None = field(default=None, init=False, repr=False, compare=False)
    # Current (possibly annealed) values.
    _current_step_penalty: float = field(default=None, init=False, repr=False, compare=False)
    _current_action_bonus_coef: float | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        n = self.env.n_robots
        dev = self.env.device
        self._prev_positions = torch.zeros(n, 3, device=dev)
        self._prev_lin_vel   = torch.zeros(n, 3, device=dev)

        if self.step_penalty_scheduler is not None:
            self._step_penalty_sched = _SchedulerCfg(**self.step_penalty_scheduler)
        if self.action_bonus_coef_scheduler is not None and self.action_bonus_coef is not None:
            self._action_bonus_sched = _SchedulerCfg(**self.action_bonus_coef_scheduler)

        self._current_step_penalty = self.step_penalty
        self._current_action_bonus_coef = self.action_bonus_coef

    # ── Scheduler API ──────────────────────────────────────────────────────

    def scheduler_step(self, iteration: int) -> None:
        """Call once per training batch to anneal step_penalty and action_bonus_coef."""
        if self._step_penalty_sched is not None:
            self._current_step_penalty = self.step_penalty * self._step_penalty_sched.factor(iteration)
        if self._action_bonus_sched is not None and self.action_bonus_coef is not None:
            self._current_action_bonus_coef = self.action_bonus_coef * self._action_bonus_sched.factor(iteration)

    # ── Reward computation ─────────────────────────────────────────────────

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        dt = self.env.effective_dt

        # Body-frame forward velocity (for shaping velocity modulation and action bonus).
        inv_q = inverse_quaternion(curr_state.q)
        xd_body = rotate_vector_by_quaternion(curr_state.xd.unsqueeze(1), inv_q).squeeze(1)  # (B,3)
        v_forward = xd_body[:, 0]  # body-frame x = forward

        # ── Potential-based shaping ────────────────────────────────────────
        curr_dist = (goal_state.x[:, :2] - curr_state.x[:, :2]).norm(dim=-1)   # (B,)
        prev_dist = (goal_state.x[:, :2] - self._prev_positions[:, :2]).norm(dim=-1)
        v_norm = v_forward / self.track_vel_max
        vel_mod = v_norm.pow(3).clamp(min=0.0, max=1.0)
        r_shaping = self.shaping_coef * (prev_dist - self.shaping_gamma * curr_dist) * vel_mod
        reward = r_shaping + self._current_step_penalty

        # ── Movement (action) bonus ────────────────────────────────────────
        if self._current_action_bonus_coef is not None:
            lin_ratio  = self.lin_to_flipper_action_ratio
            flip_ratio = 1.0 - lin_ratio
            fp_max_rad = math.radians(self.flipper_pos_max_deg)
            # Blend commanded vs measured forward velocity.
            v_cmd = action[:, 0] / self.track_vel_max          # first track vel ≈ forward
            v_meas = v_norm
            v_blend = (v_cmd.pow(3) * self.lin_action_ratio
                       + v_meas.pow(3) * (1.0 - self.lin_action_ratio))
            # Flipper action magnitude.
            flipper_actions = action[:, 4:]  # last 4 entries = flipper vels
            flipper_norm = (flipper_actions / fp_max_rad).mean(dim=-1)
            r_action = self._current_action_bonus_coef * (
                v_blend.clamp(-1.0, 1.0) * lin_ratio
                + flipper_norm.pow(3).clamp(max=1.0) * flip_ratio
            )
            reward = reward + r_action

        # ── Joint-velocity variance penalty ───────────────────────────────
        if self.joint_vel_variance_coef is not None:
            flipper_vels = action[:, 4:]
            r_jvv = self.joint_vel_variance_coef * flipper_vels.abs().var(dim=-1)
            reward = reward - r_jvv

        # ── Joint-angle variance penalty ──────────────────────────────────
        if self.joint_angle_variance_coef is not None:
            fp_max_rad = math.radians(self.flipper_pos_max_deg)
            th = curr_state.thetas  # (B, 4): FL, FR, RL, RR
            asym_front = (th[:, 0] - th[:, 1]).abs() / fp_max_rad
            asym_rear  = (th[:, 2] - th[:, 3]).abs() / fp_max_rad
            r_jav = self.joint_angle_variance_coef * (asym_front + asym_rear) / 2.0
            reward = reward - r_jav

        # ── Roll penalty ──────────────────────────────────────────────────
        if self.roll_coef is not None:
            roll = quaternion_to_roll(curr_state.q)
            roll_norm = 4.0 * (roll.abs() - math.radians(15)) / math.pi
            r_roll = self.roll_coef * roll_norm.clamp(min=0.0, max=1.0)
            reward = reward - r_roll

        # ── Roll-rate penalty ─────────────────────────────────────────────
        if self.roll_rate_coef is not None:
            omega_body = rotate_vector_by_quaternion(curr_state.omega.unsqueeze(1), inv_q).squeeze(1)
            r_roll_rate = self.roll_rate_coef * omega_body[:, 0].abs() / math.pi
            reward = reward - r_roll_rate

        # ── Pitch penalty ─────────────────────────────────────────────────
        if self.pitch_coef is not None:
            pitch = quaternion_to_pitch(curr_state.q)
            pitch_norm = 4.0 * (pitch.abs() - math.radians(7.5)) / math.pi
            r_pitch = self.pitch_coef * pitch_norm.clamp(min=0.0, max=1.0)
            reward = reward - r_pitch

        # ── Pitch-rate penalty ────────────────────────────────────────────
        if self.pitch_rate_coef is not None:
            if self.roll_rate_coef is None:
                omega_body = rotate_vector_by_quaternion(curr_state.omega.unsqueeze(1), inv_q).squeeze(1)
            r_pitch_rate = self.pitch_rate_coef * omega_body[:, 1].abs() / math.pi
            reward = reward - r_pitch_rate

        # ── Clearance penalty ─────────────────────────────────────────────
        if self.clearance_coef is not None:
            ground_h = interpolate_grid(
                self.env.terrain_cfg.z_grid,
                curr_state.x[:, :2].unsqueeze(1),
                self.env.terrain_cfg.max_coord,
            ).squeeze()  # (B,)
            clearance = curr_state.x[:, 2] - self.wheel_radius - ground_h
            r_clearance = self.clearance_coef * torch.sigmoid(-(clearance - 0.2) / 0.02)
            reward = reward - r_clearance

        # ── Shock penalty ─────────────────────────────────────────────────
        if self.shock_coef is not None:
            accel_mag = (curr_state.xd - self._prev_lin_vel).norm(dim=-1) / dt
            shock_norm = 2.0 / (1.0 + torch.exp(-accel_mag / self.shock_scale)) - 1.0
            r_shock = self.shock_coef * shock_norm
            reward = reward - r_shock

        # ── Terminal bonuses ──────────────────────────────────────────────
        terminal = fail  # explosion / rollover are always "fail"
        reward[terminal] = 0.0
        reward[success] = reward[success] + self.goal_reached_reward
        reward[fail]    = reward[fail]    + self.failed_reward
        if self.timeout_penalty is not None:
            truncated = self.env.step_count >= self.env.step_limits
            reward[truncated & ~terminal & ~success] = (
                reward[truncated & ~terminal & ~success] + self.timeout_penalty
            )

        # ── Bookkeeping ───────────────────────────────────────────────────
        self._prev_positions[:] = curr_state.x.detach()
        self._prev_lin_vel[:]   = curr_state.xd.detach()

        return reward.unsqueeze(-1).to(self.env.out_dtype)  # (B, 1)

    # ── Reset / state_dict ────────────────────────────────────────────────

    def reset(self, reset_mask: torch.Tensor, training: bool) -> None:
        self._prev_positions[reset_mask] = 0.0
        self._prev_lin_vel[reset_mask]   = 0.0

    def state_dict(self) -> dict:
        return {
            "prev_positions": self._prev_positions,
            "prev_lin_vel":   self._prev_lin_vel,
            "step_penalty":   self._current_step_penalty,
            "action_bonus":   self._current_action_bonus_coef,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "prev_positions" in state_dict:
            self._prev_positions[:] = state_dict["prev_positions"]
        if "prev_lin_vel" in state_dict:
            self._prev_lin_vel[:] = state_dict["prev_lin_vel"]
        if "step_penalty" in state_dict:
            self._current_step_penalty = state_dict["step_penalty"]
        if "action_bonus" in state_dict:
            self._current_action_bonus_coef = state_dict["action_bonus"]
