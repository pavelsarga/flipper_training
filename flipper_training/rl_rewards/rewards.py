from dataclasses import dataclass

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_rewards import Reward
from flipper_training.utils.geometry import quaternion_to_roll, quaternion_to_pitch, quaternion_to_yaw, normalized

__all__ = [
    "RollPitchGoal",
    "PotentialGoal",
    "PotentialGoalWithVelocityBonus",
    "PotentialGoalWithConditionalVelocityBonus",
    "PotentialGoalWithConditionalVelocityBonusAndJointCommandBonus",
    "GoalDistance",
    "PotentialGoalWithJointVelVariancePenalty",
    "PotentialGoalWithFinishVelocityPenalty",
    "PotentialGoalWithStepAscentBonus",
    "PotentialGoalWithPenaltiesConfigurable",
    "PotentialGoalWithJointVelVariancePenalty",
    "PotentialGoalWithFinishVelocityPenalty",
    "PotentialGoalWithStepAscentBonus",
    "PotentialGoalWithPenaltiesConfigurable",
]


@dataclass
class RollPitchGoal(Reward):
    """
    Reward function for the environment.
    Robot is rewarded for minimizing the roll and pitch angles and for moving towards the goal position,
    penalized for moving away from the goal position.
    """

    goal_reached_reward: float
    failed_reward: float
    omega_weight: float
    goal_weight: float

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
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        goal_diff_curr = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        goal_diff_prev = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        diff_delta = goal_diff_curr - goal_diff_prev
        reward = self.omega_weight * roll_pitch_rates_sq.sum(dim=-1, keepdim=True) - self.goal_weight * diff_delta
        reward[success] = self.goal_reached_reward
        reward[fail] = self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class GoalDistance(Reward):
    goal_reached_reward: float
    failed_reward: float
    weight: float
    exp: float | int = 1

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
        goal_diff = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True) / (self.env.terrain_cfg.max_coord * 2**1.5)
        reward = -self.weight * goal_diff.pow(self.exp)
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoal(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithVelocityBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    velocity_bonus_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef * curr_state.xd.norm(dim=-1, keepdim=True)
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithConditionalVelocityBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    velocity_bonus_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef
            * curr_state.xd.norm(dim=-1, keepdim=True)
            * (curr_dist <= prev_dist).float()  # award only if the robot is getting closer to the goal
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithConditionalVelocityBonusAndJointCommandBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    velocity_bonus_coef: float
    joint_command_bonus_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef
            * curr_state.xd.norm(dim=-1, keepdim=True)
            * (curr_dist <= prev_dist).float()  # award only if the robot is getting closer to the goal
            + action[..., action.shape[1] // 2 :].pow(2).sum(dim=-1, keepdim=True) * self.joint_command_bonus_coef
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithJointVelVariancePenalty(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    joint_vel_variance_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        joint_vel_variances = action[..., action.shape[1] // 2 :].abs().var(dim=-1, keepdim=True)
        reward -= self.joint_vel_variance_coef * joint_vel_variances
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithFinishVelocityPenalty(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    finish_velocity_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        reward[success] += self.goal_reached_reward - self.finish_velocity_coef * curr_state.xd[success].norm(dim=-1, keepdim=True)
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithStepAscentBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    step_bonus_coef: float

    def __post_init__(self):
        self.terrain_cfg = self.env.terrain_cfg
        if not self.terrain_cfg.grid_extras or "step_indices" not in self.terrain_cfg.grid_extras:
            raise ValueError("step_indices must be provided in the terrain_cfg.grid_extras for PotentialGoalWithStepAscentBonus.")
        self.B_range = torch.arange(self.env.phys_cfg.num_robots, device=self.env.device)

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        prev_xy = prev_state.x[..., :2]  # (B,2)
        curr_xy = curr_state.x[..., :2]  # (B,2)
        prev_ij = self.terrain_cfg.xy2ij(prev_xy)  # (B,2)
        curr_ij = self.terrain_cfg.xy2ij(curr_xy)  # (B,2)
        ti = self.terrain_cfg.grid_extras["step_indices"]  # (B,2)
        prev_idx = ti[self.B_range, *prev_ij.unbind(1)]  # (B,)
        curr_idx = ti[self.B_range, *curr_ij.unbind(1)]  # (B,)
        is_closer_mask = (curr_dist < prev_dist).float()  # (B,)
        reward += (
            self.step_bonus_coef * (curr_idx != prev_idx).float().unsqueeze(-1) * is_closer_mask
        )  # (B,1) # For the robots that got closer and changed step
        reward += self.goal_reached_reward * success.float().unsqueeze(-1)  # (B,1)
        reward += self.failed_reward * fail.float().unsqueeze(-1)  # (B,1)
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithPenaltiesConfigurable(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    joint_vel_variance_coef: float | None = None
    joint_angle_variance_coef: float | None = None
    track_vel_variance_coef: float | None = None
    roll_coef: float | None = None
    roll_rate_coef: float | None = None
    pitch_coef: float | None = None
    pitch_rate_coef: float | None = None

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        if self.joint_vel_variance_coef is not None:
            joint_vel_variances = action[..., action.shape[1] // 2 :].abs().var(dim=-1, keepdim=True)
            reward -= self.joint_vel_variance_coef * joint_vel_variances
        if self.joint_angle_variance_coef is not None:
            reward -= self.joint_angle_variance_coef * curr_state.thetas.abs().var(dim=-1, keepdim=True)
        if self.track_vel_variance_coef is not None:
            track_vel_variances = action[..., : action.shape[1] // 2].abs().var(dim=-1, keepdim=True)
            reward -= self.track_vel_variance_coef * track_vel_variances
        if self.roll_coef is not None:
            reward -= self.roll_coef * quaternion_to_roll(curr_state.q).abs().unsqueeze(-1) / torch.pi
        if self.roll_rate_coef is not None:
            roll_rate = curr_state.omega[..., 0, None] / torch.pi
            reward -= self.roll_rate_coef * roll_rate.abs()
        if self.pitch_coef is not None:
            reward -= self.pitch_coef * quaternion_to_pitch(curr_state.q).abs().unsqueeze(-1) / torch.pi
        if self.pitch_rate_coef is not None:
            pitch_rate = curr_state.omega[..., 1, None] / torch.pi
            reward -= self.pitch_rate_coef * pitch_rate.abs()
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithSideLatentPreference(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float
    side_bonus_coef: float

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
        latent_control_params = getattr(self.env, "latent_control_params", None)
        if latent_control_params is None:
            raise ValueError("latent_control_params must be provided in the environment for PotentialGoalWithSideLatentPreference.")
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        start_goal_vec = goal_state.x[:, :2] - start_state.x[:, :2]
        robot_pos_planar = curr_state.x[:, :2]
        # Compute signed distance from robot to the line from start to goal (positive on one side, negative on the other)
        # The sign is determined by the cross product between (goal - start) and (robot - start)
        # If latent_control_params is provided, use its sign as the preferred side
        # Formula: signed_dist = ((goal - start) x (robot - start)) / ||goal - start||
        vec_goal = start_goal_vec  # (B,2)
        vec_robot = robot_pos_planar - start_state.x[:, :2]  # (B,2)
        cross = vec_goal[:, 0] * vec_robot[:, 1] - vec_goal[:, 1] * vec_robot[:, 0]  # (B,)
        norm = vec_goal.norm(dim=-1) + 1e-8  # (B,)
        signed_dist = cross / norm  # (B,)
        diff = (signed_dist * latent_control_params.squeeze(-1)).unsqueeze(-1)  # (B,1)
        diff.clamp_min_(0.0)  # Don't penalize the robot for being on the wrong side
        is_closer_mask = (curr_dist < prev_dist).float()
        reward += self.side_bonus_coef * diff * is_closer_mask
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalSimplified(Reward):
    termination_reward: float
    gamma: float
    step_penalty: float
    shaping_coef: float

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
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.shaping_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        reward[success] += self.termination_reward
        reward[fail] -= self.termination_reward
        return reward.to(self.env.out_dtype)
