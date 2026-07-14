from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import (
    inverse_quaternion,
    rotate_vector_by_quaternion,
    quaternion_to_euler,
)
from . import Observation
from .robot_state import LocalStateVectorEncoder


@dataclass
class LocalStateVectorWithAction(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

    supports_vecnorm = True

    def __post_init__(self):
        if self.apply_noise:
            if not isinstance(self.noise_scale, (float, torch.Tensor)):
                raise ValueError("Noise scale must be specified if apply_noise is True and must be a float or tensor.")
            if isinstance(self.noise_scale, float):
                self.noise_scale = torch.tensor([self.noise_scale], dtype=self.env.out_dtype, device=self.env.device)
            if self.noise_scale.shape[0] not in (1, self.dim):
                raise ValueError(f"Noise scale tensor must have shape (1,) or ({self.dim},) but got {self.noise_scale.shape}.")
        self.max_dist = self.env.terrain_cfg.max_coord * 2**1.5
        self.theta_range = self.env.robot_cfg.joint_limits[1] - self.env.robot_cfg.joint_limits[0]
        self.joint_vel_range = self.env.robot_cfg.joint_max_pivot_vels[1] - self.env.robot_cfg.joint_max_pivot_vels[0]

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        goal_vecs = self.env.goal.x - curr_state.x  # (n_robots, 3)
        inv_q = inverse_quaternion(curr_state.q)  # (n_robots, 4)
        goal_vecs_local = rotate_vector_by_quaternion(goal_vecs.unsqueeze(1), inv_q).squeeze(1)  # (n_robots, 3)
        goal_vecs_local /= self.max_dist
        xd_local = rotate_vector_by_quaternion(curr_state.xd.unsqueeze(1), inv_q).squeeze(1)
        xd_local /= self.max_dist
        omega_local = rotate_vector_by_quaternion(curr_state.omega.unsqueeze(1), inv_q).squeeze(1) / torch.pi
        thetas = (curr_state.thetas - self.env.robot_cfg.joint_limits[None, 0]) / self.theta_range.unsqueeze(0) * 2 - 1  # scale to [-1, 1]
        # (n_robots, num_driving_parts)
        rolls, pitches, _ = quaternion_to_euler(curr_state.q)
        rolls.div_(torch.pi)  # scale to [-1, 1]
        pitches.div_(torch.pi)  # scale to [-1, 1]
        action_obs = action.clone()
        obs = torch.cat(
            [
                rolls.unsqueeze(1),
                pitches.unsqueeze(1),
                xd_local,
                omega_local,
                thetas,
                goal_vecs_local,
                action_obs,
            ],
            dim=1,
        ).to(self.env.out_dtype)
        if self.apply_noise:
            noise = torch.randn_like(obs) * self.noise_scale.view(1, -1)
            obs.add_(noise)
        return obs

    def from_realistic_world(self, tensordict) -> torch.Tensor:
        """Deployment path: same layout as ``__call__`` — ``[roll, pitch, xd_local,
        omega_local, thetas, goal_vec_local, prev_action]`` — computed from the deploy node's
        raw tensordict (keys per ``robot_state.LocalStateVector.from_realistic_world``, whose
        quaternion handling and normalizations this mirrors; the ONE difference from that
        class is kept: thetas scaled to [-1, 1] here vs [0, 1] there, matching each
        ``__call__``). ``prev_action`` is the node-stashed previous policy action (the
        realistic analogue of ``__call__``'s ``action`` arg); zeros on the first tick.
        """
        goal_vec_local = tensordict["goal_vec_local"].view(1, 3) / self.max_dist
        xd_local = tensordict["xd_local"].view(1, 3) / self.max_dist
        omega_local = tensordict["omega_local"].view(1, 3) / torch.pi
        thetas = (tensordict["thetas"].view(1, -1) - self.env.robot_cfg.joint_limits[0]) / self.theta_range * 2 - 1
        # ros sends [x, y, z, w]
        q_ros = tensordict["quat"].view(-1, 4)
        w, x, y, z = q_ros[:, 3], q_ros[:, 0], q_ros[:, 1], q_ros[:, 2]
        sin_p = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
        pitch = torch.asin(sin_p).view(1, 1) / torch.pi
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)).view(1, 1) / torch.pi
        prev = tensordict.get("prev_action", None)
        n_act = 2 * self.env.robot_cfg.num_driving_parts
        if prev is None:
            action_obs = torch.zeros((1, n_act), device=self.env.device)
        else:
            action_obs = prev.to(self.env.device).view(1, n_act)
        obs = torch.cat(
            [roll, pitch, xd_local, omega_local, thetas, goal_vec_local, action_obs],
            dim=1,
        ).to(self.env.out_dtype)
        return obs

    @property
    def dim(self) -> int:
        """
        The dimension of the observation vector.
        """
        dim = 3  # velocity vector
        dim += 2  # roll and pitch angles
        dim += 3  # angular velocity vector
        dim += self.env.robot_cfg.num_driving_parts * 3  # joint angles and action
        dim += 3  # goal vector
        return dim

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> LocalStateVectorEncoder:
        return LocalStateVectorEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
