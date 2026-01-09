from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from tensordict import TensorDictBase
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import (
    inverse_quaternion,
    rotate_vector_by_quaternion,
    quaternion_to_euler,
)
from flipper_training.policies import MLP
from . import Observation, ObservationEncoder


class LocalStateVectorEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super(LocalStateVectorEncoder, self).__init__(output_dim)
        self.input_dim = input_dim
        self.mlp = MLP(**mlp_kwargs | {"in_dim": input_dim, "out_dim": output_dim, "activate_last_layer": True})

    def forward(self, x):
        return self.mlp(x)


@dataclass
class LocalStateVector(Observation):
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
        thetas = (curr_state.thetas - self.env.robot_cfg.joint_limits[None, 0]) / self.theta_range.unsqueeze(0)  # (n_robots, num_driving_parts)
        rolls, pitches, _ = quaternion_to_euler(curr_state.q)
        rolls.div_(torch.pi)
        pitches.div_(torch.pi)
        obs = torch.cat(
            [
                rolls.unsqueeze(1),
                pitches.unsqueeze(1),
                xd_local,
                omega_local,
                thetas,
                goal_vecs_local,
            ],
            dim=1,
        ).to(self.env.out_dtype)
        if self.apply_noise:
            noise = torch.randn_like(obs) * self.noise_scale.view(1, -1)
            obs.add_(noise)
        return obs

    def from_realistic_world(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        Convert observation from realistic world to local state vector format.
        The input tensordict is expected to contain the following keys:
        - "goal_vec_local": The goal vector in the robot's local frame (x forward, y left, z up).
        - "xd_local": The linear velocity vector in the robot's local frame.
        - "omega_local": The angular velocity vector in the robot's local frame.
        - "thetas": The joint angles of the robot.
        - "quaternion": The orientation of the robot as a quaternion.
        Args:
            tensordict (TensorDictBase): TensorDict containing the observation from the realistic world.
        Returns:
            torch.Tensor: Converted observation tensor.
        """
        goal_vec_local = tensordict["goal_vec_local"].view(1, 3) / self.max_dist
        xd_local = tensordict["xd_local"].view(1, 3) / self.max_dist
        omega_local = tensordict["omega_local"].view(1, 3) / torch.pi
        thetas = (tensordict["thetas"].view(1, -1) - self.env.robot_cfg.joint_limits[0]) / self.theta_range
        # ros sends [x, y, z, w]
        q_ros = tensordict["quat"].view(-1, 4)
        w = q_ros[:, 3]
        x = q_ros[:, 0]
        y = q_ros[:, 1]
        z = q_ros[:, 2]
        sin_p = 2 * (w * y - z * x)
        sin_p = torch.clamp(sin_p, -1.0, 1.0)
        pitch = torch.asin(sin_p)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        roll = roll.view(1, 1) / torch.pi
        pitch = pitch.view(1, 1) / torch.pi
        obs = torch.cat(
            [
                roll,
                pitch,
                xd_local,
                omega_local,
                thetas,
                goal_vec_local,
            ],
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
        dim += self.env.robot_cfg.num_driving_parts  # joint angles
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
