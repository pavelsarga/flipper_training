from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorClass

from marv_rl_training.utils.geometry import unit_quaternion

__all__ = ["PhysicsState", "PhysicsStateDer"]

if TYPE_CHECKING:
    from flipper_training.configs.robot_config import RobotModelConfig


class PhysicsState(TensorClass):
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (num_robots, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (num_robots, 3).
        q (torch.Tensor): Orientation quaternion of the robot in the world frame. Shape (num_robots, 4).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (num_robots, 3).
        thetas (torch.Tensor): Angles of the movable joints. Shape (num_robots, num_driving_parts).
    """

    x: torch.Tensor
    xd: torch.Tensor
    q: torch.Tensor
    omega: torch.Tensor
    thetas: torch.Tensor

    @staticmethod
    def dummy(robot_model: "RobotModelConfig", **kwargs) -> "PhysicsState":
        """Create an empty dummy PhysicsState object with zero tensors.
        Some fields can be overridden by passing them as keyword arguments.
        """
        batch_size = kwargs.pop("batch_size", None)
        device = kwargs.pop("device", torch.get_default_device())
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError("Specified batch size does not match the shape of the tensors.")
            else:
                batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = dict(
            x=torch.zeros(batch_size, 3),
            xd=torch.zeros(batch_size, 3),
            q=unit_quaternion(batch_size, device=device),
            omega=torch.zeros(batch_size, 3),
            thetas=torch.zeros(batch_size, robot_model.num_driving_parts),
            batch_size=[batch_size],
        )
        return PhysicsState(**base | kwargs, device=device)


class PhysicsStateDer(TensorClass):
    """Physics State Derivative

    Attributes:
        xdd (torch.Tensor): Derivative of the velocity of the robot in the world frame. Shape (num_robots, 3).
        omega_d (torch.Tensor): Derivative of the angular velocity of the robot in the world frame. Shape (num_robots, 3).
        thetas_d (torch.Tensor): Angular velocities of the movable joints. Shape (num_robots, num_driving_parts).
        f_spring (torch.Tensor): Spring forces. Shape (num_robots, 3).
        f_friction (torch.Tensor): Friction forces. Shape (num_robots, 3).
        in_contact (torch.Tensor): Contact status. Shape (num_robots, num_pts).
        robot_points (torch.Tensor): Collision points of the robot in the world frame. Shape (num_robots, num_pts, 3).
        thrust_vectors (torch.Tensor): Thrust vectors. Shape (num_robots, num_pts, 3).
        torque (torch.Tensor): Torque generated on the robot's CoG. Shape (num_robots, 3).

    """

    xdd: torch.Tensor
    omega_d: torch.Tensor
    thetas_d: torch.Tensor
    f_spring: torch.Tensor
    f_friction: torch.Tensor
    in_contact: torch.Tensor
    thrust_vectors: torch.Tensor
    robot_points: torch.Tensor
    torque: torch.Tensor

    @staticmethod
    def dummy(robot_model: "RobotModelConfig", **kwargs) -> "PhysicsStateDer":
        """Create an empty dummy PhysicsStateDer object with zero tensors.
        Some fields can be overridden by passing them as keyword arguments.
        """
        batch_size = kwargs.pop("batch_size", None)
        device = kwargs.pop("device", torch.get_default_device())
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError("Specified batch size does not match the shape of the tensors.")
            else:
                batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = dict(
            xdd=torch.zeros(batch_size, 3),
            omega_d=torch.zeros(batch_size, 3),
            thetas_d=torch.zeros(batch_size, robot_model.num_driving_parts),
            f_spring=torch.zeros(batch_size, robot_model.n_pts, 3),
            f_friction=torch.zeros(batch_size, robot_model.n_pts, 3),
            in_contact=torch.zeros(batch_size, robot_model.n_pts, 1),
            robot_points=torch.zeros(batch_size, robot_model.n_pts, 3),
            thrust_vectors=torch.zeros(batch_size, robot_model.n_pts, 3),
            torque=torch.zeros(batch_size, 3),
            batch_size=[batch_size],
        )
        return PhysicsStateDer(**base | kwargs, device=device)
