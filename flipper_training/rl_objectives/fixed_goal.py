import torch
import logging
from typing import override, Literal
from dataclasses import dataclass
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_euler


@dataclass
class FixedStartGoalNavigation(BaseObjective):
    """
    Fixed start/goal navigation objective for the robots in the environment.
    """

    start_x_y_z: torch.Tensor
    goal_x_y_z: torch.Tensor
    iteration_limit: int
    max_feasible_pitch: float
    max_feasible_roll: float
    goal_reached_threshold: float
    init_joint_angles: torch.Tensor | Literal["max", "min", "random"]
    resample_random_joint_angles_on_reset: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.start_pos = self.start_x_y_z.repeat(self.physics_config.num_robots, 1)
        self.goal_pos = self.goal_x_y_z.repeat(self.physics_config.num_robots, 1)
        diff_vecs = self.goal_pos[..., :2] - self.start_pos[..., :2]
        self.initial_q = euler_to_quaternion(
            torch.zeros_like(diff_vecs[..., 0]),
            torch.zeros_like(diff_vecs[..., 0]),
            torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0]),
        )
        self.initial_joint_angles = self._get_initial_joint_angles(self.physics_config.num_robots)

    def _get_initial_joint_angles(self, count: int) -> torch.Tensor:
        match self.init_joint_angles:
            case "max":
                return self.robot_model.joint_limits[None, 1].to(self.device).repeat(count, 1)
            case "min":
                return self.robot_model.joint_limits[None, 0].to(self.device).repeat(count, 1)
            case "random":
                ang = (
                    torch.rand((count, self.robot_model.num_driving_parts), device=self.device)
                    * (self.robot_model.joint_limits[None, 1] - self.robot_model.joint_limits[None, 0])
                    + self.robot_model.joint_limits[None, 0]
                )
                ang = ang.clamp(
                    min=self.robot_model.joint_limits[None, 0].to(self.device),
                    max=self.robot_model.joint_limits[None, 1].to(self.device),
                )
                return ang
            case torch.Tensor():
                if len(self.init_joint_angles) != self.robot_model.num_driving_parts:
                    raise ValueError(
                        f"Invalid shape for init_joint_angles: {self.init_joint_angles.shape}. Expected {self.robot_model.num_driving_parts}."
                    )
                ang = self.init_joint_angles.to(self.device).repeat(count, 1)
                ang = ang.clamp(
                    min=self.robot_model.joint_limits[None, 0].to(self.device),
                    max=self.robot_model.joint_limits[None, 1].to(self.device),
                )
                return ang
            case _:
                raise ValueError("Invalid value for init_joint_angles.")

    def _construct_full_start_goal_states(
        self,
    ) -> tuple[PhysicsState, PhysicsState]:
        """
        Constructs the full start and goal states for the robots in the
        environment by adding the initial orientation and drop height.
        """
        if self.resample_random_joint_angles_on_reset and self.init_joint_angles == "random":
            # BUGFIX (audit round-3): _get_initial_joint_angles requires the count,
            # same as the constructor call above -- calling it bare raised TypeError
            # whenever resample_random_joint_angles_on_reset was combined with
            # init_joint_angles="random" (e.g. test_configs/deterministic_flats_debug.yaml).
            self.initial_joint_angles = self._get_initial_joint_angles(self.physics_config.num_robots)
        start_state = PhysicsState.dummy(
            robot_model=self.robot_model,
            batch_size=self.physics_config.num_robots,
            device=self.device,
            x=self.start_pos.to(self.device),
            q=self.initial_q.to(self.device),
            thetas=self.initial_joint_angles.to(self.device),
        )
        goal_state = PhysicsState.dummy(
            robot_model=self.robot_model, batch_size=self.physics_config.num_robots, device=self.device, x=self.goal_pos.to(self.device)
        )
        return start_state, goal_state

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the iteration limits for the robots.
        """
        step_limits = torch.full((self.physics_config.num_robots,), self.iteration_limit, device=self.device).int()
        start_state, goal_state = self._construct_full_start_goal_states()
        return start_state, goal_state, step_limits

    @override
    def check_reached_goal(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        return torch.linalg.norm(state.x - goal.x, dim=-1) <= self.goal_reached_threshold

    @override
    def check_terminated_wrong(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        rolls, pitches, _ = quaternion_to_euler(state.q)
        return (pitches.abs() > self.max_feasible_pitch) | (rolls.abs() > self.max_feasible_roll)

    def start_goal_to_simview(self, start: PhysicsState, goal: PhysicsState):
        try:
            from simview import SimViewStaticObject, BodyShapeType
        except ImportError:
            logging.warning("SimView is not installed. Cannot visualize start/goal positions.")
            return []

        # start
        pos = start.x
        start_object = SimViewStaticObject.create_batched(
            name="Start",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#ff0000",
                }
                for i in range(pos.shape[0])
            ],
        )
        # goal
        pos = goal.x
        goal_object = SimViewStaticObject.create_batched(
            name="Goal",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#0000ff",
                }
                for i in range(pos.shape[0])
            ],
        )
        return [start_object, goal_object]
