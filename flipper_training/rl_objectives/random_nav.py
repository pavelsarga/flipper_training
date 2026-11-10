import torch
import logging
from tqdm import trange
from typing import Literal, override
from dataclasses import dataclass
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_euler
from flipper_training.heightmaps.multi_gaussian import MultiGaussianHeightmapGenerator
from flipper_training.heightmaps.flat import FlatHeightmapGenerator


@dataclass
class RandomNavigationObjective(BaseObjective):
    """
    Objective manager that generates start/goal positions for the robots in the environment.
    The start position is generated randomly within the suitable area of the world,
    while the goal position is generated randomly within the suitable area of the world
    and at a minimum distance from the start position, but not further than the maximum distance.
    It is ensured that the goal position is not too high above the start position (to avoid unreachable goals).
    """

    higher_allowed: float
    min_dist_to_goal: float
    max_dist_to_goal: float
    goal_reached_threshold: float
    start_z_offset: float
    goal_z_offset: float
    iteration_limit_factor: float
    max_feasible_roll: float
    max_feasible_pitch: float
    start_position_orientation: Literal["random", "towards_goal"]
    init_joint_angles: torch.Tensor | Literal["max", "min", "random"] = "random"
    cache_size: int = 0
    _cache_cursor: int = 0
    supported_heightmap_generators = [MultiGaussianHeightmapGenerator, FlatHeightmapGenerator]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.terrain_config.grid_extras is None or "suitable_mask" not in self.terrain_config.grid_extras:
            raise ValueError("World configuration must contain a suitable mask in the grid extras for start/goal positions.")
        if self.cache_size > 0:
            self._init_cache()
        else:
            logging.warning("Cache size is 0, objective cannot be called on its own")

    def state_dict(self):
        return {
            "cache_cursor": self._cache_cursor,
        }

    def load_state_dict(self, state_dict):
        self._cache_cursor = state_dict["cache_cursor"]

    def _generate_cache_states(self, batch_index: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get valid indices for this robot's terrain (batch index b)
        valid_indices = torch.nonzero(self.terrain_config.grid_extras["suitable_mask"][batch_index], as_tuple=False).cpu()  # Shape: (N_valid_b, 2)
        n_valid = valid_indices.shape[0]
        # Oversample start and goal indices from valid set
        start_pos = torch.empty((count, 3), dtype=torch.float32)
        goal_pos = torch.empty((count, 3), dtype=torch.float32)
        oversample_factor = 1
        collected = 0
        while collected < count:
            remaining = count - collected
            n_samples = int(remaining * oversample_factor)
            start_idx = torch.randperm(n_valid, generator=self.rng)[:n_samples].cpu()
            goal_idx = torch.randperm(n_valid, generator=self.rng)[:n_samples].cpu()
            # Convert to ij coordinates for this robot, adding batch dimension
            start_ij = valid_indices[start_idx]  # Shape: (n_samples, 2)
            goal_ij = valid_indices[goal_idx]  # Shape: (n_samples, 2)
            # Compute xyz coordinates using this robot's terrain data
            start_xyz = torch.stack(
                [g[batch_index, *start_ij.unbind(-1)] for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]],
                dim=-1,
            ).to("cpu")
            goal_xyz = torch.stack(
                [g[batch_index, *goal_ij.unbind(-1)] for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]],
                dim=-1,
            ).to("cpu")
            # Validate pairs (ensure batch dimension is respected in _is_start_goal_xyz_valid)
            valid_mask = self._is_start_goal_xyz_valid(start_xyz.unsqueeze(1), goal_xyz.unsqueeze(1)).squeeze(1)
            valid_start = start_xyz[valid_mask]
            valid_goal = goal_xyz[valid_mask]
            n_new = min(valid_start.shape[0], remaining)
            # Store in cache for this robot (batch index b)
            start_pos[collected : collected + n_new] = valid_start[:remaining]
            goal_pos[collected : collected + n_new] = valid_goal[:n_new]
            collected += n_new
            oversample_factor *= 2

        return start_pos, goal_pos

    def _init_cache(self) -> None:
        B = self.physics_config.num_robots
        total_needed_per_robot = self.cache_size  # Number of cache entries per robot
        # Initialize cache tensors
        start_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)
        goal_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)

        # Process each robot's terrain separately
        for b in trange(B, desc="Initializing start/goal position cache"):
            start_pos_cache[:, b], goal_pos_cache[:, b] = self._generate_cache_states(b, total_needed_per_robot)

        # Store the cache
        self.cache = {
            "start": start_pos_cache,
            "goal": goal_pos_cache,
            "ori": self._get_initial_orientation_quat(start_pos_cache, goal_pos_cache),
            "step_limits": self._compute_step_limits(start_pos_cache, goal_pos_cache),
            "joint_angles": torch.stack([self._get_initial_joint_angles(self.physics_config.num_robots) for _ in range(self.cache_size)], dim=0),
        }
        self._cache_cursor = 0

    def _get_initial_joint_angles(self, count: int) -> torch.Tensor:
        high = self.robot_model.joint_limits[None, 1].cpu()
        low = self.robot_model.joint_limits[None, 0].cpu()
        match self.init_joint_angles:
            case "max":
                return high.repeat(count, 1)
            case "min":
                return low.repeat(count, 1)
            case "random":
                ang = torch.rand((count, self.robot_model.num_driving_parts), generator=self.rng) * (high - low) + low
                return ang.clamp(min=low, max=high)
            case torch.Tensor():
                if self.init_joint_angles.numel() != self.robot_model.num_driving_parts:
                    raise ValueError(f"Invalid shape for init_joint_angles: {self.init_joint_angles.shape}")
                ang = self.init_joint_angles.repeat(count, 1)
                return ang.clamp(min=low, max=high)
            case _:
                raise ValueError("Invalid value for init_joint_angles.")

    def _construct_full_start_goal_states(
        self, start_pos: torch.Tensor, goal_pos: torch.Tensor, oris: torch.Tensor, thetas: torch.Tensor
    ) -> tuple[PhysicsState, PhysicsState]:
        """
        Constructs the full start and goal states for the robots in the
        environment by adding the initial orientation and drop height.
        """
        start_state = PhysicsState.dummy(
            robot_model=self.robot_model,
            batch_size=start_pos.shape[0],
            device=self.device,
            x=start_pos.to(self.device),
            q=oris,
            thetas=thetas.to(self.device),
        )
        goal_state = PhysicsState.dummy(
            robot_model=self.robot_model, batch_size=goal_pos.shape[0], device=self.device, x=goal_pos.to(self.device), q=oris
        )
        start_state.x[..., 2] += self.start_z_offset
        goal_state.x[..., 2] += self.goal_z_offset
        return start_state, goal_state

    def _is_start_goal_xyz_valid(self, start_xyz: torch.Tensor, goal_xyz: torch.Tensor) -> torch.BoolTensor:
        """
        Checks if the start and goal positions are valid based on the suitability mask of the world configuration.

        Args:
        - start_xyz: Tensor of shape (B, 3) representing the start positions.
        - goal_xyz: Tensor of shape (B, 3) representing the goal positions.

        Returns:
        - A boolean tensor indicating whether each start/goal pair is valid.
        """
        dist = torch.linalg.norm(start_xyz - goal_xyz, dim=-1)  # distance
        z_diff = goal_xyz[..., 2] - start_xyz[..., 2]  # height difference
        start_xy_to_edge = self.terrain_config.max_coord - start_xyz[..., :2].abs()  # vector from start to edge
        goal_xy_to_edge = self.terrain_config.max_coord - goal_xyz[..., :2].abs()  # vector from goal to edge
        return (
            (dist >= self.min_dist_to_goal)
            & (dist <= self.max_dist_to_goal)
            & (z_diff <= self.higher_allowed)
            & (start_xy_to_edge.min(dim=-1).values > (2 * self.robot_model.radius))
            & (goal_xy_to_edge.min(dim=-1).values > (2 * self.robot_model.radius))
        )

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the iteration limits for the robots.
        """
        if self._cache_cursor < self.cache_size:
            start_state, goal_state = self._construct_full_start_goal_states(
                self.cache["start"][self._cache_cursor],
                self.cache["goal"][self._cache_cursor],
                self.cache["ori"][self._cache_cursor],
                self.cache["joint_angles"][self._cache_cursor],
            )
            step_limits = self.cache["step_limits"][self._cache_cursor].to(self.device)
            self._cache_cursor += 1
            return start_state, goal_state, step_limits
        else:
            logging.warning("Start/goal cache exhausted, doubling size and generating new start/goal positions")
            self.cache_size *= 2
            self._init_cache()
            return self.generate_start_goal_states()

    def _get_initial_orientation_quat(self, starts_x: torch.Tensor, goals_x: torch.Tensor) -> torch.Tensor:
        match self.start_position_orientation:
            case "towards_goal":
                diff_vecs = goals_x[..., :2] - starts_x[..., :2]
                ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
            case "random":
                ori = torch.rand(starts_x.shape[0], device=starts_x.device) * 2 * torch.pi  # random orientation
            case _:
                raise ValueError(f"Invalid start_position_orientation: {self.start_position_orientation}")
        return euler_to_quaternion(torch.zeros_like(ori), torch.zeros_like(ori), ori)

    @override
    def check_reached_goal(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        return torch.linalg.norm(state.x - goal.x, dim=-1) <= self.goal_reached_threshold

    @override
    def check_terminated_wrong(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        rolls, pitches, _ = quaternion_to_euler(state.q)
        return (pitches.abs() > self.max_feasible_pitch) | (rolls.abs() > self.max_feasible_roll)

    def _compute_step_limits(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.IntTensor:
        dists = torch.linalg.norm(goal_pos - start_pos, dim=-1)  # distances from starts to goals
        fastest_traversal = dists / (self.robot_model.v_max * self.env.effective_dt)  # time to reach the furthest goal
        steps = (fastest_traversal * self.iteration_limit_factor).ceil()
        return steps.int()

    def start_goal_to_simview(self, start: PhysicsState, goal: PhysicsState):
        try:
            from simview import SimViewStaticObject, BodyShapeType
        except ImportError:
            logging.warning("SimView is not installed. Cannot visualize start/goal positions.")
            return {}

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
