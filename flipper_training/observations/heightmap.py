from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import planar_rot_from_q

from . import Observation, ObservationEncoder

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class HeightmapEncoder(ObservationEncoder):
    def __init__(
        self,
        img_shape: tuple[int, int],
        output_dim: int,
        activate_output: bool = False,
        **kwargs,
    ):
        super(HeightmapEncoder, self).__init__(output_dim)
        self.img_shape = img_shape  # Keep for reference if needed, but not used in layer defs anymore
        # Define the sequential convolutional layers
        # Each block roughly corresponds to a downsampling stage in the original
        self.encoder = nn.Sequential(
            # Layer 1: Similar to the original stem but using 3x3 kernel
            # Input: (B, 1, H, W)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 16, H/2, W/2)
            # Layer 2: Downsample, increase channels
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 32, H/4, W/4)
            # Layer 3: Downsample, increase channels
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Output: (B, 32, H/8, W/8)
            nn.AdaptiveAvgPool2d((2, 2)),  # Pool to 2x2 spatial dimensions
            nn.Flatten(),  # Flatten features -> (B, 32 * 2 * 2)
            nn.Linear(4 * 32, output_dim),  # Linear layer -> (B, output_dim)
            nn.ReLU(inplace=True) if activate_output else nn.Identity(),
        )

    def forward(self, hm):
        # Handle potential time dimension (same as before)
        if hm.ndim > 4:
            B, T = hm.shape[:2]
            # Input shape expected: (B, T, C, H, W)
            C, H, W = hm.shape[2:]
            hm = hm.view(B * T, C, H, W)  # Use view for efficiency
            y_ter = self.encoder(hm)
            # Output shape expected: (B, T, output_dim)
            y_ter = y_ter.view(B, T, -1)
        else:
            # Input shape expected: (B, C, H, W)
            y_ter = self.encoder(hm)
            # Output shape expected: (B, output_dim)
        return y_ter


@dataclass
class Heightmap(Observation):
    """
    Generates heightmap observation from the environment using 2D transformations.
    """

    percep_shape: tuple[int, int]
    percep_extent: tuple[float, float, float, float]
    interval: tuple[float, float]
    shift: float | None = None
    normalize_to_interval: bool = False
    supports_vecnorm = False

    def __post_init__(self):
        if self.apply_noise and not isinstance(self.noise_scale, float):
            raise ValueError("Noise scale must be specified if apply_noise is True and must be a float.")
        self._initialize_perception_grid()

    def _initialize_perception_grid(self) -> None:
        """
        Initialize the 2D perception grid points.
        """
        x_space = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0])
        y_space = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1])
        px, py = torch.meshgrid(x_space, y_space, indexing="ij")
        # Store as 2D points (N, 2)
        percep_grid_points_2d = torch.dstack([px, py]).reshape(-1, 2)
        # Repeat for batch size (B, N, 2)
        self.percep_grid_points_2d = percep_grid_points_2d.unsqueeze(0).repeat(self.env.n_robots, 1, 1).to(self.env.device)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        B = curr_state.x.shape[0]
        # Get 2D rotation matrix from quaternion (B, 2, 2)
        R_yaw_2d = planar_rot_from_q(curr_state.q)

        # Rotate local 2D points (B, N, 2)
        rotated_points_2d = torch.bmm(self.percep_grid_points_2d, R_yaw_2d.transpose(1, 2))

        # Translate points by robot's XY position (B, N, 2)
        global_percep_points_2d = rotated_points_2d + curr_state.x[..., :2].unsqueeze(1)

        # Interpolate height at global 2D points (B, N)
        z_coords = interpolate_grid(self.env.terrain_cfg.z_grid, global_percep_points_2d, self.env.terrain_cfg.max_coord)

        # Reshape and make height relative to robot's Z coordinate
        hm = z_coords.reshape(B, 1, self.percep_shape[0], self.percep_shape[1]) - curr_state.x[..., 2].reshape(-1, 1, 1, 1)
        # Apply noise if specified
        if self.apply_noise:
            noise = torch.randn_like(hm) * self.noise_scale
            hm.add_(noise)
        hm.clamp_(self.interval[0], self.interval[1])
        # Normalize to interval if specified
        if self.normalize_to_interval:
            hm.div_(self.interval[1] - self.interval[0])
        return hm.to(self.env.out_dtype)

    def from_realistic_world(self, tensordict: "TensorDictBase") -> torch.Tensor:
        """
        Process heightmap from realistic world to match training observation format.
        The heightmap is expected to be in the same local coordinate frame as the robot (i.e., centered around the robot's position).
        Increasing row index -> decreasing X coordinate (backward from robot's perspective)
        Increasing column index -> decreasing Y coordinate (right from robot's perspective)
        The input tensordict is expected to contain the following keys:
        - "heightmap": The heightmap tensor from the realistic world (H, W)
        - "heightmap_extent": The extent of the heightmap in the format (x_max, y_max, x_min, y_min)
        Args:
            tensordict (TensorDictBase): The input tensordict containing the heightmap and its extent.
        Returns:
            torch.Tensor: Processed heightmap tensor of shape (1, H_percep, W_percep).
        """
        hm: torch.Tensor = tensordict["heightmap"].to(self.env.device)  # (H_src, W_src)
        extent: tuple[float, float, float, float] = tensordict["heightmap_extent"]  # (x_max, y_max, x_min, y_min)
        if isinstance(extent, torch.Tensor):
            extent = extent.cpu().squeeze().tolist()
        # check if the extent contains the percep_extent
        if (
            extent[0] < self.percep_extent[0]
            or extent[1] < self.percep_extent[1]
            or extent[2] > self.percep_extent[2]
            or extent[3] > self.percep_extent[3]
        ):
            raise ValueError("The provided extent does not fully contain the percep_extent.")
        # construct sampling grid
        x_target = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0], device=hm.device)
        y_target = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1], device=hm.device)
        px, py = torch.meshgrid(x_target, y_target, indexing="ij")
        grid_v = 2 * (px - extent[0]) / (extent[2] - extent[0]) - 1
        grid_u = 2 * (py - extent[1]) / (extent[3] - extent[1]) - 1
        grid = torch.stack((grid_u, grid_v), dim=-1)  # (H_p, W_p, 2)
        # Add batch dimensions
        grid = grid.unsqueeze(0)  # (1, H_p, W_p, 2)
        hm_input = hm.unsqueeze(0)  # (1, H_src, W_src)
        z_coords = torch.nn.functional.grid_sample(
            hm_input,
            grid,  # (1, H_percep, W_percep, 2)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        z_coords.clamp_(self.interval[0], self.interval[1])
        # Normalize to interval if specified
        if self.normalize_to_interval:
            z_coords.div_(self.interval[1] - self.interval[0])
        return z_coords.to(self.env.out_dtype)  # (1, H_percep, W_percep)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 1, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> HeightmapEncoder:
        return HeightmapEncoder(self.percep_shape, **self.encoder_opts)
