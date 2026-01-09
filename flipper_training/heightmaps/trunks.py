from dataclasses import dataclass

import torch
from enum import IntEnum

from . import BaseHeightmapGenerator


class TrunkSide(IntEnum):
    TRUNK = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class TrunkHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with a large obstacle resembling the trunk of a fallen tree.
    """

    min_trunk_height: float = 0.3
    max_trunk_height: float = 0.4
    max_dist_from_origin: float = 0.5  # meters from origin to the middle of the trunk
    trunk_deadzone_width: float = 0.5  # width of the deadzone around the trunk where robots cannot start or end
    max_trunk_width: float = 1.0  # maximum width of the trunk
    min_trunk_width: float = 0.6  # minimum width of the trunk
    exp: float | int = 4  # exponent for the heightmap function

    def __post_init__(self):
        if self.exp < 0:
            raise ValueError("Exponent must be non-negative.")
        if self.min_trunk_height < 0:
            raise ValueError("Minimum trunk height must be non-negative.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B = x.shape[0]
        z = torch.zeros_like(x)
        left_right_mask = torch.full_like(x, TrunkSide.TRUNK, dtype=torch.uint8)
        for i in range(B):
            angle = torch.rand((1,), generator=rng) * 2 * torch.pi
            offset = torch.rand((1,), generator=rng) * self.max_dist_from_origin
            normal = torch.stack([torch.cos(angle), torch.sin(angle)])
            height = torch.rand((1,), generator=rng) * (self.max_trunk_height - self.min_trunk_height) + self.min_trunk_height
            width = torch.rand((1,), generator=rng) * (self.max_trunk_width - self.min_trunk_width) + self.min_trunk_width
            dist = x[i] * normal[0] + y[i] * normal[1] - offset  # signed distance from the line
            mask = dist.abs() > (self.trunk_deadzone_width + width / 2)
            z_i = (1 - 1 / ((width / 2) ** self.exp) * (dist.abs() ** self.exp)) * height
            z_i.clamp_min_(0)
            left_mask = mask & (dist < 0)
            right_mask = mask & (dist > 0)
            left_right_mask[i, left_mask] = TrunkSide.LEFT
            left_right_mask[i, right_mask] = TrunkSide.RIGHT
            z[i] = z_i
        return z, {"trunk_sides": left_right_mask, "suitable_mask": left_right_mask != TrunkSide.TRUNK}


@dataclass
class FixedTrunkHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with a large obstacle resembling the trunk of a fallen tree.
    """

    normal_angle: float = 0.0  # angle of the trunk normal in radians
    trunk_height: float = 0.45
    dist_from_origin: float = 0.0  # meters from origin to the middle of the trunk
    trunk_deadzone_width: float = 0.5  # width of the deadzone around the trunk where robots cannot start or end
    trunk_width: float = 1.0  # maximum width of the trunk
    exp: float | int = 6  # exponent for the heightmap function

    def __post_init__(self):
        if self.exp < 0:
            raise ValueError("Exponent must be non-negative.")
        if self.trunk_height < 0:
            raise ValueError("Minimum trunk height must be non-negative.")
        if self.trunk_width <= 0:
            raise ValueError("Trunk width must be positive.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        left_right_mask = torch.full_like(x, TrunkSide.TRUNK, dtype=torch.uint8)
        dist = x * torch.cos(torch.tensor(self.normal_angle)) + y * torch.sin(torch.tensor(self.normal_angle)) - self.dist_from_origin
        mask = dist.abs() > (self.trunk_deadzone_width + self.trunk_width / 2)
        left_mask = mask & (dist < 0)
        right_mask = mask & (dist > 0)
        left_right_mask[left_mask] = TrunkSide.LEFT
        left_right_mask[right_mask] = TrunkSide.RIGHT
        z = (1 - 1 / ((self.trunk_width / 2) ** self.exp) * (dist.abs() ** self.exp)) * self.trunk_height
        z.clamp_min_(0)
        return z, {"trunk_sides": left_right_mask}
