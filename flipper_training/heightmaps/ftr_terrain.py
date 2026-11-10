"""
FTR terrain patch generator — loads a FTR .map numpy heightmap file and samples
random per-robot patches using bilinear interpolation.

The .map files are plain numpy float32 arrays saved with np.save / np.load.
Each file represents one terrain at 0.05 m/cell resolution.

The generator is designed to be used as a drop-in replacement for the existing
procedural generators, fitting the BaseHeightmapGenerator interface.  Patches are
sampled once at startup; the same fixed patches are used for the entire training run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import BaseHeightmapGenerator


@dataclass(kw_only=True)
class FtrTerrainPatchGenerator(BaseHeightmapGenerator):
    """Samples per-robot terrain patches from a pre-computed FTR .map heightmap.

    Args:
        map_path: Path to the .map file (numpy float32 array, shape H×W).
        terrain_cell_size: Resolution of the .map file in metres (default 0.05).
        terrain_lower: World-space origin (x_min, y_min) of the .map array in metres.
            Used to determine which part of the map to sample. For cur_mixed: (-27, -27).
        z_offset: Height offset to subtract so that flat ground ≈ 0.
            For cur_mixed the USD is placed at z=0.5, so the .map values are ~0.5 on flat
            ground; setting z_offset=0.5 makes flat ground ≈ 0.
        suitable_height_threshold: Heights above this value (after z_offset) are marked as
            unsuitable for spawning (e.g., interior of tall obstacles).  null = all suitable.
    """

    map_path: str
    terrain_cell_size: float = 0.05
    terrain_lower: tuple[float, float] = (-27.0, -27.0)
    z_offset: float = 0.5
    suitable_height_threshold: float | None = 0.4

    # Cached map tensor, populated lazily on first call.
    _map_tensor: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def _load_map(self, device: torch.device) -> torch.Tensor:
        """Load .map file and return shape (1, 1, H, W) float32 on *device*."""
        if self._map_tensor is not None and self._map_tensor.device == device:
            return self._map_tensor
        raw = np.load(self.map_path, allow_pickle=True).astype(np.float32)
        t = torch.from_numpy(raw).to(device)  # (H, W)
        self._map_tensor = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return self._map_tensor

    def _generate_heightmap(
        self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample B random patches of shape (B, D, D) from the FTR terrain map.

        Args:
            x: Local x-coordinates, shape (B, D, D), range [-max_coord, max_coord].
            y: Local y-coordinates, shape (B, D, D), same range.
            max_coord: Half-size of the terrain patch in metres.
            rng: Reproducible RNG.

        Returns:
            z: Heights shape (B, D, D), zero-centred around flat ground.
            extras: {"suitable_mask": bool tensor shape (B, D, D)}.
        """
        device = x.device
        map_4d = self._load_map(device)  # (1, 1, H_map, W_map)
        H_map, W_map = map_4d.shape[2], map_4d.shape[3]
        B, D = x.shape[0], x.shape[1]

        # --- Terrain extent in world coordinates ---
        lx, ly = self.terrain_lower
        map_width_m  = W_map * self.terrain_cell_size   # x-direction
        map_height_m = H_map * self.terrain_cell_size   # y-direction

        # Margin: keep patch fully inside the map.
        margin = max_coord + self.terrain_cell_size
        cx_min = lx + margin
        cx_max = lx + map_width_m - margin
        cy_min = ly + margin
        cy_max = ly + map_height_m - margin

        if cx_min >= cx_max or cy_min >= cy_max:
            raise ValueError(
                f"FtrTerrainPatchGenerator: max_coord={max_coord} is too large for the "
                f"map ({map_width_m:.1f}×{map_height_m:.1f} m).  Reduce max_coord."
            )

        # Sample random patch centres for each robot.
        rand_cx = torch.rand(B, generator=rng, device=device) * (cx_max - cx_min) + cx_min  # (B,)
        rand_cy = torch.rand(B, generator=rng, device=device) * (cy_max - cy_min) + cy_min  # (B,)

        # Convert world coords (patch_x, patch_y) to normalised grid_sample coords [-1, 1].
        # grid_sample convention: x=col (W dim), y=row (H dim).
        # World x ↔ map col, world y ↔ map row.
        # normalised coord: u = 2*(world_x - lx)/map_width_m  - 1
        #                   v = 2*(world_y - ly)/map_height_m - 1

        # For each robot b, the query points are (rand_cx[b] + x[b], rand_cy[b] + y[b]).
        global_x = x + rand_cx.view(B, 1, 1)  # (B, D, D)  world x
        global_y = y + rand_cy.view(B, 1, 1)  # (B, D, D)  world y

        # Normalise to [-1, 1] for grid_sample.
        u = (2.0 * (global_x - lx) / map_width_m  - 1.0).clamp(-1.0, 1.0)  # (B, D, D)
        v = (2.0 * (global_y - ly) / map_height_m - 1.0).clamp(-1.0, 1.0)  # (B, D, D)

        # grid_sample expects grid of shape (B, H_out, W_out, 2) with (x, y) = (col, row).
        grid = torch.stack([u, v], dim=-1)  # (B, D, D, 2)

        # Expand map to batch: (1,1,H,W) → (B,1,H,W) via broadcasting inside grid_sample.
        map_batch = map_4d.expand(B, -1, -1, -1)  # (B, 1, H_map, W_map)
        z_sampled = F.grid_sample(map_batch, grid, mode="bilinear", align_corners=True, padding_mode="border")
        # z_sampled: (B, 1, D, D)
        z = z_sampled.squeeze(1) - self.z_offset  # (B, D, D), flat ground ≈ 0

        # Suitability mask: robots should not be spawned on tall obstacles or inside structures.
        if self.suitable_height_threshold is not None:
            suitable_mask = z < self.suitable_height_threshold
        else:
            suitable_mask = torch.ones_like(z, dtype=torch.bool)

        return z, {"suitable_mask": suitable_mask}
