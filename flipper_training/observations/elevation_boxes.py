"""Azayev & Zimmermann (2022) deployed terrain features: robust elevation-box statistics.

The author's DEPLOYED classifier input (``augmented_robot_trackers``,
``src/control/marv_flipper_controller.py:169-206``) is NOT the paper's Eq.-10 /
Fig.-5 per-flipper feature vectors (computed but never consumed — see
``state_machine_policy.py``'s module docstring and AUTHOR_CODE_FINDINGS.md §2.2):
it is ``pitch`` + TWO pooled full-robot-width boxes, each reduced to 4 robust
statistics by ``src/perception/marv_feature_processor.py``'s ``get_pc_feat``:

* ``avg_height`` — MEDIAN z of the points inside the box (median, not mean),
* ``min_bnd``    — median of the 10 LOWEST points (robust floor height),
* ``max_bnd``    — median of the 10 HIGHEST points (robust obstacle-top height),
* ``intensity``  — point count / ``n_max_intensity`` (=100), clamped to 1.

Box bounds (``src/perception/configs/marv_feature_processor_config.yaml``,
``[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]`` meters in the ZERO-ROLL-PITCH base-link
frame — origin at base_link, X = yaw heading, Z = gravity-vertical):

* ``front_low = [0.35, 0.7, -0.35, 0.35, -0.3, 0.4]`` — the volume the front
  flippers can act on next (MARV: pivots x=+-0.256, reach 0.3815 -> tips ~0.64),
* ``rear_low  = [-0.4,  0.0, -0.35, 0.35, -0.2, 0.3]`` — the ground under/behind
  the rear half of the chassis.

This observation reproduces those statistics from THIS stack's mapped elevation
representation instead of his RDS traversability cloud. Structural equivalence
(verified against his vendored RDS configs, ``launch/rds/map.yaml``): his map is
a LAYERED 2.5D grid at ``map_resolution: 0.05`` m with 1.5 m vertical layers —
the boxes' z-windows (<=0.7 m tall) live inside a single layer, so within the
feature volume his input is functionally a 5 cm heightmap, same as ours. His
"points" are therefore ~cells: ``n_max_intensity = 100`` ~= the 98 cells of the
0.35 x 0.7 m box footprint at 5 cm — i.e. his intensity IS (approximately) the
box coverage fraction, which is exactly what we compute (fraction of box cells
whose height falls inside the z-window). A cell participates in a box iff its
(robot-relative) height is inside the box's z-window; an empty box yields
``(0, 0, 0, 0)``, matching his ``len(pc) == 0`` branch.

Output layout: ``concat_over_boxes([avg_height, min_bnd, max_bnd, intensity])``
in the order of ``boxes`` (default: front_low, rear_low -> 8-D). Pitch is NOT
duplicated here — the config pairs this observation with ``LocalStateVector``,
which already carries roll/pitch (the gate reads the shared encoding of both,
reproducing the author's 9-D pitch+boxes information content).

Training path samples the engine's ground-truth ``z_grid`` on a fixed 5 cm
local-frame cell grid per box (yaw-only rotation — the [L]-style frame used by
``pan_terrain.sample_terrain_points_relative``, which this reuses); deployment
(``from_realistic_world``) resamples the deploy node's robot-local heightmap at
the same cell centers, so both paths feed identical statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.policies import MLP
from .pan_terrain import sample_terrain_points_relative
from . import Observation, ObservationEncoder

__all__ = ["ElevationBoxFeatures", "ElevationBoxFeaturesEncoder", "AUTHOR_BOXES", "robust_box_stats"]

# the author's deployed boxes, verbatim from marv_feature_processor_config.yaml
AUTHOR_BOXES: dict[str, tuple[float, float, float, float, float, float]] = {
    "front_low": (0.35, 0.7, -0.35, 0.35, -0.3, 0.4),
    "rear_low": (-0.4, 0.0, -0.35, 0.35, -0.2, 0.3),
}


def _masked_median(vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Median over the masked entries of each row of ``vals`` (B, N); 0 where mask empty.

    Matches ``np.median`` (mean of the two middle order statistics for even counts).
    """
    big = torch.where(mask, vals, torch.full_like(vals, torch.inf))
    srt, _ = torch.sort(big, dim=-1)
    k = mask.sum(dim=-1)  # (B,)
    lo = ((k - 1).clamp_min(0) // 2)
    hi = (k // 2).clamp_min(0)
    lo_v = srt.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    hi_v = srt.gather(-1, torch.minimum(hi, (k - 1).clamp_min(0)).unsqueeze(-1)).squeeze(-1)
    med = 0.5 * (lo_v + hi_v)
    return torch.where(k > 0, med, torch.zeros_like(med))


def robust_box_stats(z_rel: torch.Tensor, z_lo: float, z_hi: float, m_extreme: int = 10) -> torch.Tensor:
    """``get_pc_feat`` on heightmap cells: (B, N) robot-relative cell heights of one box
    footprint -> (B, 4) ``[median, median-of-m-lowest, median-of-m-highest, coverage]``.

    A cell is "in the box" iff its height lies in the box z-window. ``m_extreme=10``
    mirrors the author's 10-point robust min/max medians; coverage = in-window cells /
    all footprint cells (see module docstring for why this matches his intensity scale).
    Empty box -> all zeros (his ``len(pc) == 0`` branch).
    """
    mask = (z_rel > z_lo) & (z_rel < z_hi) & torch.isfinite(z_rel)
    avg = _masked_median(z_rel, mask)
    # m lowest / highest in-window cells per row
    k = mask.sum(dim=-1, keepdim=True)  # (B, 1)
    m = torch.minimum(k, torch.full_like(k, m_extreme))  # (B, 1)
    idx = torch.arange(z_rel.shape[-1], device=z_rel.device).view(1, -1)
    asc, _ = torch.sort(torch.where(mask, z_rel, torch.full_like(z_rel, torch.inf)), dim=-1)
    dsc, _ = torch.sort(torch.where(mask, z_rel, torch.full_like(z_rel, -torch.inf)), dim=-1, descending=True)
    low_mask = idx < m
    min_bnd = _masked_median(asc, low_mask & torch.isfinite(asc))
    max_bnd = _masked_median(dsc, low_mask & torch.isfinite(dsc))
    coverage = mask.float().sum(dim=-1) / z_rel.shape[-1]
    return torch.stack([avg, min_bnd, max_bnd, coverage.clamp(max=1.0)], dim=-1)


class ElevationBoxFeaturesEncoder(ObservationEncoder):
    """Small MLP encoder (same pattern as the other flat-vector observations)."""

    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.mlp = MLP(**mlp_kwargs | {"in_dim": input_dim, "out_dim": output_dim, "activate_last_layer": True})

    def forward(self, x):
        return self.mlp(x)


@dataclass
class ElevationBoxFeatures(Observation):
    """Author-deployed pooled elevation-box statistics (see module docstring).

    Args:
        boxes: name -> ``(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)`` in the zero-roll-pitch
            base-link frame. Default: the author's two deployed boxes, verbatim.
        cell_size: sampling-grid resolution (m). Default 0.05 = his ``map_resolution``.
        m_extreme: robust min/max median depth. Default 10 = his hardcoded 10.
    """

    supports_vecnorm = True

    boxes: dict[str, tuple[float, float, float, float, float, float]] = field(
        default_factory=lambda: dict(AUTHOR_BOXES))
    cell_size: float = 0.05
    m_extreme: int = 10

    def __post_init__(self):
        if self.apply_noise:
            if not isinstance(self.noise_scale, (float, torch.Tensor)):
                raise ValueError("Noise scale must be specified if apply_noise is True and must be a float or tensor.")
            if isinstance(self.noise_scale, float):
                self.noise_scale = torch.tensor([self.noise_scale], dtype=self.env.out_dtype, device=self.env.device)
            if self.noise_scale.shape[0] not in (1, self.dim):
                raise ValueError(f"Noise scale tensor must have shape (1,) or ({self.dim},) but got {self.noise_scale.shape}.")
        # fixed local-frame cell-center grids per box (built once)
        self._box_grids = {}   # name -> (local_x (N,), local_y (N,), z_lo, z_hi)
        for name, (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) in self.boxes.items():
            nx = max(1, round((x_hi - x_lo) / self.cell_size))
            ny = max(1, round((y_hi - y_lo) / self.cell_size))
            cx = torch.linspace(x_lo + self.cell_size / 2, x_hi - self.cell_size / 2, nx)
            cy = torch.linspace(y_lo + self.cell_size / 2, y_hi - self.cell_size / 2, ny)
            gx, gy = torch.meshgrid(cx, cy, indexing="ij")
            self._box_grids[name] = (
                gx.reshape(-1).to(self.env.device),
                gy.reshape(-1).to(self.env.device),
                z_lo, z_hi,
            )

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        feats = []
        for name, (lx, ly, z_lo, z_hi) in self._box_grids.items():
            z_rel = sample_terrain_points_relative(self.env, curr_state, lx, ly)  # (B, N)
            feats.append(robust_box_stats(z_rel, z_lo, z_hi, self.m_extreme))
        obs = torch.cat(feats, dim=-1).to(self.env.out_dtype)
        if self.apply_noise:
            obs = obs + torch.randn_like(obs) * self.noise_scale.view(1, -1)
        return obs

    def from_realistic_world(self, tensordict) -> torch.Tensor:
        """Deployment path: same statistics from the deploy node's robot-local heightmap
        (yaw-aligned, robot-relative heights — the zero-roll-pitch base-link frame the
        author's boxes are defined in). Conventions identical to
        ``pan_terrain.PanTerrainState.from_realistic_world``."""
        hm: torch.Tensor = tensordict["heightmap"].to(self.env.device)
        while hm.ndim > 2:
            hm = hm.squeeze(0)
        extent = tensordict["heightmap_extent"]
        if isinstance(extent, torch.Tensor):
            extent = extent.cpu().squeeze().tolist()
        feats = []
        for name, (lx, ly, z_lo, z_hi) in self._box_grids.items():
            if extent[0] < lx.max() or extent[1] < ly.max() or extent[2] > lx.min() or extent[3] > ly.min():
                raise ValueError(f"Real-world heightmap extent {extent} does not cover box '{name}'.")
            grid_v = 2 * (lx - extent[0]) / (extent[2] - extent[0]) - 1
            grid_u = 2 * (ly - extent[1]) / (extent[3] - extent[1]) - 1
            grid = torch.stack((grid_u, grid_v), dim=-1).view(1, 1, -1, 2)
            z = torch.nn.functional.grid_sample(
                hm.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).view(1, -1)  # (1, N) robot-relative heights at the box cell centers
            feats.append(robust_box_stats(z, z_lo, z_hi, self.m_extreme))
        return torch.cat(feats, dim=-1).to(self.env.out_dtype)

    @property
    def dim(self) -> int:
        return 4 * len(self.boxes)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> ElevationBoxFeaturesEncoder:
        return ElevationBoxFeaturesEncoder(input_dim=self.dim, **(self.encoder_opts or {}))
