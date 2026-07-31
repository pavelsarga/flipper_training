"""Per-terrain env-type (row) layouts for per-env-type / per-spot eval breakdowns.

Each curriculum terrain used for training is a grid of tiles: rows are distinct
obstacle types ("env types"), columns ("depth cols") grade a row's difficulty
from easiest (col 0) to hardest (col num_depth_cols-1). This module maps a
robot's target (world x, y) back to (env_type_idx, depth_col) and gives the
env-type name list for a terrain, so eval code doesn't need to hardcode one
terrain's geometry.

Terrains built by src/FTR-Bench-terrain-gen ship the generator config they were
built from as ftr_envs/assets/terrain/gen_config/<name>.yaml, and their layout is
read straight out of it — row order/names from `rows`, grid geometry from `tile`
and `repeats`, fed into CourseGrid's placement formula (ftr_terrain_gen/grid.py):
    y_center(row) = (row - (n_rows-1)/2) * tile_depth
    x_center(col) = -(repeats*tile_width)/2 + tile_width/2 + col*tile_width
So regenerating a terrain with a different row list needs no change here; there
is deliberately no hardcoded copy of a generated terrain's rows to drift out of
sync with the config that produced it.

"cur_mixed" is a legacy, hand-authored terrain (not produced by that
generator — see ftr_envs/assets/terrain/config/cur_mixed.yaml, which composes
cur_mixed.usd with a separately-placed flat_patch.usd) with empirically
measured row/flat-patch geometry that doesn't fit the generator's uniform
grid formula, so it stays in TERRAIN_ENV_TYPES below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from marv_rl_training.training.terrain_assets import gen_config_geometry


class TerrainLayout(Protocol):
    env_type_names: list[str]
    num_depth_cols: int

    def locate(self, target_x: float, target_y: float) -> tuple[int, int]:
        """Map a robot's target (world x, y) to (env_type_idx, depth_col)."""
        ...


@dataclass(frozen=True)
class GridTerrainLayout:
    """Uniform row/column grid, as laid out by ftr_terrain_gen.grid.CourseGrid."""

    env_type_names: list[str]
    tile_width: float   # X (repeat/difficulty) direction
    tile_depth: float   # Y (row/env-type) direction
    repeats: int        # number of depth columns

    @property
    def num_depth_cols(self) -> int:
        return self.repeats

    def locate(self, target_x: float, target_y: float) -> tuple[int, int]:
        n_rows = len(self.env_type_names)
        row = round(target_y / self.tile_depth + (n_rows - 1) / 2)
        row = max(0, min(n_rows - 1, row))
        x0 = -(self.repeats * self.tile_width) / 2 + self.tile_width / 2
        col = round((target_x - x0) / self.tile_width)
        col = max(0, min(self.repeats - 1, col))
        return row, col


@dataclass(frozen=True)
class CurMixedLayout:
    """Legacy cur_mixed geometry — empirically measured, not generator-derived.

    Rows 0-14 are cur_mixed.usd's obstacle rows (Y ≈ 23.3 … -21.5); row 15 is a
    separate flat_patch.usd placed adjacent to cur_mixed's +X edge (X≈28-34),
    hence the special-cased flat-patch branch below instead of a uniform grid.

    Note both axes count DOWN (env type rises as Y falls, depth col as X falls).
    That is not arbitrary: config/cur_mixed.yaml sets xformOp:orient to
    [0, 0, 0, 1.0], which terrain.py applies as Gf.Quatd(w=0, xyz=(0,0,1)) — a 180
    degree rotation about Z. These constants were measured against the terrain as
    actually loaded, so they already absorb that rotation and report the obstacle
    the robot is physically on. cur_mixed's .map is consistent with the same
    (rotated) frame, so its heightmap observations line up too. Don't "fix" that
    orient without re-measuring these numbers.
    """

    env_type_names: list[str] = field(default_factory=lambda: [
        "raised_platform",     # 0  Y≈23.3
        "lowered_platform",    # 1  Y≈20.1
        "rails",                # 2  Y≈16.9
        "diag_rail",            # 3  Y≈13.7
        "asymmetric_platform",  # 4  Y≈10.5
        "diag_mound",           # 5  Y≈ 7.3
        "trenches",              # 6  Y≈ 4.1
        "cobblestone",           # 7  Y≈ 0.9
        "steps_lowered",         # 8  Y≈-2.3
        "steps_raised",          # 9  Y≈-5.5
        "large_blocks",          # 10 Y≈-8.7
        "medium_blocks",         # 11 Y≈-11.9
        "diag_blocks",           # 12 Y≈-15.1
        "diag_platform",         # 13 Y≈-18.3
        "large_steps",           # 14 Y≈-21.5
        "flat_patch",            # 15 flat patch (X≈28)
    ])
    num_depth_cols: int = 10

    _Y0: float = 23.3     # Y of env_type 0
    _DY: float = 3.2      # Y step between env types
    _X0: float = 19.3     # target X of depth col 0
    _DX: float = 4.8      # X step between depth cols
    _FP_X: float = 34.0   # flat patch target X
    _FP_Y0: float = -5.0  # flat patch target Y at depth col 0

    def locate(self, target_x: float, target_y: float) -> tuple[int, int]:
        n = len(self.env_type_names)
        if abs(target_x - self._FP_X) < 2.0:
            depth_col = max(0, min(self.num_depth_cols - 1, round(target_y - self._FP_Y0)))
            return n - 1, depth_col
        env_type = max(0, min(n - 2, round((self._Y0 - target_y) / self._DY)))
        depth_col = max(0, min(self.num_depth_cols - 1, round((self._X0 - target_x) / self._DX)))
        return env_type, depth_col


# ── Registry ────────────────────────────────────────────────────────────────

# Only hand-authored terrains need an entry here; generated ones are derived
# from their shipped gen_config (see _layout_from_gen_config).
TERRAIN_ENV_TYPES: dict[str, TerrainLayout] = {
    "cur_mixed": CurMixedLayout(),
}


def unique_row_names(row_types: list[str]) -> list[str]:
    """Disambiguate repeated obstacle types into distinct env-type names.

    A course may use the same obstacle in several rows at different difficulty
    settings (e.g. two `cobblestones` rows with different grid_n); those are
    separate env types and need separate names: the second occurrence becomes
    ``cobblestones_2``, the third ``cobblestones_3``, and so on.
    """
    counts: dict[str, int] = {}
    names: list[str] = []
    for t in row_types:
        counts[t] = counts.get(t, 0) + 1
        names.append(t if counts[t] == 1 else f"{t}_{counts[t]}")
    return names


_GEN_LAYOUT_CACHE: dict[str, GridTerrainLayout | None] = {}


def _layout_from_gen_config(terrain: str) -> GridTerrainLayout | None:
    """Build a layout from the terrain generator's own config, if it ships one."""
    if terrain in _GEN_LAYOUT_CACHE:
        return _GEN_LAYOUT_CACHE[terrain]

    geom = gen_config_geometry(terrain)
    layout = None if geom is None else GridTerrainLayout(
        env_type_names=unique_row_names(geom["row_types"]),
        tile_width=geom["tile_width"],
        tile_depth=geom["tile_depth"],
        repeats=geom["repeats"],
    )
    _GEN_LAYOUT_CACHE[terrain] = layout
    return layout


def get_terrain_layout(terrain: str | None) -> TerrainLayout | None:
    """Return the layout for *terrain*, or None if it isn't a known curriculum terrain.

    Generated terrains resolve from their gen_config first, so a regenerated
    course is picked up automatically; TERRAIN_ENV_TYPES covers the hand-authored
    ones (and acts as a fallback when the asset tree isn't reachable).
    """
    if terrain is None:
        return None
    return _layout_from_gen_config(terrain) or TERRAIN_ENV_TYPES.get(terrain)


def default_env_type_names(terrain: str | None, num_env_types: int | None = None) -> list[str]:
    """Default env-type names for *terrain*, padded/truncated to *num_env_types* if given."""
    layout = get_terrain_layout(terrain)
    names = list(layout.env_type_names) if layout is not None else []
    n = num_env_types if num_env_types is not None else len(names) or 16
    return [names[i] if i < len(names) else f"env_{i:02d}" for i in range(n)]


def default_num_env_types(terrain: str | None) -> int:
    layout = get_terrain_layout(terrain)
    return len(layout.env_type_names) if layout is not None else 16


def default_num_depth_cols(terrain: str | None) -> int:
    layout = get_terrain_layout(terrain)
    return layout.num_depth_cols if layout is not None else 10


def locate_env_type(
    terrain: str | None,
    target_x: float,
    target_y: float,
    num_env_types: int = 16,
    num_depth_cols: int = 10,
) -> tuple[int, int]:
    """Map a robot's target (world x, y) to (env_type_idx, depth_col) for *terrain*.

    Falls back to the legacy cur_mixed formula (bucketed to *num_env_types*/
    *num_depth_cols*) for terrains not in the registry — an approximation,
    since only cur_mixed/custom_mixed/pan_symmetric are curriculum-row terrains.
    """
    layout = get_terrain_layout(terrain)
    if layout is not None:
        return layout.locate(target_x, target_y)
    fallback = CurMixedLayout(
        env_type_names=default_env_type_names(None, num_env_types),
        num_depth_cols=num_depth_cols,
    )
    return fallback.locate(target_x, target_y)
