"""Terrain asset lookup / export and the per-eval terrain manifest.

Terrains used for training are produced by src/FTR-Bench-terrain-gen from a
generator config, and installed into FTR-Benchmark's asset tree as::

    ftr_envs/assets/terrain/
        gen_config/<name>.yaml   the generator config the terrain was built from
        plot/<name>.svg          heightmap preview (tile grid + start/goal markers)
        birth/<name>.json        per-tile start/target points
        config/<name>.yaml       map extent (lower/upper/cell_size) + camera
        map/<name>.map           the raw heightmap (np.save'd float array)

The generator config is the authoritative description of a terrain's row/column
layout — which obstacle sits in which row, how many difficulty columns there
are, and the tile geometry that maps a world position back to (row, col). This
module reads it (see ``env_type_registry.py``, which derives its layouts from
here) and, at the start of an eval run, copies it plus the preview plot into the
eval output directory alongside the CSVs.

That copy is what makes ``notebooks/eval_analysis.ipynb`` self-contained: an eval
directory rsync'd off the cluster carries its own record of which terrain it ran
on and what that terrain looks like, so the notebook can label rows, size the
per-spot grid, and show the matching terrain preview without the asset tree —
and can refuse to merge evals that ran on *different* terrains.

Nothing here imports Isaac Sim, torch, or omni.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Name of the manifest written into an eval's --output_dir.
MANIFEST_NAME = "eval_terrain.json"

# Subdirectory of the eval output dir that mirrors the asset tree layout.
TERRAIN_SUBDIR = "terrain"


# ── Asset tree location ──────────────────────────────────────────────────────

def terrain_assets_dir() -> Path | None:
    """Locate ``ftr_envs/assets/terrain``.

    Honours ``$FTR_TERRAIN_DIR`` first (useful when the asset tree is mounted
    somewhere other than the installed package), then falls back to the
    installed ``ftr_envs`` package. Returns None if neither resolves.
    """
    env_dir = os.environ.get("FTR_TERRAIN_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    try:
        import ftr_envs  # noqa: PLC0415  (optional dependency — analysis code may not have it)

        # __file__ is None for a namespace package; anything unexpected here must
        # degrade to "no asset tree", never break the caller — get_terrain_layout()
        # runs on every train/eval startup.
        p = Path(ftr_envs.__file__).parent / "assets" / "terrain"
    except Exception:  # noqa: BLE001
        return None

    return p if p.is_dir() else None


def _asset_path(terrain: str, sub: str, suffix: str) -> Path | None:
    root = terrain_assets_dir()
    if root is None or not terrain:
        return None
    p = root / sub / f"{terrain}{suffix}"
    return p if p.is_file() else None


_GEN_CONFIG_CACHE: dict[str, dict[str, Any] | None] = {}


def load_terrain_gen_config(terrain: str | None) -> dict[str, Any] | None:
    """Return the parsed ``gen_config/<terrain>.yaml``, or None if there isn't one.

    Hand-authored (non-generated) terrains such as ``cur_mixed`` have no
    generator config; callers must handle None.
    """
    if not terrain:
        return None
    if terrain in _GEN_CONFIG_CACHE:
        return _GEN_CONFIG_CACHE[terrain]

    cfg: dict[str, Any] | None = None
    path = _asset_path(terrain, "gen_config", ".yaml")
    if path is not None:
        import yaml  # noqa: PLC0415

        try:
            loaded = yaml.safe_load(path.read_text())
            if isinstance(loaded, dict) and loaded.get("rows"):
                cfg = loaded
        except Exception:  # noqa: BLE001 — a malformed config must not break eval
            cfg = None

    _GEN_CONFIG_CACHE[terrain] = cfg
    return cfg


def gen_config_geometry(terrain: str | None) -> dict[str, Any] | None:
    """Flatten a terrain's generator config into the geometry fields plots need."""
    cfg = load_terrain_gen_config(terrain)
    if cfg is None:
        return None
    tile = cfg.get("tile", {}) or {}
    return {
        "row_types": [r["type"] for r in cfg["rows"]],
        "n_rows": len(cfg["rows"]),
        "repeats": int(cfg.get("repeats", 1)),
        "tile_width": float(tile.get("width", 5.0)),
        "tile_depth": float(tile.get("depth", 10.0 / 3.0)),
        "cell_size": float(cfg.get("cell_size", 0.05)),
        "border_width": float(cfg.get("border_width", 2.0)),
    }


# ── Preview rendering ────────────────────────────────────────────────────────

def _mask_sentinel(heightmap):
    """NaN out a map's uniform out-of-bounds sentinel value, if it has one.

    Terrains are padded with a height far below the real surface to mark
    off-course cells. Only treated as a sentinel when it sits well clear of
    everything else, so a genuinely low terrain isn't punched full of holes.
    """
    import numpy as np  # noqa: PLC0415

    lo = float(heightmap.min())
    rest = heightmap[heightmap > lo + 1e-6]
    if rest.size == 0:
        return heightmap
    spread = max(float(rest.max() - rest.min()), 1e-6)
    if float(rest.min()) - lo < 0.25 * spread:
        return heightmap
    return np.where(heightmap <= lo + 1e-6, np.nan, heightmap)


def _render_preview_png(terrain: str, dest: Path) -> bool:
    """Rasterise a terrain preview to *dest* from the raw heightmap.

    The generator already ships a vector preview (``plot/<name>.svg``), but
    matplotlib cannot read SVG, so an analysis notebook can't place it beside a
    heatmap without an external rasteriser. Re-rendering the same view as a PNG
    here — where the asset tree is guaranteed present — means the notebook only
    ever needs ``imread``. Mirrors ``ftr_terrain_gen.assembler.build_plot``.

    Returns True if the PNG was written.
    """
    map_path = _asset_path(terrain, "map", ".map")
    cfg_path = _asset_path(terrain, "config", ".yaml")
    if map_path is None or cfg_path is None:
        return False

    try:
        import numpy as np  # noqa: PLC0415
        import yaml  # noqa: PLC0415
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        with open(map_path, "rb") as f:
            heightmap = np.load(f, allow_pickle=True)
        map_cfg = (yaml.safe_load(cfg_path.read_text()) or {}).get("map", {})
        lower = list(map_cfg["lower"])[:2]
        upper = list(map_cfg["upper"])[:2]
        cell_size = float(map_cfg.get("cell_size", 0.05))

        geom = gen_config_geometry(terrain)

        # Drop the sentinel border ring: left in, its far-below-terrain height
        # stretches the colormap until the real steps all wash out to one shade.
        if geom is not None:
            b = round(geom["border_width"] / cell_size)
            if b > 0:
                heightmap = heightmap[b:-b, b:-b]
            half_x = geom["repeats"] * geom["tile_width"] / 2
            half_y = geom["n_rows"] * geom["tile_depth"] / 2
            extent = (-half_x, half_x, -half_y, half_y)
        else:
            # No generator config to say how wide the border is — blank the
            # sentinel cells instead so imshow autoscales over real terrain only.
            extent = (lower[0], upper[0], lower[1], upper[1])
            heightmap = _mask_sentinel(heightmap)

        width_m = extent[1] - extent[0]
        depth_m = extent[3] - extent[2]
        fig_w = 12.0
        fig = Figure(figsize=(fig_w, max(fig_w * depth_m / width_m, 1.0)), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(heightmap.T, origin="lower", cmap="terrain", interpolation="nearest", extent=extent)

        if geom is not None:
            for k in range(geom["repeats"] + 1):
                ax.axvline(extent[0] + k * geom["tile_width"], color="black", linewidth=1.5)
            for r in range(geom["n_rows"] + 1):
                ax.axhline(extent[2] + r * geom["tile_depth"], color="black", linewidth=1.5)

        birth_path = _asset_path(terrain, "birth", ".json")
        if birth_path is not None:
            birth = json.loads(birth_path.read_text())
            starts = np.array([e["start_point"][:2] for e in birth])
            targets = np.array([e["target_point"][:2] for e in birth])
            ax.scatter(starts[:, 0], starts[:, 1], marker="o", color="lime", edgecolors="black", s=40, zorder=5)
            ax.scatter(targets[:, 0], targets[:, 1], marker="*", color="red", edgecolors="black", s=90, zorder=5)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.axis("off")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format="png", bbox_inches="tight", pad_inches=0)
        return True
    except Exception:  # noqa: BLE001 — a missing preview must never abort an eval
        return False


# ── Export + manifest ────────────────────────────────────────────────────────

# (asset subdir, filename suffix, manifest key) — the .map itself is deliberately
# not copied: it is several MB and only the preview derived from it is needed.
_EXPORTED_ASSETS = [
    ("gen_config", ".yaml", "gen_config"),
    ("plot",       ".svg",  "plot_svg"),
    ("birth",      ".json", "birth"),
    ("config",     ".yaml", "map_config"),
]


def export_terrain_assets(output_dir: Path, terrain: str | None) -> dict[str, Any]:
    """Copy *terrain*'s generator config / preview / birth points into *output_dir*.

    Files land in ``<output_dir>/terrain/<subdir>/<name>.<ext>``, mirroring the
    asset tree. Returns the manifest entry for this terrain (paths relative to
    *output_dir*), which is empty of file keys when the asset tree isn't
    reachable — export is best-effort and never raises.
    """
    entry: dict[str, Any] = {"terrain": terrain or ""}
    geom = gen_config_geometry(terrain)
    if geom is not None:
        entry.update(geom)
    if not terrain:
        return entry

    dest_root = Path(output_dir) / TERRAIN_SUBDIR
    for sub, suffix, key in _EXPORTED_ASSETS:
        src = _asset_path(terrain, sub, suffix)
        if src is None:
            continue
        dest = dest_root / sub / f"{terrain}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(src, dest)
            entry[key] = str(dest.relative_to(output_dir))
        except OSError:
            continue

    png = dest_root / "plot" / f"{terrain}.png"
    if _render_preview_png(terrain, png):
        entry["plot_png"] = str(png.relative_to(output_dir))

    return entry


def write_terrain_manifest(
    output_dir: Path,
    eval_id: str,
    terrain: str | None,
    env_type_names: list[str],
    num_depth_cols: int,
    policy: str = "",
) -> None:
    """Record which terrain *eval_id* ran on, and export that terrain's assets.

    Merges into an existing ``eval_terrain.json`` so several eval runs can share
    one output directory (the notebook keys off ``eval_id``, exactly as the CSVs
    do). Best-effort: never raises, since losing the manifest must not lose the
    eval results written alongside it.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / MANIFEST_NAME

        manifest: dict[str, Any] = {"evals": {}, "terrains": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    manifest["evals"].update(loaded.get("evals", {}))
                    manifest["terrains"].update(loaded.get("terrains", {}))
            except (OSError, ValueError):
                pass

        manifest["evals"][eval_id] = {
            "terrain": terrain or "",
            "policy": policy,
            "env_type_names": list(env_type_names),
            "num_env_types": len(env_type_names),
            "num_depth_cols": int(num_depth_cols),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if terrain:
            manifest["terrains"][terrain] = export_terrain_assets(output_dir, terrain)

        path.write_text(json.dumps(manifest, indent=2))
    except Exception:  # noqa: BLE001
        pass
