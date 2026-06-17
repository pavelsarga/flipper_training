"""Shared data structures and CSV utilities for per-env-type evaluation tracking.

Imported by eval_ftr.py and eval_ftr_rand.py (no Isaac Sim / omni imports here).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# ── Default env-type names for cur_mixed terrain ──────────────────────────────
# 16 types: indices 0-14 are obstacle rows in cur_mixed (Y ≈ 23.3 … -21.5),
# index 15 is the flat patch (X≈28, Y=-5).
NUM_ENV_TYPES_CUR_MIXED: int = 16
NUM_DEPTH_COLS_CUR_MIXED: int = 10
DEFAULT_ENV_NAMES_CUR_MIXED: list[str] = [
    "raised_platform",    # 0  Y≈23.3
    "lowered_platform",   # 1  Y≈20.1
    "rails",              # 2  Y≈16.9
    "diag_rail",          # 3  Y≈13.7
    "asymmetric_platform",# 4  Y≈10.5
    "diag_mound",         # 5  Y≈ 7.3
    "trenches",           # 6  Y≈ 4.1
    "cobblestone",        # 7  Y≈ 0.9
    "steps_lowered",      # 8  Y≈-2.3
    "steps_raised",       # 9  Y≈-5.5
    "large_blocks",       # 10 Y≈-8.7
    "medium_blocks",      # 11 Y≈-11.9
    "diag_blocks",        # 12 Y≈-15.1
    "diag_platform",      # 13 Y≈-18.3
    "large_steps",        # 14 Y≈-21.5
    "flat_patch",         # 15 flat patch (X≈28)
]


def load_env_type_names(path: str | None, num_env_types: int) -> list[str]:
    """Return a list of *num_env_types* env-type names.

    If *path* is given, load from a YAML file that contains either:
    - a plain list: ``["rails", "steps", ...]``
    - a mapping: ``{0: "rails", 1: "steps", ...}``

    Missing indices fall back to the hardcoded cur_mixed names, then ``env_NN``.
    """
    defaults = [
        DEFAULT_ENV_NAMES_CUR_MIXED[i] if i < len(DEFAULT_ENV_NAMES_CUR_MIXED) else f"env_{i:02d}"
        for i in range(num_env_types)
    ]
    if path is None:
        return defaults

    import yaml  # PyYAML is available in the isaaclab env

    with open(path) as f:
        data = yaml.safe_load(f)

    names = list(defaults)
    if isinstance(data, list):
        for i, name in enumerate(data):
            if i < num_env_types:
                names[i] = str(name)
    elif isinstance(data, dict):
        for k, name in data.items():
            idx = int(k)
            if idx < num_env_types:
                names[idx] = str(name)
    return names


# ── cur_mixed birth-position geometry ─────────────────────────────────────────
# Terrain rows 0-14: target Y = 23.3 - row * 3.2  (target X varies by depth col)
# Flat patch (row 15): target X ≈ 34.0, target Y = -5 + depth_col
# Depth cols 0-9: target X = 19.3 - col * 4.8  (terrain rows only)
_CUR_MIXED_Y0   = 23.3   # Y of env_type 0
_CUR_MIXED_DY   = 3.2    # Y step between env types
_CUR_MIXED_X0   = 19.3   # target X of depth col 0
_CUR_MIXED_DX   = 4.8    # X step between depth cols
_CUR_MIXED_FP_X = 34.0   # flat patch target X
_CUR_MIXED_FP_Y0 = -5.0  # flat patch target Y at depth col 0


def env_type_depth_col_from_target(
    target_x: float,
    target_y: float,
    num_env_types: int = NUM_ENV_TYPES_CUR_MIXED,
    num_depth_cols: int = NUM_DEPTH_COLS_CUR_MIXED,
) -> tuple[int, int]:
    """Infer (env_type_idx, depth_col) from the actual target position.

    Uses the known cur_mixed terrain geometry. More reliable than robot_idx % N
    because the birth-entry cycling means env_type cannot be inferred from robot
    index alone after the first episode.
    """
    if abs(target_x - _CUR_MIXED_FP_X) < 2.0:
        # Flat patch: depth col encoded in target Y
        depth_col = max(0, min(num_depth_cols - 1, round(target_y - _CUR_MIXED_FP_Y0)))
        return num_env_types - 1, depth_col
    env_type = max(0, min(num_env_types - 2, round((_CUR_MIXED_Y0 - target_y) / _CUR_MIXED_DY)))
    depth_col = max(0, min(num_depth_cols - 1, round((_CUR_MIXED_X0 - target_x) / _CUR_MIXED_DX)))
    return env_type, depth_col


def make_eval_id(prefix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{ts}" if prefix else ts


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    eval_id: str
    policy: str          # path or "random"
    terrain: str
    repeat: int
    robot_idx: int
    env_type_idx: int
    env_type_name: str
    depth_col: int       # (robot_idx // num_env_types) % num_depth_cols
    episode_idx: int     # per-robot episode counter (resets each repeat)
    outcome: str         # "success" | "failure" | "explosion" | "truncated"
    steps: int
    cumulative_reward: float
    dist_to_goal_final: float


@dataclass
class SummaryRow:
    eval_id: str
    policy: str
    terrain: str
    num_envs: int
    num_env_types: int
    repeat: int
    timestamp: str
    success_rate: float
    failure_rate: float
    explosion_rate: float
    mean_step_reward: float
    shock_mean: float
    shock_p90: float
    shock_p95: float
    shock_p99: float


@dataclass
class PerEnvRow:
    eval_id: str
    policy: str
    terrain: str
    repeat: int
    env_type_idx: int
    env_type_name: str
    episodes_total: int
    success_rate: float
    failure_rate: float
    explosion_rate: float
    mean_cumulative_reward: float
    mean_dist_to_goal_final: float
    flipper_fl_mean: float
    flipper_fr_mean: float
    flipper_rl_mean: float
    flipper_rr_mean: float


@dataclass
class PerSpotRow:
    eval_id: str
    policy: str
    terrain: str
    repeat: int
    env_type_idx: int
    env_type_name: str
    depth_col: int
    episodes_total: int
    success_rate: float
    failure_rate: float
    explosion_rate: float
    mean_cumulative_reward: float
    mean_dist_to_goal_final: float


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _field_names(dc_class) -> list[str]:
    return [f.name for f in fields(dc_class)]


def _append_rows(csv_path: Path, rows: list[Any], dc_class) -> None:
    """Append *rows* (dataclass instances) to *csv_path*, writing header if new."""
    header = _field_names(dc_class)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in header})


def save_eval_csvs(
    output_dir: Path,
    summary_rows: list[SummaryRow],
    per_env_rows: list[PerEnvRow],
    per_spot_rows: list[PerSpotRow],
    episode_records: list[EpisodeRecord],
) -> None:
    """Write / append the four CSV files into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _append_rows(output_dir / "eval_summary.csv",   summary_rows,    SummaryRow)
    _append_rows(output_dir / "eval_per_env.csv",   per_env_rows,    PerEnvRow)
    _append_rows(output_dir / "eval_per_spot.csv",  per_spot_rows,   PerSpotRow)
    _append_rows(output_dir / "eval_episodes.csv",  episode_records, EpisodeRecord)


# ── Per-env aggregation helper ────────────────────────────────────────────────

def aggregate_per_env(
    episode_records: list[EpisodeRecord],
    env_type_names: list[str],
    eval_id: str,
    policy: str,
    terrain: str,
    repeat: int,
    obs_stats: dict[str, float],
) -> list[PerEnvRow]:
    """Group *episode_records* by env_type_idx and produce one PerEnvRow each."""
    from collections import defaultdict

    buckets: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for rec in episode_records:
        if rec.repeat == repeat:
            buckets[rec.env_type_idx].append(rec)

    rows: list[PerEnvRow] = []
    for idx, name in enumerate(env_type_names):
        eps = buckets.get(idx, [])
        total = len(eps)
        if total == 0:
            rows.append(PerEnvRow(
                eval_id=eval_id, policy=policy, terrain=terrain, repeat=repeat,
                env_type_idx=idx, env_type_name=name,
                episodes_total=0,
                success_rate=float("nan"), failure_rate=float("nan"),
                explosion_rate=float("nan"),
                mean_cumulative_reward=float("nan"),
                mean_dist_to_goal_final=float("nan"),
                flipper_fl_mean=float("nan"), flipper_fr_mean=float("nan"),
                flipper_rl_mean=float("nan"), flipper_rr_mean=float("nan"),
            ))
            continue
        successes   = sum(1 for e in eps if e.outcome == "success")
        failures    = sum(1 for e in eps if e.outcome == "failure")
        explosions  = sum(1 for e in eps if e.outcome == "explosion")
        rows.append(PerEnvRow(
            eval_id=eval_id, policy=policy, terrain=terrain, repeat=repeat,
            env_type_idx=idx, env_type_name=name,
            episodes_total=total,
            success_rate=successes / total,
            failure_rate=failures / total,
            explosion_rate=explosions / total,
            mean_cumulative_reward=sum(e.cumulative_reward for e in eps) / total,
            mean_dist_to_goal_final=sum(e.dist_to_goal_final for e in eps) / total,
            flipper_fl_mean=obs_stats.get("observations/flipper_fl_mean", float("nan")),
            flipper_fr_mean=obs_stats.get("observations/flipper_fr_mean", float("nan")),
            flipper_rl_mean=obs_stats.get("observations/flipper_rl_mean", float("nan")),
            flipper_rr_mean=obs_stats.get("observations/flipper_rr_mean", float("nan")),
        ))
    return rows


# ── Per-spot aggregation helper ───────────────────────────────────────────────

def aggregate_per_spot(
    episode_records: list[EpisodeRecord],
    env_type_names: list[str],
    num_depth_cols: int,
    eval_id: str,
    policy: str,
    terrain: str,
    repeat: int,
) -> list[PerSpotRow]:
    """Group *episode_records* by (env_type_idx, depth_col) → one PerSpotRow each."""
    from collections import defaultdict

    buckets: dict[tuple[int, int], list[EpisodeRecord]] = defaultdict(list)
    for rec in episode_records:
        if rec.repeat == repeat:
            buckets[(rec.env_type_idx, rec.depth_col)].append(rec)

    rows: list[PerSpotRow] = []
    for env_idx, name in enumerate(env_type_names):
        for col in range(num_depth_cols):
            eps = buckets.get((env_idx, col), [])
            total = len(eps)
            if total == 0:
                rows.append(PerSpotRow(
                    eval_id=eval_id, policy=policy, terrain=terrain, repeat=repeat,
                    env_type_idx=env_idx, env_type_name=name, depth_col=col,
                    episodes_total=0,
                    success_rate=float("nan"), failure_rate=float("nan"),
                    explosion_rate=float("nan"),
                    mean_cumulative_reward=float("nan"),
                    mean_dist_to_goal_final=float("nan"),
                ))
                continue
            successes  = sum(1 for e in eps if e.outcome == "success")
            failures   = sum(1 for e in eps if e.outcome == "failure")
            explosions = sum(1 for e in eps if e.outcome == "explosion")
            rows.append(PerSpotRow(
                eval_id=eval_id, policy=policy, terrain=terrain, repeat=repeat,
                env_type_idx=env_idx, env_type_name=name, depth_col=col,
                episodes_total=total,
                success_rate=successes / total,
                failure_rate=failures / total,
                explosion_rate=explosions / total,
                mean_cumulative_reward=sum(e.cumulative_reward for e in eps) / total,
                mean_dist_to_goal_final=sum(e.dist_to_goal_final for e in eps) / total,
            ))
    return rows


# ── Observation stats helpers (shared between eval_ftr.py and eval_ftr_rand.py) ──

# 16 observation groups: name → (start_idx, end_idx) in the 966-dim flat obs vector
_OBS_SLICES = [
    ("heightmap",   0,   945),
    ("roll",        945, 946),
    ("pitch",       946, 947),
    ("lin_vel_x",   947, 948),
    ("lin_vel_y",   948, 949),
    ("lin_vel_z",   949, 950),
    ("ang_vel_x",   950, 951),
    ("ang_vel_y",   951, 952),
    ("ang_vel_z",   952, 953),
    ("flipper_fl",  953, 954),
    ("flipper_fr",  954, 955),
    ("flipper_rl",  955, 956),
    ("flipper_rr",  956, 957),
    ("goal_x",      957, 958),
    ("goal_y",      958, 959),
    ("goal_z",      959, 960),
]


def _compute_obs_stats(obs: torch.Tensor) -> dict[str, float]:
    obs_flat = obs.reshape(-1, obs.shape[-1])
    stats: dict[str, float] = {}
    for name, s, e in _OBS_SLICES:
        vals = obs_flat[:, s:e].reshape(-1)
        stats[f"observations/{name}_mean"] = vals.mean().item()
        stats[f"observations/{name}_std"] = vals.std().item()
        stats[f"observations/{name}_min"] = vals.min().item()
        stats[f"observations/{name}_max"] = vals.max().item()
    return stats
