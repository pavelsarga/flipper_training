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
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marv_rl_training.training.env_type_registry import (
    default_env_type_names,
    default_num_env_types,
    locate_env_type,
)


def load_env_type_names(terrain: str | None, path: str | None, num_env_types: int | None = None) -> list[str]:
    """Return a list of env-type names for *terrain*.

    Defaults come from the terrain's registered layout (env_type_registry.py);
    unregistered terrains fall back to generic ``env_NN`` names. If *path* is
    given, it overrides names by loading a YAML file containing either:
    - a plain list: ``["rails", "steps", ...]``
    - a mapping: ``{0: "rails", 1: "steps", ...}``
    """
    n = num_env_types if num_env_types is not None else default_num_env_types(terrain)
    names = default_env_type_names(terrain, n)
    if path is None:
        return names

    import yaml  # PyYAML is available in the isaaclab env

    with open(path) as f:
        data = yaml.safe_load(f)

    names = list(names)
    if isinstance(data, list):
        for i, name in enumerate(data):
            if i < n:
                names[i] = str(name)
    elif isinstance(data, dict):
        for k, name in data.items():
            idx = int(k)
            if idx < n:
                names[idx] = str(name)
    return names


def env_type_depth_col_from_target(
    terrain: str | None,
    target_x: float,
    target_y: float,
    num_env_types: int = 16,
    num_depth_cols: int = 10,
) -> tuple[int, int]:
    """Infer (env_type_idx, depth_col) from a robot's target position on *terrain*.

    Uses the terrain's registered layout (env_type_registry.py) when known.
    More reliable than robot_idx % N because the birth-entry cycling means
    env_type cannot be inferred from robot index alone after the first episode.
    """
    return locate_env_type(terrain, target_x, target_y, num_env_types, num_depth_cols)


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


def save_per_spot_csv(csv_path: Path, per_spot_rows: list[PerSpotRow]) -> None:
    """Append *per_spot_rows* to a single local CSV file (not sent to wandb)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _append_rows(csv_path, per_spot_rows, PerSpotRow)


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
    """_OBS_SLICES describes the 966-dim flat heightmap+state observation (PPO's
    MarvRLFlatObservation). Other PPO-trained modules (e.g. mitriakov's 8-D obs) use a
    much smaller hand-crafted vector that doesn't match these slices — skip whatever
    doesn't fit rather than reducing over an empty (0-width) slice, which raises on
    min()/max() (mirrors run_tracked_rollout's identical guard in this same file)."""
    obs_flat = obs.reshape(-1, obs.shape[-1])
    obs_dim = obs_flat.shape[-1]
    stats: dict[str, float] = {}
    for name, s, e in _OBS_SLICES:
        if e > obs_dim:
            continue
        vals = obs_flat[:, s:e].reshape(-1)
        stats[f"observations/{name}_mean"] = vals.mean().item()
        stats[f"observations/{name}_std"] = vals.std().item()
        stats[f"observations/{name}_min"] = vals.min().item()
        stats[f"observations/{name}_max"] = vals.max().item()
    return stats


# ── Shared tracked-rollout runner (per-env-type / per-spot bookkeeping) ───────
#
# Used both by eval_ftr.py's standalone --output_dir CSV export and by the
# periodic in-training eval of train_ftr.py / train_icmd3qn.py / train_d3qn.py.
# Requires ``ftr_torchrl_env.enable_per_env_tracking()`` to have been called
# beforehand so per-robot success/failure/explosion counts are collected.

def run_tracked_rollout(
    env,
    ftr_torchrl_env,
    ftr_gym_env,
    actor,
    max_steps: int,
    num_env_types: int,
    env_type_names: list[str],
    num_depth_cols: int = 10,
    repeat: int = 1,
    eval_id: str = "",
    policy_label: str = "",
    terrain: str = "",
) -> tuple[dict[str, float], list[EpisodeRecord]]:
    """Manual step loop that records per-robot episode data for env-type/spot aggregation.

    Performance notes vs a plain ``env.rollout(...)``:
    - Per-robot reward/step accumulators are GPU tensors (no Python list updates).
    - Obs stats and reward min/max/terminated/truncated fractions are computed online
      (running sum/sq/min/max) — no large rollout tensor is ever materialised.
    - .cpu() transfers only happen on steps where at least one robot is done, and
      only for the done-robot subset of position/target tensors.
    """
    from marv_rl_training.environment.ftr_env_adapter import OBS_KEY

    device = ftr_torchrl_env.device
    num_envs = ftr_torchrl_env.batch_size[0]
    unwrapped = ftr_gym_env.unwrapped

    # Per-robot accumulators — kept on GPU to avoid per-step Python overhead
    robot_rewards = torch.zeros(num_envs, device=device)
    robot_steps   = torch.zeros(num_envs, dtype=torch.long, device=device)
    robot_ep_idx  = [0] * num_envs  # small Python list; only updated on episode end

    # Online obs statistics — avoids storing all obs tensors
    obs_dim: int = ftr_torchrl_env.observations[0].dim
    _obs_sum    = torch.zeros(obs_dim, device=device)
    _obs_sq_sum = torch.zeros(obs_dim, device=device)
    _obs_min    = torch.full((obs_dim,), float("inf"),  device=device)
    _obs_max    = torch.full((obs_dim,), float("-inf"), device=device)
    _obs_count  = 0

    episode_records: list[EpisodeRecord] = []

    total_step_reward_gpu = torch.zeros(1, device=device)
    min_reward_gpu = torch.full((1,), float("inf"), device=device)
    max_reward_gpu = torch.full((1,), float("-inf"), device=device)
    terminated_count_gpu = torch.zeros(1, device=device)
    truncated_count_gpu  = torch.zeros(1, device=device)
    n_steps = 0

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        td = env.reset()
        for _ in range(max_steps):
            # ── Online obs stats update (no list accumulation) ──────────────
            obs  = td[OBS_KEY].detach()                    # [num_envs, obs_dim]
            flat = obs.reshape(-1, obs_dim)                # [num_envs, obs_dim]
            _obs_sum    += flat.sum(0)
            _obs_sq_sum += flat.pow(2).sum(0)
            _obs_min    = torch.minimum(_obs_min, flat.min(0).values)
            _obs_max    = torch.maximum(_obs_max, flat.max(0).values)
            _obs_count += flat.shape[0]

            # Snapshot positions before step: IsaacLab resets terminated robots
            # inside env.step(), so reading positions after step gives the new
            # start position rather than the terminal position.
            _pos_snap = unwrapped.positions.clone()
            _tgt_snap = unwrapped.target_positions.clone()

            td = actor(td)
            td = env.step(td)

            rewards    = td["next", "reward"].squeeze(-1)     # [num_envs], on device
            dones      = td["next", "done"].squeeze(-1)       # [num_envs], on device
            terminated = td["next", "terminated"].squeeze(-1)
            truncated  = td["next", "truncated"].squeeze(-1)

            robot_rewards += rewards
            robot_steps   += 1

            total_step_reward_gpu += rewards.mean()  # accumulate on GPU, no sync per step
            min_reward_gpu = torch.minimum(min_reward_gpu, rewards.min())
            max_reward_gpu = torch.maximum(max_reward_gpu, rewards.max())
            terminated_count_gpu += terminated.float().sum()
            truncated_count_gpu  += truncated.float().sum()
            n_steps += 1

            # ── Record completed episodes ───────────────────────────────────
            if dones.any():
                per_env     = ftr_torchrl_env.pop_per_env_termination()
                done_mask   = dones.cpu()
                done_idx    = done_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                positions   = _pos_snap.cpu()
                target_pos  = _tgt_snap.cpu()
                rew_cpu     = robot_rewards.cpu()
                steps_cpu   = robot_steps.cpu()

                for i in done_idx:
                    counts = per_env.get(i, {})
                    if counts.get("successes", 0) > 0:
                        outcome = "success"
                    elif counts.get("explosions", 0) > 0:
                        outcome = "explosion"
                    elif counts.get("failures", 0) > 0:
                        outcome = "failure"
                    else:
                        outcome = "truncated"

                    dist = float((target_pos[i, :2] - positions[i, :2]).norm().item())
                    env_type_idx, depth_col = env_type_depth_col_from_target(
                        terrain, float(target_pos[i, 0]), float(target_pos[i, 1]),
                        num_env_types, num_depth_cols,
                    )
                    episode_records.append(EpisodeRecord(
                        eval_id=eval_id,
                        policy=policy_label,
                        terrain=terrain,
                        repeat=repeat,
                        robot_idx=i,
                        env_type_idx=env_type_idx,
                        env_type_name=env_type_names[env_type_idx],
                        depth_col=depth_col,
                        episode_idx=robot_ep_idx[i],
                        outcome=outcome,
                        steps=int(steps_cpu[i].item()),
                        cumulative_reward=float(rew_cpu[i].item()),
                        dist_to_goal_final=dist,
                    ))
                    robot_rewards[i] = 0.0
                    robot_steps[i]   = 0
                    robot_ep_idx[i] += 1

            if dones.all():
                break
            td = td["next"]

    # ── Assemble obs stats from online accumulators ─────────────────────────
    obs_stats: dict[str, float] = {}
    if _obs_count > 0 and _obs_sum is not None:
        mean_gpu = _obs_sum / _obs_count
        var  = (_obs_sq_sum / _obs_count - mean_gpu ** 2).clamp(min=0)
        mean = mean_gpu.cpu()
        std  = var.sqrt().cpu()
        mn   = _obs_min.cpu()
        mx   = _obs_max.cpu()
        # _OBS_SLICES describes the 966-dim flat heightmap+state observation (PPO's
        # MarvRLFlatObservation). D3QN variants use a much smaller hand-crafted obs
        # (17-18 dims) that doesn't match these slices — skip whatever doesn't fit
        # rather than reducing over an empty (0-width) slice.
        for name, s, e in _OBS_SLICES:
            if e > obs_dim:
                continue
            obs_stats[f"observations/{name}_mean"] = mean[s:e].mean().item()
            obs_stats[f"observations/{name}_std"]  = std[s:e].mean().item()
            obs_stats[f"observations/{name}_min"]  = mn[s:e].min().item()
            obs_stats[f"observations/{name}_max"]  = mx[s:e].max().item()

    total_cells = max(n_steps * num_envs, 1)
    results: dict[str, float] = {
        "eval/mean_step_reward": total_step_reward_gpu.item() / max(n_steps, 1),
        "eval/max_step_reward":  max_reward_gpu.item(),
        "eval/min_step_reward":  min_reward_gpu.item(),
        "eval/pct_terminated":   terminated_count_gpu.item() / total_cells,
        "eval/pct_truncated":    truncated_count_gpu.item() / total_cells,
        "eval/rollout_steps":    float(n_steps),
    }
    results.update(obs_stats)
    term_info = ftr_torchrl_env.pop_termination_info()
    results.update({("eval/explosion_rate" if k == "explosions/rate" else "eval/" + k.split("/", 1)[-1]): v for k, v in term_info.items()})
    results.update(ftr_torchrl_env.pop_reward_info())
    return results, episode_records
