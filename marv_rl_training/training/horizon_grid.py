"""The receding-horizon grid: one array task = one grid point, deterministically.

Why not Optuna's GridSampler. Two properties of it break a study whose trials run for a day
and crash a few times each (the transient PhysX fault costs the diffusion runs several
respawns per job):

  * In Optuna 4.3 `GridSampler._get_unvisited_grid_ids` counts every *finished* trial as a
    visited grid point, and FAIL is a finished state -- so a grid point whose trial crashed
    is never sampled again, and the grid silently ends with holes.
  * Re-attaching a respawned task to its own point through `study.enqueue_trial` is racy:
    the WAITING trial goes into the shared DB and any concurrent array task can pop it, so
    task 3 can end up training task 5's point in task 3's log directory, where the resume
    machinery then loads the wrong checkpoint.

So the mapping lives here instead: the search-space YAML lists the axes, `grid_points`
expands their product in a fixed order, and `point_for_task(task_id)` is what SLURM array
task N trains -- forced onto its trial with `PartialFixedSampler`, which never touches the
shared DB to decide parameters. A respawned task recomputes the same point; a task whose
respawn loop gives up leaves a FAILED trial that is fixed by resubmitting that one array
index. Optuna still records everything (values, intermediate evals, pruning, the dashboard);
it just does not choose.

Also importable without Isaac, so the sbatch can ask "what is execution_horizon for task N"
before the trainer starts -- slurm/lib/respawn_common.sh needs T_a to size the frame budget
and pin total_iters on a respawn, and in this study T_a is a grid axis, not a config value.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_axes(optuna_config: str | Path) -> dict[str, list[Any]]:
    cfg = yaml.safe_load(Path(optuna_config).read_text())
    keys, types, values = cfg["optuna_keys"], cfg["optuna_types"], cfg["optuna_values"]
    if not (len(keys) == len(types) == len(values)):
        raise ValueError("optuna_keys, optuna_types and optuna_values must have the same length")
    bad = [k for k, t in zip(keys, types) if t not in ("categorical", "bool")]
    if bad:
        raise ValueError(f"a grid needs categorical/bool axes only; {bad} are not")
    return {k: (list(v) if t == "categorical" else [True, False]) for k, t, v in zip(keys, types, values)}


def feasible(point: dict[str, Any]) -> bool:
    """The two constraints the receding-horizon scaffold imposes, when both axes are present.

    prediction_horizon >= execution_horizon: ActionChunkEnv executes the first T_a steps of a
    T_p-step chunk, so a chunk shorter than what is executed cannot exist.
    prediction_horizon even: ConditionalUnet1D halves the horizon once per level below the
    first; with the [64,128] head that is one halving, so T_p must divide by 2 (T_p=4..32 all
    do -- the check is here so a future odd axis value fails loudly at grid time, not one
    GPU-day into the study).
    """
    tp, ta = point.get("prediction_horizon"), point.get("execution_horizon")
    if tp is not None and ta is not None and tp < ta:
        return False
    if tp is not None and tp % 2 != 0:
        return False
    return True


def grid_points(optuna_config: str | Path) -> list[dict[str, Any]]:
    """Product of the axes, minus infeasible points, in a fixed order (first axis slowest)."""
    axes = load_axes(optuna_config)
    keys = list(axes)
    pts = [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]
    return [pt for pt in pts if feasible(pt)]


def point_for_task(optuna_config: str | Path, task_id: int) -> dict[str, Any]:
    pts = grid_points(optuna_config)
    if not 0 <= task_id < len(pts):
        raise IndexError(f"task {task_id} outside the {len(pts)}-point grid (use --array=0-{len(pts) - 1})")
    return pts[task_id]


def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("optuna_config")
    p.add_argument("task_id", nargs="?", type=int, help="omit to list every grid point")
    p.add_argument("--key", help="print only this parameter of the task's point")
    a = p.parse_args(argv)
    if a.task_id is None:
        for i, pt in enumerate(grid_points(a.optuna_config)):
            print(i, json.dumps(pt))
        return 0
    pt = point_for_task(a.optuna_config, a.task_id)
    print(pt[a.key] if a.key else json.dumps(pt))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
