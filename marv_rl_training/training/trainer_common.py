"""Pieces of the mid-training evaluation and crash handling shared by the trainers.

``FtrPPOTrainer``, ``FtrD3QNTrainer`` and ``FtrSACTrainer`` are separate classes on purpose —
on-policy actor-critic, off-policy value-based and off-policy actor-critic have genuinely
different training loops — but the parts around the loop are the same in all three. Those
live here so a fix lands once.
"""

from pathlib import Path

from marv_rl_training.training.eval_data import aggregate_per_env, aggregate_per_spot, save_per_spot_csv

__all__ = [
    "attach_per_env_eval_rows",
    "average_eval_repeats",
    "is_unrecoverable_gpu_error",
]

# Number of depth columns the per-spot breakdown splits each env type into for the
# mid-training eval. The eval scripts read this from the terrain's layout; the trainers use a
# fixed value because the CSV is only a within-run diagnostic.
_TRAIN_EVAL_DEPTH_COLS = 10


def attach_per_env_eval_rows(results: dict, episode_records, *, env_type_names, run_name: str,
                             terrain: str, per_spot_csv: Path) -> None:
    """Add ``eval_per_env/<name>_success_rate`` to ``results`` and write the per-spot CSV.

    The per-(env-type, depth-column) "spot" breakdown goes to a local CSV only, never to
    W&B — it is one row per spot per checkpoint, far too wide to log.
    """
    if not episode_records:
        return
    per_env_rows = aggregate_per_env(
        episode_records=episode_records, env_type_names=env_type_names,
        eval_id="train", policy=run_name, terrain=terrain, repeat=1, obs_stats=results,
    )
    for row in per_env_rows:
        results[f"eval_per_env/{row.env_type_name}_success_rate"] = row.success_rate

    per_spot_rows = aggregate_per_spot(
        episode_records=episode_records, env_type_names=env_type_names,
        num_depth_cols=_TRAIN_EVAL_DEPTH_COLS, eval_id="train", policy=run_name,
        terrain=terrain, repeat=1,
    )
    save_per_spot_csv(per_spot_csv, per_spot_rows)


def average_eval_repeats(get_results, repeats: int) -> dict[str, float]:
    """Run the eval ``repeats`` times, average the metrics, and print them."""
    avg = get_results()
    for _ in range(repeats - 1):
        for k, v in get_results().items():
            avg[k] += v
    for k in avg:
        avg[k] /= repeats
    print("\nFinal evaluation results:")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    return avg


def is_unrecoverable_gpu_error(exc: BaseException) -> bool:
    """True when the CUDA context is dead, or W&B's transport is.

    Saving weights or running the atexit / Isaac Sim cleanup handlers deadlocks once the CUDA
    context is gone, so the caller must ``os._exit(75)`` instead — freeing the SLURM slot for
    the respawn rather than hanging on it for hours. W&B CommError (usually a TLS cert
    problem) gets the same treatment because it is equally transient and equally fatal to
    this attempt.
    """
    return (
        "CUDA error" in str(exc)
        or "CUDA out of memory" in str(exc)
        or "CommError" in type(exc).__name__
    )
