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
    "collector_opts",
    "is_unrecoverable_gpu_error",
    "is_wandb_transport_error",
    "reconcile_resumed_lrs",
]


def reconcile_resumed_lrs(optim, scheduler, cfg_lr: dict, log=None) -> list[str]:
    """After ``optim.load_state_dict`` / ``scheduler.load_state_dict`` on a resume, honour a
    learning rate that was CHANGED in the config, without discarding the schedule's progress.

    Each param group carries two rates after a restore: ``lr`` (the current, already decayed
    value) and ``initial_lr`` (what the schedule started from). The config's value is compared
    with ``initial_lr``, not with ``lr``:

    - equal: the config was not changed, so the restored ``lr`` is kept untouched. Comparing
      with ``lr`` instead (the previous behaviour) found a "difference" on every resume once
      the schedule had decayed at all, and reset the rate to its initial value — with a
      chainable scheduler such as ``LinearLR`` (each step multiplies the current rate) every
      relaunch then threw away the decay so far, so a run reloaded every period for terrain
      rotation would never decay.
    - different: the schedule restarts from the new base at the SAME progress — ``lr`` is
      rescaled by ``new / old`` and ``initial_lr`` / ``scheduler.base_lrs`` are set to the
      config value, so a deliberate LR change takes effect immediately and keeps decaying.

    ``cfg_lr`` maps param-group name -> configured lr (groups without a name use key None).
    Returns one human-readable line per changed group (also passed to ``log`` if given).
    """
    msgs = []
    for pg in optim.param_groups:
        want = cfg_lr.get(pg.get("name"))
        if want is None:
            continue
        old_base = pg.get("initial_lr", pg["lr"])
        if old_base == want:
            continue
        scale = want / old_base if old_base else 1.0
        msg = (f"Resume: config lr for {pg.get('name')} is {want:.3g}, checkpoint schedule started at "
               f"{old_base:.3g} — rescaling the current rate {pg['lr']:.3g} -> {pg['lr'] * scale:.3g}.")
        pg["lr"] = pg["lr"] * scale
        pg["initial_lr"] = want
        msgs.append(msg)
        if log is not None:
            log(msg)
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [pg.get("initial_lr", pg["lr"]) for pg in optim.param_groups]
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [pg["lr"] for pg in optim.param_groups]
    return msgs

# Number of depth columns the per-spot breakdown splits each env type into for the
# mid-training eval. The eval scripts read this from the terrain's layout; the trainers use a
# fixed value because the CSV is only a within-run diagnostic.
_TRAIN_EVAL_DEPTH_COLS = 10


def collector_opts(opts) -> dict:
    """``data_collector_opts`` for ``SyncDataCollector``, with ``no_cuda_sync`` defaulting to True.

    With ``no_cuda_sync=False`` (torchrl's default) ``SyncDataCollector.iterator()`` creates a
    high-priority side CUDA stream and keeps it current across every ``yield`` — so env.step,
    and with it every IsaacLab tensor op that builds the index/target tensors handed to the
    PhysX tensor API, runs on that stream. omni.physics.tensors orders its GPU work against
    the default stream only, so PhysX could read those tensors before the side stream had
    written them: the intermittent ``CUDA error: an illegal memory access`` reported at
    GpuArticulationView.cpp:749. On a single device the collector's explicit syncs it also
    skips are not needed. An explicit ``no_cuda_sync`` in the config still wins.
    """
    out = dict(opts or {})
    out.setdefault("no_cuda_sync", True)
    return out


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
        or is_wandb_transport_error(exc)
    )


def is_wandb_transport_error(exc: BaseException) -> bool:
    """True when W&B itself is gone: a CommError, or its backend process having died (raised
    as a plain ``Exception`` from any ``wandb.log``/``wandb.save``). The GPU is still healthy,
    so a trainer can save locally before exiting 75."""
    return "CommError" in type(exc).__name__ or "wandb backend process has shutdown" in str(exc)
