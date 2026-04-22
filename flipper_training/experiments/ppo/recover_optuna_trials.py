# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Recover cancelled Optuna trials: run eval on saved weights and submit results to DB."
)
parser.add_argument(
    "--trial_num", "-n", type=int, required=True,
    help="Optuna trial number to recover (e.g. 39).",
)
parser.add_argument(
    "--optuna_config", "-o", type=str, required=True,
    help="Path to optuna search-space YAML (e.g. /ws/configs/optuna_ftr.yaml).",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app  # noqa: F841


# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import datetime
import importlib
import sqlite3
import traceback
from pathlib import Path

import optuna
import torch
from omegaconf import OmegaConf
from optuna.storages import RDBStorage

import flipper_training  # registers OmegaConf resolvers
from flipper_training import ROOT
from flipper_training.experiments.ppo.train_ftr import FtrPPOConfig, FtrPPOTrainer
from flipper_training.utils.logutils import get_terminal_logger

import gymnasium

LOGGER = get_terminal_logger("recover_optuna")


# ============================================================
# BLOCK 3 — Helpers
# ============================================================

def find_best_checkpoint(weights_dir: Path) -> tuple[Path, Path] | None:
    """Return (policy_path, vecnorm_path) for the best available checkpoint.

    Prefers policy_final.pth; falls back to the latest policy_step_*.pth.
    """
    policy_final = weights_dir / "policy_final.pth"
    vecnorm_final = weights_dir / "vecnorm_final.pth"
    if policy_final.exists() and vecnorm_final.exists():
        return policy_final, vecnorm_final

    # Fall back to the highest-numbered policy_step checkpoint
    step_policies = sorted(
        weights_dir.glob("policy_step_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for policy_path in reversed(step_policies):
        step_num = policy_path.stem.split("_")[-1]
        vecnorm_path = weights_dir / f"vecnorm_step_{step_num}.pth"
        if vecnorm_path.exists():
            return policy_path, vecnorm_path

    return None


def resolve_container_path(container_path: str, ws_root: Path) -> Path:
    """Convert /ws/... container path to a host path using ws_root."""
    if container_path.startswith("/ws/"):
        return ws_root / container_path[4:]
    return Path(container_path)


def submit_result_to_db(db_path: str, trial_id: int, value: float) -> None:
    """Directly write a COMPLETE result into the Optuna SQLite DB.

    Transitions the trial from FAIL (or any state) → COMPLETE by:
      1. Removing any pre-existing trial_values rows (idempotent).
      2. Inserting the new objective value.
      3. Setting state = COMPLETE and datetime_complete.
    """
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("DELETE FROM trial_values WHERE trial_id = ?", (trial_id,))
        conn.execute(
            "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (?, 0, ?, 'FINITE')",
            (trial_id, value),
        )
        conn.execute(
            "UPDATE trials SET state = 'COMPLETE', datetime_complete = ? WHERE trial_id = ?",
            (datetime.datetime.utcnow().isoformat(sep=" "), trial_id),
        )
        conn.commit()
        LOGGER.info(f"DB updated: trial_id={trial_id} → COMPLETE, value={value:.4f}")
    finally:
        conn.close()


# ============================================================
# BLOCK 4 — Recovery logic
# ============================================================

def fix_lowercase_value_types(sqlite_path: str) -> None:
    """Fix 'finite' (lowercase) value_type rows that crash Optuna's SQLAlchemy enum.

    Must be called via direct sqlite3 BEFORE any RDBStorage / study.trials access,
    because SQLAlchemy raises LookupError on the malformed enum and aborts the read.
    """
    conn = sqlite3.connect(sqlite_path, timeout=60)
    try:
        n = conn.execute(
            "UPDATE trial_values SET value_type = 'FINITE' WHERE value_type = 'finite'"
        ).rowcount
        conn.commit()
        if n > 0:
            LOGGER.warning(f"Fixed {n} trial_values row(s) with lowercase 'finite' → 'FINITE'")
    finally:
        conn.close()


def recover_trial(trial_num: int, optuna_cfg_path: str, ws_root: Path) -> bool:
    """Load saved weights for *trial_num*, run evaluation, and submit results to DB.

    Returns True on success, False on any non-fatal error.
    """
    # ---- connect to study ----
    db_secret = OmegaConf.load(ROOT / "optuna_db.yaml")
    conn_str = db_secret["url"]
    # Resolve /ws/ prefix in the SQLite file path
    sqlite_path = conn_str.removeprefix("sqlite:///")
    if sqlite_path.startswith("/ws/"):
        sqlite_path = str(ws_root / sqlite_path[4:])

    # Fix any 'finite' (lowercase) rows BEFORE RDBStorage reads them — those rows
    # cause a LookupError in SQLAlchemy's enum processor and crash study.trials.
    if conn_str.startswith("sqlite:///"):
        fix_lowercase_value_types(sqlite_path)

    engine_kwargs = {"connect_args": {"timeout": 300}} if conn_str.startswith("sqlite:///") else {}
    storage = RDBStorage(conn_str, engine_kwargs=engine_kwargs)

    optuna_cfg_raw = OmegaConf.load(optuna_cfg_path)
    study_name = str(optuna_cfg_raw["study_name"])

    # Use fewer repeats than the training default (10 × 2400 steps ≈ 36 min).
    # break_when_all_done=True + auto_reset=True never exits early — every repeat
    # runs the full max_eval_steps.  3 repeats × 1200 steps ≈ 5 min total.
    eval_repeats = 10
    max_eval_steps_override = 5000  # one full episode timeout (episode_length_s=30 / dt=0.025)

    study = optuna.load_study(study_name=study_name, storage=storage)
    frozen = next((t for t in study.trials if t.number == trial_num), None)
    if frozen is None:
        LOGGER.error(f"Trial {trial_num} not found in study '{study_name}'")
        return False

    LOGGER.info(f"Trial {trial_num} — current state: {frozen.state.name}")

    # Skip trials that are already COMPLETE — don't overwrite valid results
    if frozen.state == optuna.trial.TrialState.COMPLETE:
        LOGGER.info(f"Trial {trial_num} is already COMPLETE (value={frozen.values}) — skipping.")
        return True

    LOGGER.info(f"Trial {trial_num} — params: {frozen.params}")

    # ---- locate trial log directory ----
    logpath_raw = frozen.user_attrs.get("logpath", "")
    if not logpath_raw:
        LOGGER.error(f"Trial {trial_num}: no 'logpath' user attribute in DB")
        return False
    logpath = resolve_container_path(logpath_raw, ws_root)
    if not logpath.exists():
        LOGGER.error(f"Trial {trial_num}: logpath {logpath} does not exist")
        return False

    # ---- find best available checkpoint ----
    weights_dir = logpath / "weights"
    checkpoint = find_best_checkpoint(weights_dir)
    if checkpoint is None:
        LOGGER.error(f"Trial {trial_num}: no usable checkpoint in {weights_dir}")
        return False
    policy_path, vecnorm_path = checkpoint
    LOGGER.info(f"Trial {trial_num}: using {policy_path.name} + {vecnorm_path.name}")

    # ---- load and patch trial config ----
    config_path = logpath / "config.yaml"
    if not config_path.exists():
        LOGGER.error(f"Trial {trial_num}: config.yaml not found in {logpath}")
        return False

    config = OmegaConf.load(config_path)
    # Disable W&B/TB (no logging needed for recovery), inject weight paths and eval repeats
    config = OmegaConf.merge(config, OmegaConf.create({
        "use_wandb": False,
        "use_tensorboard": False,
        "policy_weights_path": str(policy_path),
        "vecnorm_weights_path": str(vecnorm_path),
        "eval_repeats_after_training": eval_repeats,
        "max_eval_steps": max_eval_steps_override,
    }))

    trial_cfg = FtrPPOConfig(**config)
    task = trial_cfg.task

    # ---- recreate gymnasium env (mirrors objective() in optuna_train_ftr.py) ----
    spec = gymnasium.spec(task)
    _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
        _mod, _cls = _env_cfg_entry.rsplit(":", 1)
        EnvCfgClass = getattr(importlib.import_module(_mod), _cls)
    elif isinstance(_env_cfg_entry, type):
        EnvCfgClass = _env_cfg_entry
    else:
        from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg
        EnvCfgClass = CrossingEnvCfg

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = trial_cfg.num_robots
    env_cfg.terrain_name = trial_cfg.terrain
    env_cfg.sim.physx.gpu_heap_capacity = trial_cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = trial_cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = trial_cfg.physx_gpu_max_num_partitions
    env_cfg.sim.dt = trial_cfg.sim_dt
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = trial_cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = trial_cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = trial_cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = trial_cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = trial_cfg.robot_angular_damping
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = trial_cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = trial_cfg.solver_velocity_iterations
    env_cfg.sim.physx.min_position_iteration_count = trial_cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = trial_cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = trial_cfg.bounce_threshold_velocity
    for k, v in (trial_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(task, cfg=env_cfg)

    # ---- run evaluation ----
    # max_episode_length = episode_length_s / (sim_dt * decimation) = 30 / (0.005 * 5) = 1200
    # max_eval_steps     = 2 * max_episode_length = 2400  (stops early via break_when_all_done)
    # Each rollout runs all num_robots envs in parallel until every robot is done or timed out.
    # Estimated wall-clock: ~2–5 min eval + ~10 min Isaac Sim init = 15–20 min total.
    LOGGER.info(
        f"Trial {trial_num}: starting {eval_repeats} eval rollout(s) "
        f"with {trial_cfg.num_robots} parallel envs "
        f"(max_eval_steps={max_eval_steps_override}, ~{max_eval_steps_override * 0.09 / 60:.1f} min/rollout) ..."
    )

    try:
        trainer = FtrPPOTrainer(config, ftr_gym_env, optuna_trial=None)

        # Replicate _post_training_evaluation() with per-rollout progress logging
        avg: dict[str, float] = {}
        for rep in range(eval_repeats):
            LOGGER.info(f"  [{rep + 1}/{eval_repeats}] rollout starting ...")
            result = trainer._get_eval_rollout_results()
            sr  = result.get("eval/success_rate", 0.0)
            fr  = result.get("eval/failure_rate", 0.0)
            ptr = result.get("eval/pct_truncated", 0.0)
            LOGGER.info(
                f"  [{rep + 1}/{eval_repeats}] done — "
                f"success={sr:.3f}  failure={fr:.3f}  timeout={ptr:.3f}"
            )
            if not avg:
                avg = dict(result)
            else:
                for k, v in result.items():
                    avg[k] = avg.get(k, 0.0) + v

        for k in avg:
            avg[k] /= eval_repeats

        metrics: dict[str, float] = avg

    except Exception as e:
        traceback.print_exception(e)
        LOGGER.error(f"Trial {trial_num}: evaluation raised an exception: {e}")
        try:
            ftr_gym_env.close()
        except Exception:
            pass
        return False

    try:
        ftr_gym_env.close()
    except Exception:
        pass

    # ---- report and submit ----
    LOGGER.info(f"Trial {trial_num} — averaged results over {eval_repeats} rollout(s):")
    for k, v in sorted(metrics.items()):
        LOGGER.info(f"  {k}: {v:.4f}")

    success_rate = metrics.get("eval/success_rate", 0.0)
    submit_result_to_db(sqlite_path, frozen._trial_id, success_rate)
    LOGGER.info(f"Trial {trial_num} recovered — success_rate={success_rate:.4f}")
    return True


# ============================================================
# BLOCK 5 — Entry point
# ============================================================

if __name__ == "__main__":
    import os

    if not torch.cuda.is_available():
        print(
            "FATAL: torch.cuda.is_available() returned False.\n"
            "Isaac Sim failed to create a CUDA context — check .err for 'CUDA error 46'.",
            flush=True,
        )
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    # Inside the Apptainer container the workspace is mounted at /ws
    ws_root = Path("/ws") if Path("/ws").exists() else Path(ROOT).parent.parent.parent

    try:
        ok = recover_trial(
            trial_num=args.trial_num,
            optuna_cfg_path=args.optuna_config,
            ws_root=ws_root,
        )
    except Exception as _e:
        traceback.print_exception(_e)
        ok = False

    # os._exit bypasses Isaac Sim atexit handlers that frequently deadlock
    os._exit(0 if ok else 1)
