# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Optuna study for the receding-horizon (Phase 1) policy — one trial per process."
)
parser.add_argument("--train_config", "-t", type=str, required=True, help="Base diffusion train config YAML.")
parser.add_argument("--optuna_config", "-o", type=str, required=True, help="Search-space YAML (see configs/optuna/).")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in the train config.")
parser.add_argument(
    "--task_id", type=int, default=None,
    help="Grid point to train (sampler: grid). Defaults to SLURM_ARRAY_TASK_ID; the sbatch passes it "
         "explicitly because it hides the array variables from this process (see slurm/optuna_diffusion.sbatch).",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import os
import traceback
from dataclasses import dataclass, field
from typing import Any

import optuna
from omegaconf import OmegaConf
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback

import marv_rl_training  # registers OmegaConf resolvers
from marv_rl_training import ROOT
from marv_rl_training.training.env_setup import build_ftr_gym_env, import_ftr_tasks, require_cuda
from marv_rl_training.training.horizon_grid import grid_points, point_for_task
from marv_rl_training.training.train_diffusion import FtrDiffusionConfig, FtrDiffusionTrainer
from marv_rl_training.utils.logutils import RunLogger, get_terminal_logger

TERM_LOGGER = get_terminal_logger("optuna_diffusion")


# ============================================================
# BLOCK 3 — Config and search-space helpers
# ============================================================

@dataclass
class OptunaDiffusionConfig:
    """Mirror of optuna_train_ftr.OptunaFtrConfig plus the fields this runner needs.

    sampler:      "grid" — the explicit array-task -> grid-point mapping of horizon_grid.py,
                  forced onto the trial with PartialFixedSampler (see that module for why not
                  GridSampler). "tpe" — optuna_train_ftr.py's TPE settings, for a future
                  continuous search on this trainer.
    pruning:      MedianPruner on the per-eval success rate the trainer reports; off keeps every
                  grid point at full budget so the comparison is clean.
    storage_url:  overrides optuna_db.yaml — a scratch SQLite file for a smoke test, so a dry
                  run cannot write junk into the real study DB.
    """

    study_name: str
    directions: list[str]
    metrics_to_optimize: list[str]
    num_trials: int
    gpu: int
    optuna_keys: list[str]
    optuna_types: list[str]
    optuna_values: list
    train_config_overrides: dict[str, Any] = field(default_factory=dict)
    frozen_params: dict[str, Any] | None = None
    slurm_params: dict[str, Any] | None = None
    metric_weights: list[float] | None = None
    sampler: str = "grid"
    pruning: bool = False
    pruner_n_startup_trials: int = 4
    pruner_n_warmup_steps: int = 8
    pruner_n_min_trials: int = 3
    storage_url: str | None = None


def define_search_space(trial: optuna.Trial, keys: list, types: list, values: list) -> dict:
    params = {}
    for key, typ, val in zip(keys, types, values):
        if typ == "float":
            params[key] = trial.suggest_float(key, val[0], val[1])
        elif typ == "log_float":
            params[key] = trial.suggest_float(key, val[0], val[1], log=True)
        elif typ == "int":
            params[key] = trial.suggest_int(key, val[0], val[1])
        elif typ == "categorical":
            params[key] = trial.suggest_categorical(key, val)
        elif typ == "bool":
            params[key] = trial.suggest_categorical(key, [True, False])
        else:
            raise ValueError(f"Unknown optuna type '{typ}' for key '{key}'.")
    return params


class FailedTrialException(Exception):
    pass


def _mark_trial_failed(trial: optuna.Trial, why: str) -> None:
    """Mark the trial FAIL before a hard exit. Optuna API first, then direct SQLite: concurrent
    array tasks often hold the SQLite write-lock and the API call raises, which would leave
    the trial RUNNING forever."""
    from optuna.trial import TrialState as _TS

    try:
        trial._storage.set_trial_state_values(trial._trial_id, _TS.FAIL, values=None)
        TERM_LOGGER.info(f"Trial {trial.number} marked as FAIL in Optuna DB ({why}).")
        return
    except Exception as e:  # noqa: BLE001
        TERM_LOGGER.warning(f"Optuna API FAIL mark failed: {e} — trying direct SQLite.")
    try:
        import datetime as _dt
        import sqlite3 as _sq3

        url = _storage_url()
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            if path.startswith("/ws/"):
                ws_root = "/ws" if os.path.isdir("/ws") else str(ROOT).rsplit("/src/", 1)[0]
                path = ws_root + path[3:]
            with _sq3.connect(path, timeout=30) as conn:
                conn.execute(
                    "UPDATE trials SET state='FAIL', datetime_complete=? WHERE trial_id=? AND state='RUNNING'",
                    (_dt.datetime.utcnow().isoformat(sep=" "), trial._trial_id),
                )
            TERM_LOGGER.info(f"Trial {trial.number} marked as FAIL via direct SQLite ({why}).")
    except Exception as e:  # noqa: BLE001
        TERM_LOGGER.warning(f"Direct SQLite FAIL mark also failed: {e}")


_STORAGE_URL: str | None = None


def _storage_url() -> str:
    return _STORAGE_URL or ""


# ============================================================
# BLOCK 4 — Objective (one trial = create env + train + close env)
# ============================================================

def objective(trial: optuna.Trial, base_config, optuna_cfg: OptunaDiffusionConfig) -> tuple[float, ...]:
    params = define_search_space(trial, optuna_cfg.optuna_keys, optuna_cfg.optuna_types, optuna_cfg.optuna_values)
    TERM_LOGGER.info(f"Trial {trial.number} — parameters:")
    for k, v in params.items():
        TERM_LOGGER.info(f"  {k} = {v}")
    dotlist = [f"{k}={v}" for k, v in params.items()]
    updated_config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))
    trial_cfg = FtrDiffusionConfig(**updated_config)

    # Same env construction as train_diffusion.py, per trial: the grid's execution_horizon and
    # history_len only touch the trainer, but prediction_horizon changes the action spec the
    # chunked env exposes, so nothing about the env can be shared across trials.
    ftr_gym_env = build_ftr_gym_env(trial_cfg, physx_autotune="full")
    try:
        trainer = FtrDiffusionTrainer(updated_config, ftr_gym_env, optuna_trial=trial)
        metrics = trainer.train()
    except optuna.TrialPruned:
        ftr_gym_env.close()
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exception(e)
        if "CUDA error" in str(e) or "CUDA out of memory" in str(e):
            # The CUDA context is dead: env.close() and Isaac Sim's atexit handlers deadlock.
            # Mark FAIL, then exit 75 so slurm/optuna_diffusion.sbatch respawns this task; it
            # comes back with the same --task_id, i.e. the same grid point, and the trainer
            # resumes from the checkpoints in this job's attempt_N directories.
            _mark_trial_failed(trial, "CUDA crash")
            os._exit(75)
        ftr_gym_env.close()
        raise FailedTrialException(f"Trial {trial.number} failed: {e}") from e

    ftr_gym_env.close()
    if metrics is None:
        raise FailedTrialException(f"Trial {trial.number} returned None (training was interrupted).")

    result = []
    for metric in optuna_cfg.metrics_to_optimize:
        if metric not in metrics:
            TERM_LOGGER.warning(f"Metric '{metric}' missing from trial results (available: {list(metrics)}). Using 0.0.")
            result.append(0.0)
        else:
            result.append(metrics[metric])

    if optuna_cfg.metric_weights is not None:
        score = sum(w * v for w, v in zip(optuna_cfg.metric_weights, result))
        for name, val in zip(optuna_cfg.metrics_to_optimize, result):
            trial.set_user_attr(name, val)
        trial.set_user_attr("score", score)
        TERM_LOGGER.info(f"Trial {trial.number} done — score={score:.4f} components={dict(zip(optuna_cfg.metrics_to_optimize, result))}")
        return (score,)
    TERM_LOGGER.info(f"Trial {trial.number} done — {dict(zip(optuna_cfg.metrics_to_optimize, result))}")
    return tuple(result)


# ============================================================
# BLOCK 5 — Study runner
# ============================================================

def _resolve_storage(optuna_cfg: OptunaDiffusionConfig) -> RDBStorage:
    global _STORAGE_URL
    if optuna_cfg.storage_url:
        conn_str = str(optuna_cfg.storage_url)
    else:
        db_secret_path = ROOT / "optuna_db.yaml"
        if not db_secret_path.exists():
            raise FileNotFoundError(f"optuna_db.yaml not found at {db_secret_path}")
        db_secret = OmegaConf.load(db_secret_path)
        if "url" in db_secret:
            conn_str = str(db_secret["url"])
        else:
            sslmode = db_secret.get("sslmode", "require")
            conn_str = (
                f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
                f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode={sslmode}"
            )
    _STORAGE_URL = conn_str
    engine_kwargs = {}
    if conn_str.startswith("sqlite:///"):
        import pathlib
        import sqlite3 as _sqlite3

        sqlite_path = conn_str[len("sqlite:///"):]
        pathlib.Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        # Parallel array tasks race on schema init and the file lock; 5 min is generous.
        engine_kwargs = {"connect_args": {"timeout": 300}}
    storage = RDBStorage(conn_str, engine_kwargs=engine_kwargs)   # creates the schema on a fresh DB
    if conn_str.startswith("sqlite:///"):
        # Rows written as lowercase 'finite' by an old recovery script make SQLAlchemy's enum
        # processor raise. Must run AFTER the storage exists: on a fresh file (a smoke test)
        # the table is not there yet, and the UPDATE itself is what failed.
        with _sqlite3.connect(sqlite_path, timeout=60) as conn:
            n = conn.execute("UPDATE trial_values SET value_type = 'FINITE' WHERE value_type = 'finite'").rowcount
            if n > 0:
                TERM_LOGGER.warning(f"Fixed {n} trial_values row(s) with lowercase 'finite' -> 'FINITE'")
    return storage


def _make_sampler(optuna_cfg: OptunaDiffusionConfig, task_id: int | None, optuna_config_path: str):
    if optuna_cfg.sampler == "grid":
        if task_id is None:
            raise ValueError("sampler: grid needs --task_id (or SLURM_ARRAY_TASK_ID) to pick the grid point")
        point = point_for_task(optuna_config_path, task_id)
        n_points = len(grid_points(optuna_config_path))
        if optuna_cfg.num_trials != n_points:
            raise ValueError(f"num_trials ({optuna_cfg.num_trials}) must equal the grid size ({n_points})")
        TERM_LOGGER.info(f"Grid point {task_id}/{n_points - 1}: {point}")
        return optuna.samplers.PartialFixedSampler(point, optuna.samplers.RandomSampler()), point
    if optuna_cfg.sampler == "tpe":
        return optuna.samplers.TPESampler(
            n_startup_trials=20, n_ei_candidates=32, multivariate=True, group=True, constant_liar=True
        ), None
    raise ValueError(f"sampler must be 'grid' or 'tpe', got {optuna_cfg.sampler!r}")


def perform_study(optuna_cfg: OptunaDiffusionConfig, train_config, task_id: int | None, optuna_config_path: str) -> None:
    storage = _resolve_storage(optuna_cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(optuna_cfg.gpu)
    train_config["device"] = "cuda:0"

    sampler, point = _make_sampler(optuna_cfg, task_id, optuna_config_path)
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=optuna_cfg.pruner_n_startup_trials,
            n_warmup_steps=optuna_cfg.pruner_n_warmup_steps,
            interval_steps=1,
            n_min_trials=optuna_cfg.pruner_n_min_trials,
        )
        if optuna_cfg.pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=optuna_cfg.study_name, storage=storage, directions=optuna_cfg.directions,
        load_if_exists=True, sampler=sampler, pruner=pruner,
    )

    _complete = (optuna.trial.TrialState.COMPLETE,)
    done = [t for t in study.trials if t.state in _complete]
    TERM_LOGGER.info(
        f"Study '{optuna_cfg.study_name}' — {len(done)} complete of {optuna_cfg.num_trials} "
        f"({len(study.trials)} total incl. failed/running). Running 1 trial this process."
    )
    if len(done) >= optuna_cfg.num_trials:
        TERM_LOGGER.info("Target number of trials already reached. Exiting.")
        return
    if point is not None and any(t.params == point for t in done):
        # Resubmitting an array index whose point already finished must be a no-op, or a
        # careless `sbatch --array=0-11` after a partial run would train finished points again.
        TERM_LOGGER.info(f"Grid point {point} already has a COMPLETE trial. Exiting.")
        return

    study.optimize(
        lambda trial: objective(trial, train_config, optuna_cfg),
        n_trials=1,
        callbacks=[MaxTrialsCallback(optuna_cfg.num_trials, states=_complete)],
        catch=[FailedTrialException],
        gc_after_trial=True,
    )
    try:
        best = study.best_trial
        TERM_LOGGER.info(f"Best trial so far: #{best.number}  values={best.values}  params={best.params}")
    except Exception:  # noqa: BLE001 — only failed trials so far
        pass


# ============================================================
# BLOCK 6 — Entry point
# ============================================================

if __name__ == "__main__":
    require_cuda()
    import_ftr_tasks()
    try:
        optuna_cfg_raw = OmegaConf.load(args.optuna_config)
        if unknown_args:
            optuna_cfg_raw = OmegaConf.merge(optuna_cfg_raw, OmegaConf.from_dotlist(unknown_args))
        optuna_cfg = OptunaDiffusionConfig(**optuna_cfg_raw)

        # On a respawn the sbatch has made this task look like a single SLURM job, so the
        # previous attempt's config.yaml is found the same way train_diffusion.py finds it. It
        # already carries the grid point (the trainer saved the merged config), and the point
        # is re-applied on top anyway — the resume must train the parameters the checkpoint was
        # trained with, not whatever the base config on disk says today.
        prev_cfg_path = RunLogger.latest_attempt_config()
        if prev_cfg_path is not None:
            TERM_LOGGER.info(f"Respawn detected — base config from previous attempt: {prev_cfg_path}")
            train_config = OmegaConf.load(prev_cfg_path)
        else:
            train_config = OmegaConf.load(args.train_config)
        train_config = OmegaConf.merge(train_config, optuna_cfg.train_config_overrides)
        if optuna_cfg.frozen_params:
            train_config = OmegaConf.merge(
                train_config, OmegaConf.from_dotlist([f"{k}={v}" for k, v in optuna_cfg.frozen_params.items()])
            )
        if args.num_envs is not None:
            train_config.num_robots = args.num_envs

        n_keys = len(optuna_cfg.optuna_keys)
        if len(optuna_cfg.optuna_types) != n_keys or len(optuna_cfg.optuna_values) != n_keys:
            raise ValueError("optuna_keys, optuna_types, and optuna_values must have the same length.")
        if optuna_cfg.metric_weights is None and len(optuna_cfg.directions) != len(optuna_cfg.metrics_to_optimize):
            raise ValueError("directions and metrics_to_optimize must have the same length.")
        if optuna_cfg.metric_weights is not None and len(optuna_cfg.directions) != 1:
            raise ValueError("When metric_weights is set, directions must have exactly one entry.")

        task_id = args.task_id
        if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        TERM_LOGGER.info(f"Search space: {n_keys} parameters, {optuna_cfg.num_trials} trials, sampler={optuna_cfg.sampler}")
        TERM_LOGGER.info(f"Optimising:   {optuna_cfg.metrics_to_optimize}  ({optuna_cfg.directions})")

        perform_study(optuna_cfg, train_config, task_id, args.optuna_config)
    except Exception as _e:  # noqa: BLE001
        traceback.print_exception(_e)
        os._exit(1)

    # Skip simulation_app.close(): Isaac Sim's shutdown frequently deadlocks. One trial per
    # process, so a hard exit is safe — flush first, or the summary above is lost (stdout is a
    # file under SLURM and therefore block-buffered).
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)
