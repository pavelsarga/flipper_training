# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Optuna hyperparameter search for FTR PPO (Isaac Sim backend).")
parser.add_argument("--train_config", "-t", type=str, required=True, help="Path to base ftr_config YAML.")
parser.add_argument("--optuna_config", "-o", type=str, required=True, help="Path to optuna search-space YAML.")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in train config.")
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import copy
import importlib
import os
import traceback
from dataclasses import dataclass
from typing import Any

import optuna
from omegaconf import OmegaConf
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback

import flipper_training  # registers OmegaConf resolvers
from flipper_training import ROOT
from flipper_training.experiments.ppo.train_ftr import FtrPPOConfig, FtrPPOTrainer
from flipper_training.utils.logutils import get_terminal_logger

import gymnasium

TERM_LOGGER = get_terminal_logger("optuna_ftr")


# ============================================================
# BLOCK 3 — Config and search-space helpers
# ============================================================

@dataclass
class OptunaFtrConfig:
    study_name: str
    directions: list[str]
    metrics_to_optimize: list[str]
    num_trials: int                    # total across all workers (enforced by MaxTrialsCallback)
    gpu: int
    train_config_overrides: dict[str, Any]
    optuna_keys: list[str]
    optuna_types: list[str]            # float | log_float | int | categorical | bool
    optuna_values: list                # [low, high] for numeric; list of choices for categorical
    frozen_params: dict[str, Any] | None = None  # params frozen from a previous stage (dotlist keys)
    slurm_params: dict[str, Any] | None = None


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


# ============================================================
# BLOCK 4 — Objective (one trial = create env + train + close env)
# ============================================================

def objective(
    trial: optuna.Trial,
    base_config,
    env_cfg_base,          # env cfg WITHOUT env_cfg_overrides applied — we apply per-trial below
    EnvCfgClass: type,
    task: str,
    optuna_cfg: OptunaFtrConfig,
) -> tuple[float, ...]:
    """
    Run a single FTR PPO training run for the sampled hyperparameters.

    The gymnasium env is created fresh for this trial so that env_cfg_overrides
    (reward parameters) are applied with the trial-specific values, then closed
    on completion.  Each SLURM array task runs n_trials=1, so this function is
    called exactly once per Isaac Sim process.
    """
    params = define_search_space(
        trial, optuna_cfg.optuna_keys, optuna_cfg.optuna_types, optuna_cfg.optuna_values
    )
    TERM_LOGGER.info(f"Trial {trial.number} — sampled parameters:")
    for k, v in params.items():
        TERM_LOGGER.info(f"  {k} = {v}")

    # Merge trial parameters into the training config
    dotlist = [f"{k}={v}" for k, v in params.items()]
    updated_config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))

    # Build a fresh env_cfg with trial-specific overrides applied
    trial_cfg = FtrPPOConfig(**updated_config)
    env_cfg = copy.deepcopy(env_cfg_base)

    # Physics params (top-level config keys → env_cfg.sim / env_cfg.robot)
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

    # Reward / env params (env_cfg_overrides → setattr on env_cfg)
    for k, v in (trial_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(task, cfg=env_cfg)
    try:
        trainer = FtrPPOTrainer(updated_config, ftr_gym_env, optuna_trial=trial)
        metrics = trainer.train()
    except optuna.TrialPruned:
        ftr_gym_env.close()
        raise  # Optuna handles this internally, marks trial as PRUNED
    except Exception as e:
        traceback.print_exception(e)
        if "CUDA error" in str(e) or "CUDA out of memory" in str(e):
            # CUDA context is dead — env.close() and Isaac Sim atexit handlers will
            # deadlock.  Mark the trial as FAIL in Optuna before hard-exiting.
            try:
                from optuna.trial import TrialState as _TS
                trial._storage.set_trial_state_values(trial._trial_id, _TS.FAIL, values=None)
                TERM_LOGGER.info(f"Trial {trial.number} marked as FAIL in Optuna DB (CUDA crash).")
            except Exception as _mark_err:
                TERM_LOGGER.warning(f"Could not mark trial as FAIL before exit: {_mark_err}")
            import os as _os
            _os._exit(75)
        ftr_gym_env.close()
        raise FailedTrialException(f"Trial {trial.number} failed: {e}") from e

    ftr_gym_env.close()

    if metrics is None:
        raise FailedTrialException(f"Trial {trial.number} returned None (training was interrupted).")

    result = []
    for metric in optuna_cfg.metrics_to_optimize:
        if metric not in metrics:
            TERM_LOGGER.warning(
                f"Metric '{metric}' missing from trial results "
                f"(available: {list(metrics.keys())}). Using 0.0."
            )
            result.append(0.0)
        else:
            result.append(metrics[metric])

    TERM_LOGGER.info(f"Trial {trial.number} done — {dict(zip(optuna_cfg.metrics_to_optimize, result))}")
    return tuple(result)


# ============================================================
# BLOCK 5 — Study runner
# ============================================================

def perform_study(
    optuna_cfg: OptunaFtrConfig,
    train_config,
    env_cfg_base,
    EnvCfgClass: type,
    task: str,
) -> None:
    db_secret_path = ROOT / "optuna_db.yaml"
    if not db_secret_path.exists():
        raise FileNotFoundError(
            f"optuna_db.yaml not found at {db_secret_path}. "
            "Create it with keys: db_user, db_password, db_host, db_port, db_name."
        )
    db_secret = OmegaConf.load(db_secret_path)
    if "url" in db_secret:
        conn_str = db_secret["url"]
        if conn_str.startswith("sqlite:///"):
            import pathlib
            pathlib.Path(conn_str[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        # SQLite: give each writer up to 5 min to acquire the file lock.
        # Without this, parallel SLURM tasks racing on schema init get "database is locked".
        engine_kwargs = {"connect_args": {"timeout": 300}} if conn_str.startswith("sqlite:///") else {}
    else:
        sslmode = db_secret.get("sslmode", "require")
        conn_str = (
            f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
            f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode={sslmode}"
        )
        engine_kwargs = {}
    storage = RDBStorage(conn_str, engine_kwargs=engine_kwargs)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(optuna_cfg.gpu)
    train_config["device"] = "cuda:0"

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=storage,
        directions=optuna_cfg.directions,
        load_if_exists=True,
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,    # 2x your parallelism — more robust with noisy RL objectives
            n_ei_candidates=32,     # more candidates per suggestion (worth it for expensive trials)
            multivariate=True,      # model parameter interactions jointly (critical for 20 params)
            group=True,             # decompose into correlated subgroups (reduces curse of dimensionality)
            constant_liar=True,     # essential for ~10 parallel SLURM workers
        ),
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,   # don't prune until 20 trials complete (reliable baseline)
            n_warmup_steps=30,     # don't prune before eval report 30
            interval_steps=1,      # check at every eval report
            n_min_trials=5,        # need >= 5 completed trials at this step to compare
        )
    )

    _count_states = (optuna.trial.TrialState.COMPLETE,)
    n_existing = len([t for t in study.trials if t.state in _count_states])
    TERM_LOGGER.info(
        f"Study '{optuna_cfg.study_name}' — {n_existing} complete trials "
        f"({len(study.trials)} total incl. failed/running), "
        f"targeting {optuna_cfg.num_trials}. Running 1 trial this process."
    )

    if n_existing >= optuna_cfg.num_trials:
        TERM_LOGGER.info("Target number of trials already reached. Exiting.")
        return

    # One trial per Isaac Sim process; MaxTrialsCallback stops all workers once done.
    # Only count COMPLETE trials — FAIL and RUNNING don't consume budget.
    study.optimize(
        lambda trial: objective(trial, train_config, env_cfg_base, EnvCfgClass, task, optuna_cfg),
        n_trials=1,
        callbacks=[MaxTrialsCallback(optuna_cfg.num_trials, states=_count_states)],
        catch=[FailedTrialException],
        gc_after_trial=True,
    )

    try:
        best = study.best_trial
        TERM_LOGGER.info(
            f"Best trial so far: #{best.number}  values={best.values}  params={best.params}"
        )
    except Exception:
        pass  # study may have only failed trials; ignore


# ============================================================
# BLOCK 6 — Entry point
# ============================================================

if __name__ == "__main__":
    import os
    import torch

    # Verify CUDA is accessible before importing FTR tasks.
    # Importing ftr_envs.tasks triggers wp.init() (via omni.isaac.lab.envs chain),
    # which crashes with an unhelpful RuntimeError if the CUDA context is dead.
    # Use os._exit() — not sys.exit/raise — so Isaac Sim's atexit handlers are bypassed
    # and the apptainer process terminates immediately instead of hanging for minutes.
    if not torch.cuda.is_available():
        print(
            "FATAL: torch.cuda.is_available() returned False after AppLauncher init.\n"
            "Isaac Sim failed to create a CUDA context (check .err for 'CUDA error 46').\n"
            "This is usually a node-level GPU issue — try a different compute node.",
            flush=True,
        )
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    try:
        # ---- load configs ----
        optuna_cfg_raw = OmegaConf.load(args.optuna_config)
        if unknown_args:
            optuna_cfg_raw = OmegaConf.merge(optuna_cfg_raw, OmegaConf.from_dotlist(unknown_args))
        optuna_cfg = OptunaFtrConfig(**optuna_cfg_raw)

        train_config = OmegaConf.load(args.train_config)
        train_config = OmegaConf.merge(train_config, optuna_cfg.train_config_overrides)
        if optuna_cfg.frozen_params:
            frozen_dotlist = [f"{k}={v}" for k, v in optuna_cfg.frozen_params.items()]
            train_config = OmegaConf.merge(train_config, OmegaConf.from_dotlist(frozen_dotlist))
            TERM_LOGGER.info(f"Frozen params from previous stage: {optuna_cfg.frozen_params}")
        if args.num_envs is not None:
            train_config.num_robots = args.num_envs

        # ---- validate ----
        n_keys = len(optuna_cfg.optuna_keys)
        if len(optuna_cfg.optuna_types) != n_keys or len(optuna_cfg.optuna_values) != n_keys:
            raise ValueError("optuna_keys, optuna_types, and optuna_values must have the same length.")
        if len(optuna_cfg.directions) != len(optuna_cfg.metrics_to_optimize):
            raise ValueError("directions and metrics_to_optimize must have the same length.")

        TERM_LOGGER.info(f"Search space: {n_keys} parameters over {optuna_cfg.num_trials} total trials.")
        TERM_LOGGER.info(f"Optimising:   {optuna_cfg.metrics_to_optimize}  ({optuna_cfg.directions})")

        # ---- resolve env cfg class from task registry ----
        _cfg = FtrPPOConfig(**train_config)
        spec = gymnasium.spec(_cfg.task)
        _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
            _mod_path, _cls_name = _env_cfg_entry.rsplit(":", 1)
            _EnvCfgClass = getattr(importlib.import_module(_mod_path), _cls_name)
        elif isinstance(_env_cfg_entry, type):
            _EnvCfgClass = _env_cfg_entry
        else:
            from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg
            _EnvCfgClass = CrossingEnvCfg

        # ---- build base env_cfg ----
        # Only set structural params (num_envs, terrain, GPU buffers) here.
        # Physics tuning + env_cfg_overrides are applied per-trial inside objective().
        env_cfg_base = _EnvCfgClass()
        env_cfg_base.scene.num_envs = _cfg.num_robots
        env_cfg_base.terrain_name = _cfg.terrain
        env_cfg_base.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
        env_cfg_base.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
        env_cfg_base.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions

        perform_study(optuna_cfg, train_config, env_cfg_base, _EnvCfgClass, _cfg.task)
    except Exception as _e:
        traceback.print_exception(_e)
        os._exit(1)

    # Skip simulation_app.close() — Isaac Sim's shutdown re-initialises GPU foundation
    # and frequently deadlocks, keeping the SLURM slot busy for hours.
    # Each Optuna trial is a one-shot process, so os._exit(0) is safe.
    os._exit(0)
