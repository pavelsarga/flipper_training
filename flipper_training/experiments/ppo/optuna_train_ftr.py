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

    # Build a fresh env_cfg with trial-specific env_cfg_overrides applied
    trial_cfg = FtrPPOConfig(**updated_config)
    env_cfg = copy.deepcopy(env_cfg_base)
    for k, v in (trial_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(task, cfg=env_cfg)
    try:
        trainer = FtrPPOTrainer(updated_config, ftr_gym_env)
        metrics = trainer.train()
    except Exception as e:
        traceback.print_exception(e)
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
    conn_str = (
        f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
        f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode=require"
    )
    storage = RDBStorage(conn_str)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(optuna_cfg.gpu)
    train_config["device"] = "cuda:0"

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=storage,
        directions=optuna_cfg.directions,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=None),
    )

    n_existing = len(study.trials)
    TERM_LOGGER.info(
        f"Study '{optuna_cfg.study_name}' — {n_existing} existing trials, "
        f"targeting {optuna_cfg.num_trials} total. Running 1 trial this process."
    )

    if n_existing >= optuna_cfg.num_trials:
        TERM_LOGGER.info("Target number of trials already reached. Exiting.")
        return

    # One trial per Isaac Sim process; MaxTrialsCallback stops all workers once done.
    study.optimize(
        lambda trial: objective(trial, train_config, env_cfg_base, EnvCfgClass, task, optuna_cfg),
        n_trials=1,
        callbacks=[MaxTrialsCallback(optuna_cfg.num_trials)],
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
    import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls

    # ---- load configs ----
    optuna_cfg_raw = OmegaConf.load(args.optuna_config)
    if unknown_args:
        optuna_cfg_raw = OmegaConf.merge(optuna_cfg_raw, OmegaConf.from_dotlist(unknown_args))
    optuna_cfg = OptunaFtrConfig(**optuna_cfg_raw)

    train_config = OmegaConf.load(args.train_config)
    train_config = OmegaConf.merge(train_config, optuna_cfg.train_config_overrides)
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

    # ---- build base env_cfg (without env_cfg_overrides — applied per trial) ----
    env_cfg_base = _EnvCfgClass()
    env_cfg_base.scene.num_envs = _cfg.num_robots
    env_cfg_base.terrain_name = _cfg.terrain
    env_cfg_base.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
    env_cfg_base.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
    env_cfg_base.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
    env_cfg_base.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
    env_cfg_base.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
    # env_cfg_overrides applied per-trial inside objective()

    perform_study(optuna_cfg, train_config, env_cfg_base, _EnvCfgClass, _cfg.task)

    simulation_app.close()
