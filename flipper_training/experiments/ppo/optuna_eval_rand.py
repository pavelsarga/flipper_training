# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Optuna hyperparameter search for random policy (Isaac Sim backend).")
parser.add_argument("--config", "-c", type=str, required=True, help="Path to rand_policy_eval.yaml.")
parser.add_argument("--optuna_config", "-o", type=str, required=True, help="Path to optuna search-space YAML.")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in eval config.")
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
from dataclasses import dataclass, field
from typing import Any

import optuna
from omegaconf import OmegaConf
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback

import flipper_training  # registers OmegaConf resolvers
from flipper_training.environment.ftr_env_adapter import FtrTorchRLEnv
from flipper_training.experiments.ppo.common import make_transformed_env
from flipper_training.experiments.ppo.train_ftr import FtrPPOConfig
from flipper_training.utils.logutils import get_terminal_logger
from flipper_training.utils.torch_utils import seed_all, set_device
from torchrl.envs.utils import ExplorationType, set_exploration_type

import gymnasium

TERM_LOGGER = get_terminal_logger("optuna_eval_rand")


# ============================================================
# BLOCK 3 — Config and search-space helpers
# ============================================================

@dataclass
class OptunaRandConfig:
    study_name: str
    directions: list[str]
    metrics_to_optimize: list[str]
    num_trials: int
    gpu: int
    db_url: str                         # explicit DB URL — separate from the PPO optuna DB
    optuna_keys: list[str]
    optuna_types: list[str]             # float | log_float | int | categorical | bool
    optuna_values: list
    eval_repeats: int = 3               # rollouts averaged per trial
    metric_weights: list[float] | None = None


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


def _expand_aliases(params: dict) -> dict:
    """Expand amp/freq alias keys into the concrete policy_opts min/max pairs."""
    for key in ("amp", "_amp"):
        if key in params:
            v = params.pop(key)
            params["policy_opts.amp_min"] = v
            params["policy_opts.amp_max"] = v
            break
    for key in ("freq", "_freq"):
        if key in params:
            v = params.pop(key)
            params["policy_opts.freq_min"] = v
            params["policy_opts.freq_max"] = v
            break
    return params


class FailedTrialException(Exception):
    pass


def _run_eval(raw_cfg, ftr_gym_env, max_steps: int, repeats: int) -> dict[str, float]:
    """Create TorchRL wrappers + policy from raw_cfg, run rollouts, return averaged metrics."""
    import torch
    cfg = FtrPPOConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)

    ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device)
    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    policy_cfg = cfg.policy_config(**cfg.policy_opts)
    actor_value_wrapper, _, policy_transforms = policy_cfg.create(
        env=ftr_torchrl_env,
        weights_path=None,
        device=device,
    )
    actor = actor_value_wrapper.get_policy_operator()
    env, _ = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms)
    actor_value_wrapper.eval()
    env.eval()

    all_results = []
    for _ in range(repeats):
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            rollout = env.rollout(max_steps, actor, auto_reset=True, break_when_all_done=True)
        step_results: dict[str, float] = {
            "eval/mean_step_reward": rollout["next", "reward"].mean().item(),
            "eval/pct_truncated":    rollout["next", "truncated"].float().mean().item(),
        }
        del rollout
        term_info = ftr_torchrl_env.pop_termination_info()
        step_results.update({"eval/" + k.split("/", 1)[-1]: v for k, v in term_info.items()})
        step_results.update(ftr_torchrl_env.pop_reward_info())
        all_results.append(step_results)

    return {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}


# ============================================================
# BLOCK 4 — Objective (one trial = sample params + run eval)
# ============================================================

def objective(
    trial: optuna.Trial,
    base_raw_cfg,
    ftr_gym_env,
    max_steps: int,
    optuna_cfg: OptunaRandConfig,
) -> tuple[float, ...]:
    """
    Run evaluation of the random policy with sampled policy_opts.

    The gymnasium env is created once in __main__ and reused across trials —
    only the TorchRL wrappers and the policy are recreated per trial.
    """
    params = define_search_space(
        trial, optuna_cfg.optuna_keys, optuna_cfg.optuna_types, optuna_cfg.optuna_values
    )
    params = _expand_aliases(params)

    TERM_LOGGER.info(f"Trial {trial.number} — sampled parameters:")
    for k, v in params.items():
        TERM_LOGGER.info(f"  {k} = {v}")

    dotlist = [f"{k}={v}" for k, v in params.items()]
    merged_cfg = OmegaConf.merge(base_raw_cfg, OmegaConf.from_dotlist(dotlist))

    try:
        results = _run_eval(merged_cfg, ftr_gym_env, max_steps, optuna_cfg.eval_repeats)
    except Exception as e:
        traceback.print_exception(e)
        if "CUDA error" in str(e) or "CUDA out of memory" in str(e):
            _marked = False
            try:
                from optuna.trial import TrialState as _TS
                trial._storage.set_trial_state_values(trial._trial_id, _TS.FAIL, values=None)
                _marked = True
                TERM_LOGGER.info(f"Trial {trial.number} marked as FAIL in Optuna DB (CUDA crash).")
            except Exception as _mark_err:
                TERM_LOGGER.warning(f"Optuna API FAIL mark failed: {_mark_err} — trying direct SQLite.")
            if not _marked:
                try:
                    import sqlite3 as _sq3
                    import datetime as _dt
                    _url = optuna_cfg.db_url
                    if _url.startswith("sqlite:///"):
                        _sp = _url[len("sqlite:///"):]
                        with _sq3.connect(_sp, timeout=30) as _conn:
                            _conn.execute(
                                "UPDATE trials SET state='FAIL', datetime_complete=? "
                                "WHERE trial_id=? AND state='RUNNING'",
                                (_dt.datetime.utcnow().isoformat(sep=" "), trial._trial_id),
                            )
                        TERM_LOGGER.info(f"Trial {trial.number} marked as FAIL via direct SQLite.")
                except Exception as _sq_err:
                    TERM_LOGGER.warning(f"Direct SQLite FAIL mark also failed: {_sq_err}")
            import os as _os
            _os._exit(75)
        raise FailedTrialException(f"Trial {trial.number} failed: {e}") from e

    result = []
    for metric in optuna_cfg.metrics_to_optimize:
        val = results.get(metric, 0.0)
        if val == 0.0 and metric not in results:
            TERM_LOGGER.warning(f"Metric '{metric}' missing from results (available: {list(results.keys())}). Using 0.0.")
        result.append(val)

    for metric_name, metric_val in zip(optuna_cfg.metrics_to_optimize, result):
        trial.set_user_attr(metric_name, metric_val)

    if optuna_cfg.metric_weights is not None:
        score = sum(w * v for w, v in zip(optuna_cfg.metric_weights, result))
        trial.set_user_attr("score", score)
        TERM_LOGGER.info(
            f"Trial {trial.number} done — score={score:.4f}  "
            f"components={dict(zip(optuna_cfg.metrics_to_optimize, result))}"
        )
        return (score,)

    TERM_LOGGER.info(f"Trial {trial.number} done — {dict(zip(optuna_cfg.metrics_to_optimize, result))}")
    return tuple(result)


# ============================================================
# BLOCK 5 — Study runner
# ============================================================

def perform_study(
    optuna_cfg: OptunaRandConfig,
    base_raw_cfg,
    ftr_gym_env,
    max_steps: int,
) -> None:
    conn_str = optuna_cfg.db_url

    if conn_str.startswith("sqlite:///"):
        import pathlib
        db_path = conn_str[len("sqlite:///"):]
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine_kwargs = {"connect_args": {"timeout": 300}}
        # Fix any lowercase 'finite' rows written by buggy recovery scripts
        import sqlite3 as _sqlite3
        with _sqlite3.connect(db_path, timeout=60) as _conn:
            n = _conn.execute(
                "UPDATE trial_values SET value_type = 'FINITE' WHERE value_type = 'finite'"
            ).rowcount
            if n > 0:
                TERM_LOGGER.warning(f"Fixed {n} trial_values row(s) with lowercase 'finite' → 'FINITE'")
    else:
        engine_kwargs = {}

    storage = RDBStorage(conn_str, engine_kwargs=engine_kwargs)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(optuna_cfg.gpu)

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=storage,
        directions=optuna_cfg.directions,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,    # smaller search space → fewer random startup trials
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            constant_liar=True,
        ),
    )

    _count_states = (optuna.trial.TrialState.COMPLETE,)
    n_existing = len([t for t in study.trials if t.state in _count_states])
    TERM_LOGGER.info(
        f"Study '{optuna_cfg.study_name}' — {n_existing} complete trials "
        f"({len(study.trials)} total), targeting {optuna_cfg.num_trials}. Running 1 trial."
    )

    if n_existing >= optuna_cfg.num_trials:
        TERM_LOGGER.info("Target number of trials already reached. Exiting.")
        return

    study.optimize(
        lambda trial: objective(trial, base_raw_cfg, ftr_gym_env, max_steps, optuna_cfg),
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
        pass


# ============================================================
# BLOCK 6 — Entry point
# ============================================================

if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print(
            "FATAL: torch.cuda.is_available() returned False after AppLauncher init.\n"
            "Isaac Sim failed to create a CUDA context — try a different compute node.",
            flush=True,
        )
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    try:
        from pathlib import Path

        # ---- load configs ----
        optuna_cfg_raw = OmegaConf.load(args.optuna_config)
        if unknown_args:
            optuna_cfg_raw = OmegaConf.merge(optuna_cfg_raw, OmegaConf.from_dotlist(unknown_args))
        optuna_cfg = OptunaRandConfig(**optuna_cfg_raw)

        raw_cfg = OmegaConf.load(args.config)
        raw_cfg.use_wandb = False
        raw_cfg.use_tensorboard = False
        if args.num_envs is not None:
            raw_cfg.num_robots = args.num_envs

        # ---- validate ----
        n_keys = len(optuna_cfg.optuna_keys)
        if len(optuna_cfg.optuna_types) != n_keys or len(optuna_cfg.optuna_values) != n_keys:
            raise ValueError("optuna_keys, optuna_types, and optuna_values must have the same length.")
        if optuna_cfg.metric_weights is None and len(optuna_cfg.directions) != len(optuna_cfg.metrics_to_optimize):
            raise ValueError("directions and metrics_to_optimize must have the same length.")

        TERM_LOGGER.info(f"Search space: {n_keys} parameters over {optuna_cfg.num_trials} total trials.")
        TERM_LOGGER.info(f"Optimising:   {optuna_cfg.metrics_to_optimize}  ({optuna_cfg.directions})")

        # ---- build env (created once; reused across trials) ----
        _cfg = FtrPPOConfig(**raw_cfg)
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

        env_cfg = _EnvCfgClass()
        env_cfg.scene.num_envs = _cfg.num_robots
        env_cfg.terrain_name = _cfg.terrain
        env_cfg.sim.dt = _cfg.sim_dt
        env_cfg.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
        env_cfg.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
        env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = _cfg.max_depenetration_velocity
        env_cfg.robot.spawn.rigid_props.linear_damping = _cfg.robot_linear_damping
        env_cfg.robot.spawn.rigid_props.angular_damping = _cfg.robot_angular_damping
        env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = _cfg.solver_position_iterations
        env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = _cfg.solver_velocity_iterations
        env_cfg.sim.physx.min_position_iteration_count = _cfg.solver_position_iterations
        env_cfg.sim.physx.max_velocity_iteration_count = _cfg.solver_velocity_iterations
        env_cfg.sim.physx.bounce_threshold_velocity = _cfg.bounce_threshold_velocity
        env_cfg.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
        env_cfg.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
        env_cfg.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
        if _cfg.num_robots <= 64:
            env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
            env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
            env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
            env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
            env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22
        elif _cfg.num_robots > 512:
            env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 27
        for k, v in (_cfg.env_cfg_overrides or {}).items():
            setattr(env_cfg, k, v)

        max_steps = getattr(raw_cfg, "max_eval_steps", 0)

        ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)
        try:
            perform_study(optuna_cfg, raw_cfg, ftr_gym_env, max_steps)
        finally:
            ftr_gym_env.close()

    except Exception as _e:
        traceback.print_exception(_e)
        os._exit(1)

    # Skip simulation_app.close() — Isaac Sim shutdown frequently deadlocks.
    os._exit(0)
