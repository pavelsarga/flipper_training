# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate top-K Optuna trials and compare their performance."
)
parser.add_argument("--study", type=str, required=True, help="Optuna study name.")
parser.add_argument("--top", type=int, default=10, help="Number of top trials to evaluate (default: 10).")
parser.add_argument("--repeats", type=int, default=20, help="Eval rollouts per trial (default: 20).")
parser.add_argument("--num_envs", type=int, default=128, help="Override num_robots from config.")
parser.add_argument("--max_steps", type=int, default=100000, help="Override max_eval_steps from config.")
parser.add_argument(
    "--policy", type=str, default="policy_final.pth",
    help="Policy checkpoint filename inside <run>/weights/.",
)
parser.add_argument(
    "--vecnorm", type=str, default="vecnorm_final.pth",
    help="VecNorm checkpoint filename inside <run>/weights/.",
)
parser.add_argument("--output", type=str, default=None, help="Path to save results CSV (optional).")
parser.add_argument("--db", type=str, default=None, help="Optuna DB URL (e.g. sqlite:///path/to/optuna.db). Overrides optuna_db.yaml.")
parser.add_argument(
    "--runs-dir", type=str, default=None,
    help="Directory to scan for run subdirs (fallback when trials lack logpath). "
         "Defaults to logs/ and runs/ppo/ under the workspace root.",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import csv
import importlib
import os
import sys
import traceback
from pathlib import Path

import optuna
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import flipper_training  # registers OmegaConf resolvers
from flipper_training import ROOT
from flipper_training.environment.ftr_env_adapter import FtrTorchRLEnv
from flipper_training.experiments.ppo.common import make_transformed_env
from flipper_training.experiments.ppo.train_ftr import FtrPPOConfig
from flipper_training.utils.logutils import get_terminal_logger
from flipper_training.utils.torch_utils import seed_all, set_device

import gymnasium

logger = get_terminal_logger("eval_optuna_top")


# ============================================================
# BLOCK 3 — Helpers
# ============================================================

def get_storage(db_url: str | None = None):
    """Build Optuna storage from --db URL or optuna_db.yaml."""
    if db_url:
        conn_str = db_url
    else:
        db_path = ROOT / "optuna_db.yaml"
        if not db_path.exists():
            raise FileNotFoundError(f"optuna_db.yaml not found at {db_path}")
        db_secret = OmegaConf.load(db_path)
        if "url" in db_secret:
            conn_str = db_secret["url"]
        else:
            sslmode = db_secret.get("sslmode", "require")
            conn_str = (
                f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
                f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode={sslmode}"
            )
    engine_kwargs = {"connect_args": {"timeout": 300}} if conn_str.startswith("sqlite:///") else {}
    return optuna.storages.RDBStorage(conn_str, engine_kwargs=engine_kwargs)


def get_top_trials(study_name: str, top_k: int, db_url: str | None = None) -> list[optuna.trial.FrozenTrial]:
    """Return the top-K completed trials sorted by value (descending)."""
    storage = get_storage(db_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    return completed[:top_k]


def _match_trial_params(trial_params: dict, cfg_dict: dict, env_overrides_key: str = "env_cfg_overrides") -> bool:
    """Check if a config dict matches a trial's sampled params."""
    env_overrides = cfg_dict.get(env_overrides_key) or {}
    matched = 0
    for k, v in trial_params.items():
        # Check top-level config keys first, then env_cfg_overrides
        cfg_val = cfg_dict.get(k)
        if cfg_val is None:
            cfg_val = env_overrides.get(k)
        if cfg_val is None:
            # Try dotlist-style key (e.g. "env_cfg_overrides.roll_coef")
            parts = k.split(".")
            d = cfg_dict
            for part in parts:
                if isinstance(d, dict):
                    d = d.get(part)
                else:
                    d = None
                    break
            cfg_val = d
        if cfg_val is None:
            continue
        # Compare with tolerance for floats
        if isinstance(v, float) and isinstance(cfg_val, (int, float)):
            if abs(v - float(cfg_val)) < max(abs(v) * 1e-4, 1e-8):
                matched += 1
        elif v == cfg_val:
            matched += 1
    # Require at least 80% of trial params to match (some might not be in config)
    return matched >= max(len(trial_params) * 0.8, 1)


# Module-level cache for the run index
_RUN_INDEX_CACHE: list[tuple[Path, dict]] | None = None


def _get_run_index(search_dirs: list[Path]) -> list[tuple[Path, dict]]:
    global _RUN_INDEX_CACHE
    if _RUN_INDEX_CACHE is None:
        logger.info(f"Building run directory index from: {[str(d) for d in search_dirs]} ...")
        entries = []
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for config_path in search_dir.rglob("config.yaml"):
                run_dir = config_path.parent
                if not (run_dir / "weights").is_dir():
                    continue
                try:
                    cfg = OmegaConf.load(config_path)
                    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                except Exception:
                    continue
                entries.append((run_dir, cfg_dict))
        logger.info(f"  Indexed {len(entries)} run directories with weights.")
        _RUN_INDEX_CACHE = entries
    return _RUN_INDEX_CACHE


def find_run_dir(trial: optuna.trial.FrozenTrial, search_dirs: list[Path] | None = None) -> Path | None:
    """Find the run directory for a trial via user_attrs or by scanning directories."""
    # 1. Direct path from user_attrs (new trials)
    logpath = trial.user_attrs.get("logpath")
    if logpath:
        p = Path(logpath)
        if p.exists() and (p / "config.yaml").exists():
            return p

    # 2. Fallback: scan run directories and match by trial params
    if search_dirs is None:
        return None

    run_index = _get_run_index(search_dirs)
    for run_dir, cfg_dict in run_index:
        if _match_trial_params(trial.params, cfg_dict):
            return run_dir

    return None


def _run_single_rollout(env, ftr_torchrl_env: FtrTorchRLEnv, actor, max_steps: int) -> dict[str, float]:
    """Run one deterministic rollout and return a flat dict of metrics."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        rollout = env.rollout(max_steps, actor, auto_reset=True, break_when_all_done=True)

    results: dict[str, float] = {
        "eval/mean_step_reward": rollout["next", "reward"].mean().item(),
        "eval/max_step_reward":  rollout["next", "reward"].max().item(),
        "eval/min_step_reward":  rollout["next", "reward"].min().item(),
        "eval/pct_terminated":   rollout["next", "terminated"].float().mean().item(),
        "eval/pct_truncated":    rollout["next", "truncated"].float().mean().item(),
        "eval/rollout_steps":    float(rollout.shape[1]),
    }
    del rollout

    term_info = ftr_torchrl_env.pop_termination_info()
    results.update({"eval/" + k.split("/", 1)[-1]: v for k, v in term_info.items()})
    results.update(ftr_torchrl_env.pop_reward_info())
    return results


def eval_trial(
    run_dir: Path,
    repeats: int,
    num_envs: int | None,
    max_steps_override: int | None,
    policy_filename: str,
    vecnorm_filename: str,
) -> dict[str, float] | None:
    """Evaluate a single trial's weights. Returns averaged metrics or None on failure."""
    config_path = run_dir / "config.yaml"
    weights_dir = run_dir / "weights"
    policy_path = weights_dir / policy_filename
    vecnorm_path = weights_dir / vecnorm_filename

    if not policy_path.exists():
        logger.warning(f"  Policy not found: {policy_path}")
        return None
    if not vecnorm_path.exists():
        logger.warning(f"  VecNorm not found: {vecnorm_path} — proceeding without")
        vecnorm_path = None

    raw_cfg = OmegaConf.load(config_path)
    raw_cfg.policy_weights_path = str(policy_path)
    raw_cfg.vecnorm_weights_path = str(vecnorm_path) if vecnorm_path else ""
    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False
    if num_envs is not None:
        raw_cfg.num_robots = num_envs

    max_steps = max_steps_override if max_steps_override is not None else raw_cfg.get("max_eval_steps", 100000)

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
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)
    device = set_device(_cfg.device)
    seed_all(_cfg.seed)

    ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=_cfg.ftr_obs_encoder_opts, device=device)
    policy_cfg_obj = _cfg.policy_config(**_cfg.policy_opts)
    actor_value_wrapper, _, policy_transforms = policy_cfg_obj.create(
        env=ftr_torchrl_env, weights_path=_cfg.policy_weights_path, device=device,
    )
    actor = actor_value_wrapper.get_policy_operator()
    env, vecnorm = make_transformed_env(ftr_torchrl_env, _cfg, policy_transforms)
    if _cfg.vecnorm_weights_path:
        vecnorm.load_state_dict(torch.load(_cfg.vecnorm_weights_path, map_location=device), strict=False)

    actor.eval()
    env.eval()

    all_results: list[dict[str, float]] = []
    for r in range(repeats):
        logger.info(f"  Rollout {r + 1}/{repeats} ...")
        try:
            results = _run_single_rollout(env, ftr_torchrl_env, actor, max_steps)
            all_results.append(results)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(f"  CUDA error during rollout {r + 1} — stopping eval for this trial: {e}")
                break
            logger.warning(f"  RuntimeError during rollout {r + 1}: {e}")

    ftr_gym_env.close()

    if not all_results:
        return None

    averaged = {k: sum(d[k] for d in all_results) / len(all_results) for k in all_results[0]}
    averaged["eval/num_rollouts"] = float(len(all_results))

    # Compute std of success_rate across rollouts
    if "eval/success_rate" in all_results[0] and len(all_results) > 1:
        sr_vals = [d["eval/success_rate"] for d in all_results]
        mean_sr = averaged["eval/success_rate"]
        std_sr = (sum((v - mean_sr) ** 2 for v in sr_vals) / (len(sr_vals) - 1)) ** 0.5
        averaged["eval/success_rate_std"] = std_sr
        averaged["eval/success_rate_min"] = min(sr_vals)
        averaged["eval/success_rate_max"] = max(sr_vals)

    return averaged


# ============================================================
# BLOCK 4 — Main
# ============================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.", flush=True)
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    try:
        # ---- Build search directories for fallback run discovery ----
        if args.runs_dir:
            search_dirs = [Path(args.runs_dir)]
        else:
            search_dirs = [ROOT / "logs", ROOT / "runs" / "ppo"]

        # ---- Get top trials ----
        trials = get_top_trials(args.study, args.top, db_url=args.db)
        if not trials:
            logger.error("No completed trials found in study.")
            os._exit(1)

        logger.info(f"Found {len(trials)} top trials from study '{args.study}':")
        for t in trials:
            logpath = t.user_attrs.get("logpath", "???")
            logger.info(f"  Trial #{t.number:>4d}  value={t.value:.4f}  logpath={logpath}")

        # ---- Evaluate each trial ----
        all_trial_results: list[dict] = []
        skipped = []

        for i, trial in enumerate(trials):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Evaluating trial #{trial.number} ({i + 1}/{len(trials)})  "
                        f"study_value={trial.value:.4f}")
            logger.info(f"{'=' * 70}")

            run_dir = find_run_dir(trial, search_dirs=search_dirs)
            if run_dir is None:
                logger.warning(f"  Cannot find run directory for trial #{trial.number} — skipping.")
                skipped.append(trial.number)
                continue

            logger.info(f"  Run dir: {run_dir}")

            try:
                metrics = eval_trial(
                    run_dir=run_dir,
                    repeats=args.repeats,
                    num_envs=args.num_envs,
                    max_steps_override=args.max_steps,
                    policy_filename=args.policy,
                    vecnorm_filename=args.vecnorm,
                )
            except Exception as e:
                logger.error(f"  Failed to evaluate trial #{trial.number}: {e}")
                traceback.print_exception(e)
                metrics = None

            if metrics is None:
                skipped.append(trial.number)
                continue

            row = {
                "trial": trial.number,
                "study_value": trial.value,
                "run_dir": str(run_dir),
                **metrics,
            }
            all_trial_results.append(row)

            # Print per-trial summary
            sr = metrics.get("eval/success_rate", 0.0)
            sr_std = metrics.get("eval/success_rate_std", 0.0)
            sr_min = metrics.get("eval/success_rate_min", 0.0)
            sr_max = metrics.get("eval/success_rate_max", 0.0)
            fr = metrics.get("eval/failure_rate", 0.0)
            er = metrics.get("eval/explosion_rate", 0.0)
            logger.info(
                f"  Trial #{trial.number}: success={sr:.3f} +/- {sr_std:.3f} "
                f"[{sr_min:.3f}, {sr_max:.3f}]  fail={fr:.3f}  explosion={er:.3f}"
            )

        # ---- Print comparison table ----
        print(f"\n{'=' * 90}")
        print(f"COMPARISON — {len(all_trial_results)} trials evaluated "
              f"({args.repeats} rollouts each)")
        print(f"{'=' * 90}")
        print(f"{'Trial':>7s}  {'Study Val':>10s}  {'Success':>8s}  {'Std':>6s}  "
              f"{'Min':>6s}  {'Max':>6s}  {'Fail':>6s}  {'Expl':>6s}  {'Reward':>8s}")
        print("-" * 90)

        # Sort by eval success rate (our ground truth)
        all_trial_results.sort(
            key=lambda r: r.get("eval/success_rate", 0.0), reverse=True
        )

        for r in all_trial_results:
            print(
                f"  #{r['trial']:>4d}  "
                f"{r['study_value']:>10.4f}  "
                f"{r.get('eval/success_rate', 0.0):>8.4f}  "
                f"{r.get('eval/success_rate_std', 0.0):>6.4f}  "
                f"{r.get('eval/success_rate_min', 0.0):>6.4f}  "
                f"{r.get('eval/success_rate_max', 0.0):>6.4f}  "
                f"{r.get('eval/failure_rate', 0.0):>6.4f}  "
                f"{r.get('eval/explosion_rate', 0.0):>6.4f}  "
                f"{r.get('eval/mean_step_reward', 0.0):>8.4f}"
            )

        if all_trial_results:
            best = all_trial_results[0]
            print(f"\nBEST TRIAL: #{best['trial']}  "
                  f"success={best.get('eval/success_rate', 0.0):.4f} "
                  f"+/- {best.get('eval/success_rate_std', 0.0):.4f}")
            print(f"  Run dir: {best['run_dir']}")

        if skipped:
            print(f"\nSkipped trials (no run dir or eval failed): {skipped}")

        # ---- Save CSV ----
        output_path = args.output
        if output_path is None:
            output_path = str(ROOT / f"reports/eval_top_{args.study}.csv")

        if all_trial_results:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(all_trial_results[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_trial_results)
            print(f"\nResults saved to: {output_path}")

    except Exception as _e:
        traceback.print_exception(_e)
        os._exit(1)

    simulation_app.close()
