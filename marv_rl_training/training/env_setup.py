"""Build the FTR-Benchmark gymnasium env from a training/eval config.

Every ``train_*.py`` and ``eval_*.py`` entry point needs the same ~80 lines before it can do
anything: check that Isaac Sim actually got a CUDA context, import the task registrations,
resolve the env-config class out of the gymnasium registry, copy the physics fields from the
config onto it, apply ``env_cfg_overrides``, and call ``gymnasium.make``. This module holds
that once.

Nothing is imported at module level that needs Isaac Sim to be running: ``gymnasium``,
``torch`` and ``ftr_envs.tasks`` are imported inside the functions, so this module is safe to
import from BLOCK 2 of an entry point.
"""

from typing import Any

__all__ = ["require_cuda", "import_ftr_tasks", "resolve_env_cfg_class", "build_ftr_gym_env"]


def require_cuda() -> None:
    """Abort the process if Isaac Sim failed to create a CUDA context.

    Importing ``ftr_envs.tasks`` triggers ``wp.init()`` (via the omni.isaac.lab.envs chain),
    which dies with an unhelpful RuntimeError when the context is dead. ``os._exit`` rather
    than ``sys.exit``/``raise``: Isaac Sim's atexit handlers deadlock on a broken context, so
    a clean exit would hang the apptainer process — and the SLURM slot — for minutes.
    """
    import os
    import torch

    if not torch.cuda.is_available():
        print(
            "FATAL: torch.cuda.is_available() returned False after AppLauncher init.\n"
            "Isaac Sim failed to create a CUDA context (check .err for 'CUDA error 46').\n"
            "This is usually a node-level GPU issue — try a different compute node.",
            flush=True,
        )
        os._exit(1)


def import_ftr_tasks() -> None:
    """Import the FTR task registrations. Must run after AppLauncher, before gymnasium.spec."""
    import os

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as e:  # noqa: BLE001 — nothing can be done without the registrations
        print(f"FATAL: failed to import ftr_envs.tasks: {e}", flush=True)
        os._exit(1)


def resolve_env_cfg_class(task: str) -> type:
    """Look the env-config class up in the gymnasium registry, so ``--task`` actually works.

    Falls back to ``CrossingEnvCfg`` for a task registered without an ``env_cfg_entry_point``.
    """
    import importlib

    import gymnasium

    entry = gymnasium.spec(task).kwargs.get("env_cfg_entry_point", "")
    if isinstance(entry, str) and ":" in entry:
        module_path, class_name = entry.rsplit(":", 1)
        return getattr(importlib.import_module(module_path), class_name)
    if isinstance(entry, type):
        return entry
    from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg

    return CrossingEnvCfg


def build_ftr_gym_env(
    cfg: Any,
    *,
    set_decimation: bool = True,
    physx_buffers: str = "config",
    log_raw_accel: bool | None = None,
    log_raw_accel_path: str | None = None,
):
    """Create the gymnasium env described by ``cfg`` (any of the Ftr*Config dataclasses).

    ``set_decimation`` — copy ``cfg.decimation`` onto the env config. The trainers do; the
    eval scripts deliberately leave the env's own default in place.

    ``physx_buffers`` — how ``gpu_found_lost_aggregate_pairs_capacity`` and friends are sized.
    FTR_SIM_CFG's defaults assume 4096 envs on a server GPU, which does not fit a laptop GPU
    running a handful of envs and leaves headroom unused at very large counts:

      ``"config"``  take the value from the config and nothing else (what the trainers do)
      ``"small"``   the config value, then shrink every GPU buffer at num_robots <= 64
      ``"auto"``    shrink at <= 64, grow the aggregate-pairs capacity above 512, and leave
                    the env default in between — the config value is not used at all

    ``log_raw_accel`` overrides ``cfg.log_raw_accel`` (the random-policy eval turns it on from
    a CLI flag); configs whose dataclass has no such field never enable it.
    ``log_raw_accel_path`` sets the destination now; the trainers leave it None and patch it
    later, once RunLogger has decided where the run directory is.
    """
    import gymnasium

    env_cfg = resolve_env_cfg_class(cfg.task)()
    env_cfg.scene.num_envs = cfg.num_robots
    env_cfg.terrain_name = cfg.terrain

    # --- Simulation timestep and decimation ---
    env_cfg.sim.dt = cfg.sim_dt
    if set_decimation:
        env_cfg.decimation = cfg.decimation

    # --- Rigid body properties ---
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = cfg.robot_angular_damping

    # --- Per-articulation solver iterations ---
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = cfg.solver_velocity_iterations

    # --- Scene-wide PhysX solver (matched to the per-articulation values) ---
    env_cfg.sim.physx.min_position_iteration_count = cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = cfg.bounce_threshold_velocity
    env_cfg.sim.physx.gpu_heap_capacity = cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = cfg.physx_gpu_max_num_partitions

    if physx_buffers in ("config", "small"):
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = cfg.physx_gpu_found_lost_aggregate_pairs_capacity
    if physx_buffers in ("small", "auto") and cfg.num_robots <= 64:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22
    elif physx_buffers == "auto" and cfg.num_robots > 512:
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 27

    # Arbitrary direct-attribute overrides — module_name, reward params, potential-reward
    # settings and so on. A plain setattr loop, so a key that is not a real field on the env
    # config is set and then ignored with no error at all; nested settings need an explicit
    # bridge in FtrEnv.__init__ (see the CLAUDE.md section on env_cfg_overrides).
    for k, v in (cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    enable_accel_log = getattr(cfg, "log_raw_accel", False) if log_raw_accel is None else log_raw_accel
    if enable_accel_log:
        env_cfg.log_raw_accel = True
        env_cfg.log_raw_accel_interval = cfg.log_raw_accel_interval
        if log_raw_accel_path is not None:
            env_cfg.log_raw_accel_path = log_raw_accel_path

    return gymnasium.make(cfg.task, cfg=env_cfg)
