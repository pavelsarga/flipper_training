# marv_rl_training

Training, evaluation and ROS2 deployment of flipper-control policies for the MARV tracked
rover. The simulator is the sibling `FTR-Benchmark` submodule (Isaac Sim / IsaacLab); this
package wraps its env in TorchRL and holds the trainers, evaluators and the deployment nodes.
The workspace root's README documents how to launch things; this file documents the package.

This is a fork of David Korčák's `flipper_training`. The differentiable physics engine, the
procedural heightmap generators and the MPPI/grad experiments it was built around are gone —
everything here runs against Isaac Sim through `FtrTorchRLEnv`.

## Layout

```
marv_rl_training/
  __init__.py            OmegaConf resolvers (below), ROOT / PACKAGE_ROOT
  environment/
    ftr_env_adapter.py   FtrTorchRLEnv: TorchRL EnvBase over the Isaac gymnasium env. Picks
                         the observation class from env_cfg_overrides.module_name, owns the
                         termination / reward / state stats the trainers and evals drain
    chunked_env.py       ActionChunkEnv for the receding-horizon (diffusion) trainer
    transforms.py        RawRewardSaveTransform
  observations/          Observation / ObservationEncoder base classes and HeightmapEncoder.
                         The concrete observations live in FTR-Benchmark/rl_modules; they are
                         populated directly by FtrTorchRLEnv._step / _reset
  policies/              PolicyConfig base, MLP actor-critic, random baseline,
                         heuristic heightmap policy, diffusion policy + DPPO
  utils/
    logutils.py          RunLogger (run dir, CSV/W&B/TensorBoard, SLURM attempt_N layout,
                         candidate_weight_dirs for resume), LocalRunReader / WandbRunReader
    cfg_schedulers.py    linear schedulers that write into a config field (step penalty, epsilon)
    torch_utils.py       seed_all, set_device
  training/
    cli.py               launch_isaac_app + the shared argument parsers. Import first: it is
                         the one thing that may run before AppLauncher
    env_setup.py         require_cuda / import_ftr_tasks / build_ftr_gym_env
    entrypoint.py        resolve_train_config (--play, SLURM respawn) and run_trainer
    common.py            make_transformed_env (StepCounter, VecNorm, extra transforms)
    trainer_common.py    eval aggregation and fatal-error test shared by the trainers
    eval_common.py       rollout loops, result printer, exit_flushed
    eval_data.py         run_tracked_rollout, per-env / per-spot aggregation, CSV writers
    env_type_registry.py terrain layout (rows, depth columns) from gen_config
    terrain_assets.py    write_terrain_manifest
    train_ftr.py         PPO (FtrPPOConfig / FtrPPOTrainer) — marv_rl, hfc, mitriakov
    train_d3qn.py        AT-D3QN (FtrD3QNConfig / FtrD3QNTrainer)
    train_icmd3qn.py     ICM-D3QN — subclass of the above through its aux-module hooks
    train_sac.py         C-TRAC: asymmetric SAC + C-VAE (FtrSACConfig / FtrSACTrainer)
    train_creps.py       CREPS (FtrCREPSConfig / FtrCREPSTrainer)
    train_diffusion.py   receding-horizon diffusion policy
    train_hfcil.py       supervised pretraining for the HFC imitation variant
    eval_ftr.py, eval_d3qn.py, eval_sac.py, eval_creps.py, eval_diffusion.py
    eval_ftr_rand.py     random-policy baseline
    optuna_train_ftr.py, optuna_eval_rand.py, eval_optuna_top.py, recover_optuna_trials.py
    collect_ctrac_dataset.py / pretrain_ctrac_cvae.py     C-TRAC Stage I
    collect_chunk_dataset.py / pretrain_diffusion_bc.py   diffusion BC pretraining
    *_policy_inference_module.py   config + weights -> callable policy, for the ROS nodes
    replay_buffer_io.py  partial replay-buffer persistence across SLURM respawns
    test_*.py            shape / env unit tests (pytest, need the isaaclab env)
ros2/                    deployment nodes (below)
launch/                  ros2 launch file for flipper_policy_node
```

## Entry-point structure

Every `train_*.py` / `eval_*.py` has the same shape, and the order matters:

```python
# BLOCK 1 — nothing that imports omni.* may run before this
from marv_rl_training.training.cli import launch_isaac_app, train_arg_parser
if __name__ == "__main__":
    parser = train_arg_parser("...", play=True)
    args, unknown_args, simulation_app = launch_isaac_app(parser)

# BLOCK 2 — everything else; Isaac Sim is running now
...

if __name__ == "__main__":
    raw_cfg = resolve_train_config(args, unknown_args)
    require_cuda(); import_ftr_tasks()
    ftr_gym_env = build_ftr_gym_env(FtrXConfig(**raw_cfg))
    run_trainer(FtrXTrainer, raw_cfg, ftr_gym_env)
    exit_flushed()
```

- The `if __name__` guard around the launcher is load-bearing: the eval scripts import their
  config dataclass from the trainer module, and an unguarded launcher would start a second
  Isaac Sim app on import.
- `unknown_args` is the OmegaConf dotlist merged on top of the config. `launch_isaac_app`
  strips leftover `--flag value` pairs first — AppLauncher reads some of its own flags
  straight out of `sys.argv` without removing them, and they crash `OmegaConf.from_dotlist`.
- Exit through `exit_flushed()` / `os._exit`, never `sys.exit`: Isaac Sim's shutdown
  re-initialises GPU foundation and regularly deadlocks, which holds a SLURM slot for hours.
  `exit_flushed` flushes stdout first, because a redirected stdout is block-buffered and
  `os._exit` alone discards the whole results summary.
- Exit code 75 means "transient, respawn me" to `slurm/lib/respawn_common.sh`; it is what
  the trainers use for a dead CUDA context or a W&B transport failure. Anything else is a
  real failure and exits 1.

The config dataclasses are not interchangeable — `FtrPPOConfig` rejects
`replay_buffer_capacity`, `FtrD3QNConfig` rejects `icm_opts`, and so on — so the trainer and
evaluator must match the config. `scripts/lib/config_detect.sh` encodes the mapping (from
`env_cfg_overrides.module_name`, except that a config with top-level
`prediction_horizon` + `execution_horizon` is a diffusion run whatever its module_name says).

## Resume across SLURM respawns

`RunLogger` writes `logs/<job_name>_<job_id>/attempt_N/` under SLURM. On a respawn the
trainer searches `RunLogger.candidate_weight_dirs()` (this attempt, then earlier ones) for
`policy_crash.pth` / `vecnorm_crash.pth`, else the latest `policy_step_<frames>.pth` pair,
then restores `training_state.pth` — optimizer, LR scheduler, the config schedulers and the
frame counter — from `_restore_training_state()` at the *end* of `__init__`, once everything
it writes into exists. Consequence: `collected_frames` and the `policy_step_*` names are
cumulative across attempts on runs that resumed.

`FtrD3QNTrainer` extends this through `_build_aux_modules` / `_build_aux_optimizers`: an aux
module is saved as `<name>_{crash,step_N,final}.pth` beside the policy and its optimizer goes
into `training_state.pth`. That is how ICM-D3QN's curiosity module rides along.

`FtrSACTrainer` is different: SAC's `loss_module.state_dict()` is the only state that carries
the SoftUpdate targets and `log_alpha`, so that is what it checkpoints, and it persists the
replay buffer as well (memmapped once per job, shared by all attempts). See the workspace
README's C-TRAC section for why each of those was necessary.

## Config resolvers

Registered in `marv_rl_training/__init__.py`; importing the package is what makes them
available, which is why every entry point has `import marv_rl_training`:

```yaml
total_frames: ${mul:5242880,6}                          # add / mul / div / intdiv / pow
optimizer:    ${cls:torch.optim.AdamW}                  # dotted path -> class
training_dtype: ${dtype:float32}                        # torch.float32
start_pos:    ${tensor:[-1.5, 0.0, 0.2]}                # torch.tensor
```

## Observation and encoder conventions

`env_cfg_overrides.module_name` selects both the FTR-Benchmark reward module *and* the
observation class `FtrTorchRLEnv` builds (`ftr_env_adapter.py`). Observations declare
`supports_vecnorm`; `make_transformed_env` normalises exactly those keys.

Checkpoints from the original flipper_training have encoder weights under
`encoders.LocalStateVector.*` / `encoders.Heightmap.*`; this package's single
`MarvRLFlatObservation` + `FtrFlipperStyleEncoder` puts them under
`encoders.MarvRLFlatObservation.{state_encoder,cnn}.*`. `eval_common.remap_native_to_ftr_weights`
does the rename, and `mlp_policy.py` also maps the intermediate `encoders.FtrFlatObservation.*`
spelling.

## ROS2 deployment

`ros2/flipper_policy_node.py` runs a trained `marv_rl` / `hfc` policy on the robot;
`ros2/mitriakov_policy_node.py` runs the Mitriakov step-edge baseline. `ros2/send_goal.py`
publishes a goal; `launch/flipper_policy.launch.py` wires the first node up from a run
directory (`run_dir:=...`, optional `policy_filename:=`).

```bash
ros2 launch flipper_training flipper_policy.launch.py run_dir:=/path/to/logs/train_ftr_<id> device:=cuda:0
python ros2/send_goal.py <x> <y>          # world frame
```

Subscribed: `/ground_truth_odom` (Odometry), `/imu/data` (Imu), `/joint_state` (JointState),
`/elevation_map` (GridMap), `/goal_pose` and `/goal_reset` (PoseStamped), `/flipper_override`
(Bool). Published: `/cmd_vel` (Twist), `/flippers_cmd_{vel,pos,pos_rel}/{front,rear}_{left,right}`
(Float64), and the `/policy_*` debug topics (heightmap cloud/grid/image, action array, RViz
markers, the flipper-command HUD image).

Node parameters: `config_path`, `policy_weights_path`, `vecnorm_weights_path` (required for a
run; the launch file derives them from `run_dir`), `device`, `control_rate` (10 Hz),
`heightmap_decay` (0.95 — keep a slow decay in production so the map does not smear),
`heightmap_layer`, `flipper_velocity_scale`, `track_velocity_scale`, `publish_cmd_vel`,
`disable_turning`, `auto_goal_on_release` / `auto_goal_ahead_m`.

### Flipper angle conventions

- Angle 0 is horizontal. Negative rotational velocity moves the **front** flippers up and the
  **rear** flippers down.

| flipper | fully up | fully down |
|---|---|---|
| front | −π/2 | +π/2 |
| rear | +π/2 | −π/2 |

Clamp angles coming from ROS to those intervals. MARV's enforced limits are asymmetric
(front −90/+80°, rear −80/+90° in the D3QN configs), which is why the D3QN observations
normalise by the limits actually enforced rather than by a symmetric constant.

### Policy inputs and outputs

State vector: goal direction in `base_link` (m), linear velocity (m/s), angular velocity
(rad/s), flipper angles (rad, convention above), orientation quaternion (x, y, z, w — roll and
pitch are extracted from it).

Heightmap: 64×64 (a finer map is resampled), extent `[1, 1]` (top-left) to `[−1, −1]`
(bottom-right), oriented as if standing behind the robot and leaning over it — terrain in
front of the robot is in the upper rows. Check with `plt.imshow(heightmap)`.

Action: 4 track velocities (m/s, +1 forward) followed by 4 flipper rotational velocities.

## Notes

- Nothing configures a handler for the `ftr_envs.*` loggers on the eval path; only the
  `marv_rl_training` loggers reach stdout there. Use `print(..., flush=True)` in FTR-Benchmark
  code for anything that must be seen during eval.
- The unit tests need the isaaclab conda env:
  `apptainer exec containers/isaaclab_optuna.sif conda run -n isaaclab python -m pytest src/flipper_training/marv_rl_training/training/`.
- Line length 150, ruff rules E/F/Q/B (`pyproject.toml`).
