# flipper_training — legacy framework branch

This branch (`legacy-framework`) freezes the **pre-`marv_rl_training`** generic
framework (`flipper_training` Python package: engine, environment, heightmaps,
observations, policies, PPO/MPPI/grad experiments) for consumers that still need it.

**It is not the actively developed code.** The canonical training framework is
`marv_rl_training` on `main` / `diffusion-policy` of this repository (consumed as
the `src/flipper_training` submodule of MARV_RL). The old `flipper_training/`
package was removed there in `f085863` ("Clean up. Left only MARV_RL relevant files").

## Provenance

- Base: `4a54e89` ("restructure and cleanup", 2026-06-17).
- On top: one commit carrying the patches accumulated in the
  `marv_flipper_control_research` repo, where this tree used to live as a
  vendored copy (FTR-compat observations/rewards, RL-baseline inference modules,
  D3QN/ICM/off-policy AC policies, robot config fixes).
- ROS packaging (`package.xml`, `CMakeLists.txt`) removed: this is a plain Python
  library, not a colcon package. That avoids a duplicate `flipper_training`
  package name next to the MARV_RL submodule. `launch/` and `ros2/` scripts are
  kept for reference only; the deployed ROS node is
  `ros2/flipper_policy_node.py` on `main` / `diffusion-policy`.

## Consumers

- `marv_flipper_control_research/third_party/flipper_training_legacy` (git
  submodule, branch `legacy-framework`). `scripts/run_flipper_policy_sim.sh` puts
  it on `PYTHONPATH` so MARV_RL's `flipper_policy_node.py` can serve native (non-FTR)
  PPO runs through `flipper_training.experiments.ppo.policy_inference_module`.

Robot description: `robots/marv.yaml` (resolved by `configs/robot_config.py` as
`<repo root>/robots`).
