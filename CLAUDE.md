# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Flipper Training is a differentiable physics-based reinforcement learning framework for training tracked rover locomotion with flipper actuators. It provides a PyTorch-native physics engine, TorchRL-based environments, and multiple training paradigms (PPO, gradient-based, MPPI).

## Build and Development Commands

```bash
# Install dependencies (using UV package manager)
uv sync

# Run code formatting and linting (Ruff via pre-commit)
pre-commit run --all-files

# Run tests
pytest tests/

# Run a specific test
pytest tests/utils/test_environment.py::test_heightmap_gradients
```

## Training and Evaluation

```bash
# Train with PPO using a config file
python -m flipper_training.experiments.ppo.train --local test_configs/deterministic_flats_debug.yaml

# Evaluate a trained model from local run directory
python -m flipper_training.experiments.ppo.eval --local runs/ppo/<run_name>

# Evaluate from W&B run
python -m flipper_training.experiments.ppo.eval --wandb <wandb_run_name>

# Override config values via CLI
python -m flipper_training.experiments.ppo.train --local config.yaml num_robots=64 device=cuda:0
```

## Deploying Policy for Gazebo/ROS2

Key files:
- **ROS2 node**: `ros2/flipper_policy_node.py`
- **Launch file**: `ros2/flipper_policy.launch.py`
- **Goal sender utility**: `ros2/send_goal.py`
- **Jupyter notebook**: `notebooks/ppo_policy_inference.ipynb`
- **Policy inference module**: `flipper_training/experiments/ppo/policy_inference_module.py`
- **Pretrained weights**: `modified_networks/top_3_averaged/` (policy.pth, vecnorm.pth)

### Running the ROS2 Node

```bash
# Using launch file
ros2 launch ros2/flipper_policy.launch.py \
    config_path:=/path/to/config.yaml \
    policy_weights_path:=/path/to/policy.pth \
    vecnorm_weights_path:=/path/to/vecnorm.pth \
    device:=cuda:0

# Direct execution
python ros2/flipper_policy_node.py --ros-args \
    -p config_path:=/path/to/config.yaml \
    -p policy_weights_path:=/path/to/policy.pth

# Send goal to the policy node
python ros2/send_goal.py <x> <y>   # World frame coordinates
```

### ROS2 Node Topics

**Subscribed:**
- `/ground_truth_odom` (nav_msgs/Odometry) - Robot pose and velocity
- `/joint_state` (sensor_msgs/JointState) - Flipper joint angles
- `/elevation_map` (grid_map_msgs/GridMap) - Elevation map from mapping
- `/goal_pose` (geometry_msgs/PoseStamped) - Goal position in world frame

**Published:**
- `/cmd_vel` (geometry_msgs/Twist) - Track velocity commands
- `/flippers_cmd_pos/{front_left,front_right,rear_left,rear_right}` (std_msgs/Float64) - Flipper position commands

### Node Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_path` | required | Path to training config YAML |
| `policy_weights_path` | required | Path to policy weights (.pth) |
| `vecnorm_weights_path` | "" | Path to vecnorm weights (optional) |
| `device` | "cpu" | Inference device (cpu/cuda:N) |
| `control_rate` | 10.0 | Control loop rate (Hz) |
| `heightmap_decay` | 0.95 | Temporal decay for heightmap smoothing |
| `heightmap_layer` | "elevation" | GridMap layer name |
| `flipper_velocity_scale` | 1.0 | Scale factor for flipper velocities |

### Flipper Angle Conventions

- **Angle 0**: Flipper is horizontal
- **Negative rotational velocity**: Front flippers rotate **up**, rear flippers rotate **down**

| Flipper | Fully Up | Fully Down |
|---------|----------|------------|
| Front   | -π/2     | +π/2       |
| Rear    | +π/2     | -π/2       |

Clamp input angles from ROS to this interval.

### Policy Input State Vector

1. **Goal vector**: Direction to goal in base_link frame (meters)
2. **Linear velocity**: In m/s
3. **Angular velocity** (twist): In rad/s
4. **Flipper angles**: In radians, using the convention above
5. **Quaternion**: ROS convention (x, y, z, w) relative to gravity vector (used internally to extract roll/pitch)

### Heightmap Input

- **Resolution**: 64×64 (higher resolution accepted - resampled internally)
- **Physical extent**: [1, 1] (top-left) to [-1, -1] (bottom-right)
- **Orientation**: As if standing behind the robot looking forward, then leaning over it. Objects **in front** of the robot should be in the **upper part** of the heightmap.

Debug with `plt.imshow(heightmap)` to verify correct front/back orientation.

### Policy Output Actions

8 values total:
1. **4 track velocities**: In m/s (+1 = forward, -1 = backward)
2. **4 flipper rotational velocities**: Sign convention as described above

### Operational Recommendation

Set a slow temporal decay on the heightmap in production to prevent accumulation/smearing over time.

## Architecture

### Core Components

**Physics Engine** (`flipper_training/engine/`)
- `engine.py`: Main differentiable physics engine (PyTorch tensors, ~14K lines)
- `engine_warp.py`: Alternative NVIDIA Warp GPU-accelerated implementation
- `engine_state.py`: `PhysicsState` and `PhysicsStateDer` dataclasses

**Environment** (`flipper_training/environment/`)
- `env.py`: TorchRL `EnvBase` implementation with batch-locked vectorized simulation
- `transforms.py`: Observation/reward transforms compatible with TorchRL

**Factory Pattern for Composable Components**
- `observations/__init__.py` → `ObservationFactory` interface
- `rl_rewards/__init__.py` → `RewardFactory` interface
- `rl_objectives/__init__.py` → `ObjectiveFactory` interface
- `heightmaps/__init__.py` → `HeightmapGenerator` interface

**Policy Networks** (`flipper_training/policies/`)
- MLP, GRU, LSTM architectures with GSDE distribution variants
- All implement `PolicyConfig` with `create()` method returning actor-value wrapper

### Configuration System

YAML configs use OmegaConf with custom resolvers defined in `flipper_training/__init__.py`:

```yaml
# Arithmetic
total_frames: ${mul:5242880,6}          # 31457280
batch_ratio: ${div:128,64}              # 2.0

# Class instantiation (dynamically loads classes)
heightmap_gen: ${cls:flipper_training.heightmaps.trunks.TrunkHeightmapGenerator}
optimizer: ${cls:torch.optim.AdamW}

# PyTorch types
training_dtype: ${dtype:float32}        # torch.float32
start_pos: ${tensor:[-1.5, 0.0, 0.2]}   # torch.tensor(...)
```

### Data Flow

```
YAML Config
    ↓
PPOExperimentConfig (dataclass)
    ↓
┌───────────────────────────────────────┐
│ Environment (Env)                      │
│  ├─ ObservationFactory → observations │
│  ├─ RewardFactory → reward signal     │
│  ├─ ObjectiveFactory → task logic     │
│  ├─ TerrainConfig → heightmap grids   │
│  └─ PhysicsEngine → differentiable sim│
└───────────────────────────────────────┘
    ↓
TorchRL Collector → Policy → Training Loop
```

### Key Dataclasses with `.to(device)` Pattern

Configs in `flipper_training/configs/` inherit from `BaseConfig` which provides automatic tensor device movement:

```python
terrain_config.to(device)  # Moves all tensor fields to device
physics_config.to(device)
robot_model.to(device)
```

## Directory Structure

```
flipper_training/
├── configs/           # Dataclass configs (engine, robot, terrain)
├── engine/            # Differentiable physics simulation
├── environment/       # TorchRL Env implementation
├── experiments/       # Training pipelines
│   ├── ppo/          # PPO training/eval (main entry points)
│   ├── grad/         # Gradient-based control
│   └── mppi/         # Model Predictive Path Integral
├── heightmaps/        # Procedural terrain generators
├── observations/      # Observation vector builders
├── policies/          # Neural network architectures
├── rl_objectives/     # Task definitions (barrier, stairs, trunk crossing)
├── rl_rewards/        # Reward function implementations
├── utils/             # Geometry, mesh processing, logging
└── vis/               # SimView visualization integration

test_configs/          # Training config examples
sota_configs/          # State-of-the-art model configs
robots/                # Robot YAML parameter files
meshes/                # Robot mesh files (STL/OBJ)
modified_networks/     # Pretrained weights (top_3_averaged/, transferred_with_preference/)
ros2/                  # ROS2 deployment (flipper_policy_node.py, send_goal.py, launch file)
notebooks/             # Jupyter notebooks (including policy inference demo)
```

## Code Style

- Line length: 150 characters
- Formatter/Linter: Ruff with rules E, F, Q, B, S
- Python version: 3.12

## Key Patterns

**Vectorized Batch Simulation**: All physics and environment code operates on batched tensors `[num_robots, ...]` for GPU parallelism (typically 128-256 robots).

**Engine Compilation**: The physics engine supports TorchDynamo compilation with Triton CUDA graphs for performance:
```yaml
engine_compile_opts:
  max-autotune: true
  triton.cudagraphs: true
```

**Curriculum Learning**: Objectives support `state_dict()`/`load_state_dict()` for saving/restoring curriculum state.
