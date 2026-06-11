"""
Launch file for the flipper policy controller node.

Usage:
    # Specify a run directory (config and weights auto-resolved):
    ros2 launch flipper_training flipper_policy.launch.py \
        run_dir:=/path/to/logs/train_ftr_10750834

    # With a specific checkpoint instead of policy_final.pth:
    ros2 launch flipper_training flipper_policy.launch.py \
        run_dir:=/path/to/logs/train_ftr_10750834 \
        policy_filename:=policy_step_13500416.pth

    # Or specify paths individually (run_dir takes precedence if both given):
    ros2 launch flipper_training flipper_policy.launch.py \
        config_path:=/custom/config.yaml \
        policy_weights_path:=/custom/policy.pth \
        vecnorm_weights_path:=/custom/vecnorm.pth \
        device:=cpu
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_paths(context):
    run_dir = LaunchConfiguration("run_dir").perform(context)

    if run_dir:
        policy_filename = LaunchConfiguration("policy_filename").perform(context)
        vecnorm_filename = LaunchConfiguration("vecnorm_filename").perform(context)
        weights_dir = os.path.join(run_dir, "weights")
        config = os.path.join(run_dir, "config.yaml")
        policy = os.path.join(weights_dir, policy_filename)
        vecnorm = os.path.join(weights_dir, vecnorm_filename)
    else:
        config = LaunchConfiguration("config_path").perform(context)
        policy = LaunchConfiguration("policy_weights_path").perform(context)
        vecnorm = LaunchConfiguration("vecnorm_weights_path").perform(context)

    node = Node(
        package="flipper_training",
        executable="flipper_policy_node.py",
        name="flipper_policy_node",
        output="screen",
        parameters=[
            {
                "config_path": config,
                "policy_weights_path": policy,
                "vecnorm_weights_path": vecnorm,
                "device": LaunchConfiguration("device").perform(context),
                "control_rate": float(LaunchConfiguration("control_rate").perform(context)),
                "heightmap_decay": float(LaunchConfiguration("heightmap_decay").perform(context)),
                "heightmap_layer": LaunchConfiguration("heightmap_layer").perform(context),
                "flipper_velocity_scale": float(LaunchConfiguration("flipper_velocity_scale").perform(context)),
                "track_velocity_scale": float(LaunchConfiguration("track_velocity_scale").perform(context)),
                "disable_turning": LaunchConfiguration("disable_turning").perform(context).lower() == "true",
            }
        ],
    )
    return [node]


def generate_launch_description():
    # --- run_dir shortcut (auto-resolves config + weights) ---
    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        default_value="/home/robot/workspaces/robot_rodeo_gym_ws/experiments/best_long/attempt_0",
        description="Path to a training run directory. If set, config.yaml and weights/policy_final.pth "
                    "are loaded automatically (overrides config_path/policy_weights_path/vecnorm_weights_path).",
    )
    policy_filename_arg = DeclareLaunchArgument(
        "policy_filename",
        default_value="policy_final.pth",
        description="Policy checkpoint filename inside <run_dir>/weights/ (used only when run_dir is set).",
    )
    vecnorm_filename_arg = DeclareLaunchArgument(
        "vecnorm_filename",
        default_value="vecnorm_final.pth",
        description="VecNorm checkpoint filename inside <run_dir>/weights/ (used only when run_dir is set).",
    )

    # --- individual path args (used when run_dir is not set) ---
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="",
        description="Path to the training config YAML file (ignored when run_dir is set).",
    )
    policy_weights_path_arg = DeclareLaunchArgument(
        "policy_weights_path",
        default_value="",
        description="Path to the policy weights (.pth file, ignored when run_dir is set).",
    )
    vecnorm_weights_path_arg = DeclareLaunchArgument(
        "vecnorm_weights_path",
        default_value="",
        description="Path to the vecnorm weights (.pth file, optional, ignored when run_dir is set).",
    )

    # --- node config args ---
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="Device to run inference on (cpu or cuda:N)",
    )
    control_rate_arg = DeclareLaunchArgument(
        "control_rate",
        default_value="40.0",
        description="Control loop rate in Hz. Must match training: sim_dt × decimation. "
                    "FTR/CrossingEnv with sim_dt=0.005, decimation=5 → 40 Hz. "
                    "Native flipper_training with sim_dt=1/400, engine_iters_per_env_step=4 → 100 Hz.",
    )
    heightmap_decay_arg = DeclareLaunchArgument(
        "heightmap_decay",
        default_value="0.95",
        description="Temporal decay factor for heightmap smoothing (0-1)",
    )
    heightmap_layer_arg = DeclareLaunchArgument(
        "heightmap_layer",
        default_value="elevation",
        description="Name of the elevation layer in GridMap",
    )
    flipper_velocity_scale_arg = DeclareLaunchArgument(
        "flipper_velocity_scale",
        default_value="1",
        description="Scale factor for flipper velocity commands",
    )
    track_velocity_scale_arg = DeclareLaunchArgument(
        "track_velocity_scale",
        default_value="1",
        description="Scale factor for FTR track velocity commands (linear and angular). "
                    "Use <1.0 if robot moves too fast compared to Isaac Sim training.",
    )
    disable_turning_arg = DeclareLaunchArgument(
        "disable_turning",
        default_value="true",
        description="If true, forces angular velocity to 0 on every step (straight-line only).",
    )

    return LaunchDescription(
        [
            run_dir_arg,
            policy_filename_arg,
            vecnorm_filename_arg,
            config_path_arg,
            policy_weights_path_arg,
            vecnorm_weights_path_arg,
            device_arg,
            control_rate_arg,
            heightmap_decay_arg,
            heightmap_layer_arg,
            flipper_velocity_scale_arg,
            track_velocity_scale_arg,
            disable_turning_arg,
            OpaqueFunction(function=_resolve_paths),
        ]
    )
