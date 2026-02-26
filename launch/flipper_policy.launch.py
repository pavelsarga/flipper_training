"""
Launch file for the flipper policy controller node.

Usage:
    ros2 launch flipper_training flipper_policy.launch.py
    ros2 launch flipper_training flipper_policy.launch.py device:=cpu
    ros2 launch flipper_training flipper_policy.launch.py config_path:=/custom/config.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package paths
    pkg_share = get_package_share_directory("flipper_training")
    # Source directory (for development with symlink-install)
    pkg_src = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Default paths - try source dir first (symlink-install), fall back to share
    default_config = os.path.join(pkg_src, "sota_configs", "random_trunk_random_start_goal_sota.yaml")
    if not os.path.exists(default_config):
        default_config = os.path.join(pkg_share, "config", "random_trunk_random_start_goal_sota.yaml")

    default_policy = os.path.join(pkg_src, "modified_networks", "top_3_averaged", "policy.pth")
    default_vecnorm = os.path.join(pkg_src, "modified_networks", "top_3_averaged", "vecnorm.pth")

    # Declare launch arguments
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=default_config,
        description="Path to the training config YAML file",
    )

    policy_weights_path_arg = DeclareLaunchArgument(
        "policy_weights_path",
        default_value=default_policy,
        description="Path to the policy weights (.pth file)",
    )

    vecnorm_weights_path_arg = DeclareLaunchArgument(
        "vecnorm_weights_path",
        default_value=default_vecnorm,
        description="Path to the vecnorm weights (.pth file, optional)",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to run inference on (cpu or cuda:N)",
    )

    control_rate_arg = DeclareLaunchArgument(
        "control_rate",
        default_value="10.0",
        description="Control loop rate in Hz",
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
        default_value="1.0",
        description="Scale factor for flipper velocity commands",
    )

    # Node
    flipper_policy_node = Node(
        package="flipper_training",
        executable="flipper_policy_node.py",
        name="flipper_policy_node",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "policy_weights_path": LaunchConfiguration("policy_weights_path"),
                "vecnorm_weights_path": LaunchConfiguration("vecnorm_weights_path"),
                "device": LaunchConfiguration("device"),
                "control_rate": LaunchConfiguration("control_rate"),
                "heightmap_decay": LaunchConfiguration("heightmap_decay"),
                "heightmap_layer": LaunchConfiguration("heightmap_layer"),
                "flipper_velocity_scale": LaunchConfiguration("flipper_velocity_scale"),
            }
        ],
    )

    return LaunchDescription(
        [
            config_path_arg,
            policy_weights_path_arg,
            vecnorm_weights_path_arg,
            device_arg,
            control_rate_arg,
            heightmap_decay_arg,
            heightmap_layer_arg,
            flipper_velocity_scale_arg,
            flipper_policy_node,
        ]
    )
