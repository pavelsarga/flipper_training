"""
Launch file for the flipper policy controller node.

Usage:
    ros2 launch flipper_policy.launch.py config_path:=/path/to/config.yaml policy_weights_path:=/path/to/policy.pth

Or run directly:
    ros2 run flipper_training flipper_policy_node --ros-args \
        -p config_path:=/path/to/config.yaml \
        -p policy_weights_path:=/path/to/policy.pth \
        -p vecnorm_weights_path:=/path/to/vecnorm.pth
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        description="Path to the training config YAML file",
    )

    policy_weights_path_arg = DeclareLaunchArgument(
        "policy_weights_path",
        description="Path to the policy weights (.pth file)",
    )

    vecnorm_weights_path_arg = DeclareLaunchArgument(
        "vecnorm_weights_path",
        default_value="",
        description="Path to the vecnorm weights (.pth file, optional)",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
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
        executable="flipper_policy_node",
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
