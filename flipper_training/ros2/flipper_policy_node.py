#!/usr/bin/env python3
"""
ROS2 node for deploying the trained flipper control policy on the MARV robot.

Subscribes to robot state and elevation map, publishes track velocities and flipper commands.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, PointCloud2, PointField, Imu
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64, MultiArrayDimension, MultiArrayLayout, Float32MultiArray
from grid_map_msgs.msg import GridMap
import struct

import torch
from scipy.spatial.transform import Rotation


class FlipperPolicyNode(Node):
    """ROS2 node that runs the trained flipper control policy."""

    # Flipper joint name mapping (order expected by policy)
    FLIPPER_NAMES = ["front_left_flipper_j", "front_right_flipper_j", "rear_left_flipper_j", "rear_right_flipper_j"]

    # Track width for differential drive conversion (meters)
    TRACK_WIDTH = 0.36  # From robot config rover_bodyWidth

    def __init__(self):
        super().__init__("flipper_policy_node")

        # Declare parameters
        self.declare_parameter("config_path", "")
        self.declare_parameter("policy_weights_path", "")
        self.declare_parameter("vecnorm_weights_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("control_rate", 10.0)  # Hz
        self.declare_parameter("heightmap_decay", 0.95)  # Temporal decay for heightmap
        self.declare_parameter("heightmap_layer", "elevation")  # Layer name in GridMap
        self.declare_parameter("flipper_velocity_scale", 1.0)  # Scale factor for flipper velocities
        self.declare_parameter("publish_debug_cloud", True)  # Publish heightmap as point cloud for debugging

        # Get parameters
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        policy_weights_path = self.get_parameter("policy_weights_path").get_parameter_value().string_value
        vecnorm_weights_path = self.get_parameter("vecnorm_weights_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        self.heightmap_decay = self.get_parameter("heightmap_decay").get_parameter_value().double_value
        self.heightmap_layer = self.get_parameter("heightmap_layer").get_parameter_value().string_value
        self.flipper_velocity_scale = self.get_parameter("flipper_velocity_scale").get_parameter_value().double_value
        self.publish_debug_cloud = self.get_parameter("publish_debug_cloud").get_parameter_value().bool_value

        if not config_path or not policy_weights_path:
            self.get_logger().error("config_path and policy_weights_path parameters are required!")
            raise ValueError("Missing required parameters")

        # Auto-detect FTR vs native policy from config contents
        self._is_ftr = self._detect_ftr_config(config_path)
        self.get_logger().info(
            f"Loading {'FTR' if self._is_ftr else 'native'} policy from {config_path}"
        )

        if self._is_ftr:
            from flipper_training.experiments.ppo.ftr_policy_inference_module import FtrPolicyInferenceModule
            self.policy = FtrPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                vecnorm_weights_path=vecnorm_weights_path if vecnorm_weights_path else None,
                device=device,
            )
        else:
            from flipper_training.experiments.ppo.policy_inference_module import PPOPolicyInferenceModule
            self.policy = PPOPolicyInferenceModule(
                train_config_path=config_path,
                policy_weights_path=policy_weights_path,
                vecnorm_weights_path=vecnorm_weights_path if vecnorm_weights_path else None,
                device=device,
            )
        self.get_logger().info("Policy loaded successfully")

        # State storage
        self.current_odom: Odometry | None = None
        self.current_imu: Imu | None = None
        self.current_joint_state: JointState | None = None
        self.current_goal: PoseStamped | None = None
        self.current_heightmap: np.ndarray | None = None
        self.heightmap_extent: list[float] | None = None
        self.heightmap_position: tuple[float, float, float] | None = None  # Map center position
        self.accumulated_heightmap: np.ndarray | None = None  # For temporal smoothing

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscribers
        # Note: ground_truth_odom uses RELIABLE QoS, so we must match it
        self.odom_sub = self.create_subscription(Odometry, "/ground_truth_odom", self.odom_callback, reliable_qos)
        self.imu_sub = self.create_subscription(Imu, "/imu/data", self.imu_callback, sensor_qos)
        self.joint_state_sub = self.create_subscription(JointState, "/joint_state", self.joint_state_callback, sensor_qos)
        # Goal uses VOLATILE to accept messages from any publisher (RViz, ros2 topic pub, etc.)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.elevation_map_sub = self.create_subscription(GridMap, "/elevation_map", self.elevation_map_callback, reliable_qos)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.flipper_pubs = {
            "front_left": self.create_publisher(Float64, "/flippers_cmd_vel/front_left", 10),
            "front_right": self.create_publisher(Float64, "/flippers_cmd_vel/front_right", 10),
            "rear_left": self.create_publisher(Float64, "/flippers_cmd_vel/rear_left", 10),
            "rear_right": self.create_publisher(Float64, "/flippers_cmd_vel/rear_right", 10),
        }
        # FTR policies output incremental flipper velocities — publish on velocity topics
        # (same as above, but kept as a named alias for clarity)
        self.flipper_vel_pubs = self.flipper_pubs

        # Debug visualization publishers
        self.heightmap_cloud_pub = self.create_publisher(PointCloud2, "/policy_heightmap_debug", 10)
        self.heightmap_gridmap_pub = self.create_publisher(GridMap, "/policy_heightmap", 10)

        # Current flipper positions (for integrating velocity commands)
        self.flipper_positions = {
            "front_left": 0.0,
            "front_right": 0.0,
            "rear_left": 0.0,
            "rear_right": 0.0,
        }

        # Control timer
        self.dt = 1.0 / control_rate
        self.control_timer = self.create_timer(self.dt, self.control_callback)

        self.get_logger().info(f"Flipper policy node started at {control_rate} Hz")

    def odom_callback(self, msg: Odometry):
        """Store latest odometry message."""
        self.current_odom = msg

    def imu_callback(self, msg: Imu):
        """Store latest IMU message."""
        self.current_imu = msg

    def joint_state_callback(self, msg: JointState):
        """Store latest joint state and update flipper positions."""
        self.current_joint_state = msg
        # Update flipper positions from joint state
        for i, name in enumerate(msg.name):
            if "front_left_flipper" in name:
                self.flipper_positions["front_left"] = msg.position[i]
            elif "front_right_flipper" in name:
                self.flipper_positions["front_right"] = msg.position[i]
            elif "rear_left_flipper" in name:
                self.flipper_positions["rear_left"] = msg.position[i]
            elif "rear_right_flipper" in name:
                self.flipper_positions["rear_right"] = msg.position[i]

    def goal_callback(self, msg: PoseStamped):
        """Store latest goal pose."""
        self.current_goal = msg
        self.get_logger().info(
            f"Goal received: frame={msg.header.frame_id}, "
            f"pos=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def elevation_map_callback(self, msg: GridMap):
        """Process elevation map into heightmap format expected by policy."""
        try:
            # Find the elevation layer
            if self.heightmap_layer not in msg.layers:
                self.get_logger().warn(f"Layer '{self.heightmap_layer}' not found in GridMap. Available: {msg.layers}")
                return

            layer_idx = msg.layers.index(self.heightmap_layer)
            data = np.array(msg.data[layer_idx].data, dtype=np.float32)

            # GridMap stores data in column-major (Fortran) order
            # Layout dim[0] is columns (outer), dim[1] is rows (inner)
            cols = msg.data[layer_idx].layout.dim[0].size
            rows = msg.data[layer_idx].layout.dim[1].size

            # Reshape with Fortran order and handle circular buffer
            heightmap = data.reshape((cols, rows), order='F')

            # Apply circular buffer start indices if present
            if msg.outer_start_index != 0 or msg.inner_start_index != 0:
                heightmap = np.roll(heightmap, -msg.outer_start_index, axis=0)
                heightmap = np.roll(heightmap, -msg.inner_start_index, axis=1)

            # Transpose to get [rows, cols] with X along rows, Y along cols
            heightmap = heightmap.T

            # Handle NaN values (unknown areas)
            heightmap = np.nan_to_num(heightmap, nan=0.0)

            # Calculate extent from GridMap info
            # GridMap uses center position and length
            length_x = msg.info.length_x
            length_y = msg.info.length_y
            # Extent: [xmax, ymax, xmin, ymin] in robot's local frame
            self.heightmap_extent = [length_x / 2, length_y / 2, -length_x / 2, -length_y / 2]
            # Store map center position (for debugging)
            self.heightmap_position = (
                msg.info.pose.position.x,
                msg.info.pose.position.y,
                msg.info.pose.position.z,
            )
            self.get_logger().info(
                f"Elevation map: frame={msg.header.frame_id}, pos=({msg.info.pose.position.x:.2f}, {msg.info.pose.position.y:.2f})",
                throttle_duration_sec=5.0,
            )

            # Apply temporal decay for smoothing
            if self.accumulated_heightmap is None or self.accumulated_heightmap.shape != heightmap.shape:
                self.accumulated_heightmap = heightmap.copy()
            else:
                self.accumulated_heightmap = self.heightmap_decay * self.accumulated_heightmap + (1 - self.heightmap_decay) * heightmap

            self.current_heightmap = self.accumulated_heightmap.copy()

            # Publish debug visualization
            if self.publish_debug_cloud:
                self.publish_heightmap_pointcloud()

            # Save debug image periodically (every 100 updates)
            if not hasattr(self, '_heightmap_save_counter'):
                self._heightmap_save_counter = 0
            self._heightmap_save_counter += 1
            if self._heightmap_save_counter % 100 == 1:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 8))
                    plt.imshow(heightmap, cmap='terrain', origin='upper')
                    plt.colorbar(label='Height (m)')
                    plt.title('Heightmap (row 0 = front of robot?)')
                    plt.xlabel('Y axis (left/right)')
                    plt.ylabel('X axis (front=top, back=bottom)')
                    plt.savefig('/tmp/heightmap_debug.png', dpi=100)
                    plt.close()
                    self.get_logger().info('Saved debug heightmap to /tmp/heightmap_debug.png')
                except Exception as e:
                    self.get_logger().warn(f'Could not save debug heightmap: {e}')

        except Exception as e:
            self.get_logger().error(f"Error processing elevation map: {e}")

    def publish_heightmap_pointcloud(self):
        """Publish heightmap as colored point cloud for RViz debugging."""
        if self.current_heightmap is None or self.heightmap_extent is None:
            return

        heightmap = self.current_heightmap
        extent = self.heightmap_extent  # [xmax, ymax, xmin, ymin]
        rows, cols = heightmap.shape

        # Generate X, Y coordinates
        x_coords = np.linspace(extent[0], extent[2], rows)  # xmax to xmin (front to back)
        y_coords = np.linspace(extent[1], extent[3], cols)  # ymax to ymin (left to right)

        # Create point cloud data
        points = []
        z_min, z_max = heightmap.min(), heightmap.max()
        z_range = z_max - z_min if z_max > z_min else 1.0

        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                z = heightmap[i, j]
                # Color based on height (red=high, blue=low)
                t = (z - z_min) / z_range
                r = int(255 * t)
                g = int(255 * (1 - abs(t - 0.5) * 2))
                b = int(255 * (1 - t))
                rgb = struct.unpack("f", struct.pack("I", (r << 16) | (g << 8) | b))[0]
                points.append([x, y, z, rgb])

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = "base_link"

        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16
        cloud_msg.width = len(points)
        cloud_msg.height = 1
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False

        # Pack point data
        cloud_msg.data = b"".join([struct.pack("ffff", *p) for p in points])

        self.heightmap_cloud_pub.publish(cloud_msg)

    def publish_heightmap_gridmap(self):
        """Publish heightmap as GridMap for RViz/debugging."""
        if self.current_heightmap is None or self.heightmap_extent is None or self.heightmap_position is None:
            return

        heightmap = self.current_heightmap
        extent = self.heightmap_extent  # [xmax, ymax, xmin, ymin]
        rows, cols = heightmap.shape

        # Create GridMap message
        # Use same frame and position as original elevation map for correct visualization
        msg = GridMap()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.info.resolution = (extent[0] - extent[2]) / cols  # length_x / cols
        msg.info.length_x = extent[0] - extent[2]
        msg.info.length_y = extent[1] - extent[3]
        msg.info.pose.position.x = self.heightmap_position[0]
        msg.info.pose.position.y = self.heightmap_position[1]
        msg.info.pose.position.z = self.heightmap_position[2]
        msg.info.pose.orientation.w = 1.0

        # Add elevation layer
        msg.layers = ["elevation"]
        msg.basic_layers = ["elevation"]

        # Create data array
        data_array = Float32MultiArray()
        data_array.layout.dim = [
            MultiArrayDimension(label="column_index", size=cols, stride=cols * rows),
            MultiArrayDimension(label="row_index", size=rows, stride=rows),
        ]
        data_array.data = heightmap.flatten().astype(np.float32).tolist()
        msg.data = [data_array]

        # Set outer/inner start indices
        msg.outer_start_index = 0
        msg.inner_start_index = 0

        self.heightmap_gridmap_pub.publish(msg)
        self.get_logger().info(
            f"Published /policy_heightmap: {heightmap.shape}, "
            f"range=[{heightmap.min():.3f}, {heightmap.max():.3f}]",
            throttle_duration_sec=1.0,
        )

    def get_flipper_angles(self) -> np.ndarray | None:
        """Extract flipper angles from joint state in policy order."""
        if self.current_joint_state is None:
            return None

        angles = np.zeros(4, dtype=np.float32)
        name_to_idx = {name: i for i, name in enumerate(self.current_joint_state.name)}

        for i, flipper_name in enumerate(self.FLIPPER_NAMES):
            if flipper_name in name_to_idx:
                angles[i] = self.current_joint_state.position[name_to_idx[flipper_name]]
            else:
                self.get_logger().warn(f"Flipper joint '{flipper_name}' not found in joint state")
                return None

        # Clamp to valid range
        angles = np.clip(angles, -np.pi / 2, np.pi / 2)
        return angles

    def get_goal_vector_local(self) -> np.ndarray | None:
        """Compute goal vector in robot's local frame."""
        if self.current_goal is None or self.current_odom is None:
            return None

        # Robot position in world frame
        robot_pos = np.array(
            [
                self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y,
                self.current_odom.pose.pose.position.z,
            ]
        )

        # Goal position in world frame
        goal_pos = np.array(
            [
                self.current_goal.pose.position.x,
                self.current_goal.pose.position.y,
                self.current_goal.pose.position.z,
            ]
        )

        # Goal vector in world frame
        goal_vec_world = goal_pos - robot_pos

        # Get robot orientation as rotation matrix
        q = self.current_odom.pose.pose.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])

        # Transform to local frame (inverse rotation)
        goal_vec_local = rot.inv().apply(goal_vec_world)

        return goal_vec_local.astype(np.float32)

    def get_velocities_local(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get linear and angular velocities in robot's local frame."""
        # Try odom first
        if self.current_odom is not None:
            xd_local = np.array(
                [
                    self.current_odom.twist.twist.linear.x,
                    self.current_odom.twist.twist.linear.y,
                    self.current_odom.twist.twist.linear.z,
                ],
                dtype=np.float32,
            )
            omega_local = np.array(
                [
                    self.current_odom.twist.twist.angular.x,
                    self.current_odom.twist.twist.angular.y,
                    self.current_odom.twist.twist.angular.z,
                ],
                dtype=np.float32,
            )
            return xd_local, omega_local

        # Fallback to IMU (only has angular velocity, not linear)
        if self.current_imu is not None:
            xd_local = np.zeros(3, dtype=np.float32)  # No linear velocity from IMU
            omega_local = np.array(
                [
                    self.current_imu.angular_velocity.x,
                    self.current_imu.angular_velocity.y,
                    self.current_imu.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            return xd_local, omega_local

        return None

    def get_orientation_quat(self) -> np.ndarray | None:
        """Get orientation quaternion in ROS format (x, y, z, w)."""
        # Try odom first
        if self.current_odom is not None:
            q = self.current_odom.pose.pose.orientation
            return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

        # Fallback to IMU
        if self.current_imu is not None:
            q = self.current_imu.orientation
            return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

        return None

    def track_velocities_to_twist(self, track_vels: np.ndarray) -> Twist:
        """
        Convert 4 track velocities to Twist command.

        Track order: front_left, front_right, rear_left, rear_right
        For differential drive:
        - linear.x = average of all track velocities
        - angular.z = (right - left) / track_width
        """
        # Average left and right sides
        left_vel = (track_vels[0] + track_vels[2]) / 2.0  # front_left + rear_left
        right_vel = (track_vels[1] + track_vels[3]) / 2.0  # front_right + rear_right

        twist = Twist()
        twist.linear.x = float((left_vel + right_vel) / 2.0)
        twist.angular.z = float((right_vel - left_vel) / self.TRACK_WIDTH)

        return twist

    def control_callback(self):
        """Main control loop - infer action and publish commands."""
        # Log status of each input for debugging
        status = []
        if self.current_odom is None:
            status.append("odom:NO")
        else:
            status.append("odom:OK")

        if self.current_imu is None:
            status.append("imu:NO")
        else:
            status.append("imu:OK")

        if self.current_joint_state is None:
            status.append("joints:NO")
        else:
            status.append("joints:OK")

        if self.current_goal is None:
            status.append("goal:NO")
        else:
            status.append("goal:OK")

        if self.current_heightmap is None:
            status.append("heightmap:NO")
        else:
            status.append(f"heightmap:{self.current_heightmap.shape}")

        self.get_logger().info(f"Status: {', '.join(status)}", throttle_duration_sec=1.0)

        # For now, just test elevation map - skip other requirements
        if self.current_heightmap is None:
            return

        # Publish heightmap visualization (before goal check so it's always visible)
        self.publish_heightmap_gridmap()

        # Gather inputs (may be None)
        thetas = self.get_flipper_angles()
        goal_vec_local = self.get_goal_vector_local()
        velocities = self.get_velocities_local()
        quat = self.get_orientation_quat()

        # Log extracted features for debugging
        self.get_logger().info(
            f"Heightmap: shape={self.current_heightmap.shape}, "
            f"extent={self.heightmap_extent}, "
            f"min={self.current_heightmap.min():.2f}, max={self.current_heightmap.max():.2f}",
            throttle_duration_sec=1.0,
        )

        # Unpack velocities
        if velocities is None:
            return
        xd_local, omega_local = velocities

        # Check required inputs
        if goal_vec_local is None:
            # No goal set - publish zero velocities to stand in place
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            # Also publish zero flipper velocities
            for key in ["front_left", "front_right", "rear_left", "rear_right"]:
                msg = Float64()
                msg.data = 0.0
                self.flipper_pubs[key].publish(msg)
            return
        if thetas is None:
            return
        if quat is None:
            return

        # Log all inputs for debugging
        # Compute roll/pitch from quaternion for verification
        rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

        self.get_logger().info(
            f"INPUTS:\n"
            f"  goal_local: ({goal_vec_local[0]:.2f}, {goal_vec_local[1]:.2f}, {goal_vec_local[2]:.2f}) m\n"
            f"  linear_vel: ({xd_local[0]:.2f}, {xd_local[1]:.2f}, {xd_local[2]:.2f}) m/s\n"
            f"  angular_vel: ({omega_local[0]:.2f}, {omega_local[1]:.2f}, {omega_local[2]:.2f}) rad/s\n"
            f"  flippers [FL,FR,RL,RR]: ({thetas[0]:.2f}, {thetas[1]:.2f}, {thetas[2]:.2f}, {thetas[3]:.2f}) rad\n"
            f"  quat (x,y,z,w): ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}) -> roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f} deg\n"
            f"  heightmap: {self.current_heightmap.shape}, extent={self.heightmap_extent}, range=[{self.current_heightmap.min():.3f}, {self.current_heightmap.max():.3f}]",
            throttle_duration_sec=1.0,
        )

        # Run policy inference
        try:
            action = self.policy.infer_action(
                heightmap=self.current_heightmap,
                heightmap_extent=self.heightmap_extent,
                goal_vec_local=goal_vec_local,
                xd_local=xd_local,
                omega_local=omega_local,
                thetas=thetas,
                quat=quat,
            )
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        # Parse action output and publish commands
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float64)

        if self._is_ftr:
            # FTR action: [v, w, fl, fr, rl, rr] — 6-D
            # v and w are already in m/s and rad/s (bounded by action spec ≈ [-1, 1])
            twist = Twist()
            twist.linear.x = float(action[0])
            twist.angular.z = float(action[1])
            self.cmd_vel_pub.publish(twist)

            flipper_velocities = action[2:6] * self.flipper_velocity_scale
            flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
            for i, key in enumerate(flipper_keys):
                msg = Float64()
                msg.data = float(flipper_velocities[i])
                self.flipper_pubs[key].publish(msg)

            self.get_logger().info(
                f"CMD(ftr): vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
                f"flipper_vel=[{flipper_velocities[0]:.2f}, {flipper_velocities[1]:.2f}, "
                f"{flipper_velocities[2]:.2f}, {flipper_velocities[3]:.2f}]",
                throttle_duration_sec=0.5,
            )
        else:
            # Native action: [track_fl, track_fr, track_rl, track_rr, flip_fl, flip_fr, flip_rl, flip_rr] — 8-D
            track_velocities = action[:4]
            flipper_velocities = action[4:8] * self.flipper_velocity_scale

            twist = self.track_velocities_to_twist(track_velocities)
            self.cmd_vel_pub.publish(twist)

            flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
            for i, key in enumerate(flipper_keys):
                msg = Float64()
                msg.data = float(flipper_velocities[i])
                self.flipper_pubs[key].publish(msg)

            self.get_logger().info(
                f"CMD: vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
                f"flipper_vel=[{flipper_velocities[0]:.2f}, {flipper_velocities[1]:.2f}, "
                f"{flipper_velocities[2]:.2f}, {flipper_velocities[3]:.2f}]",
                throttle_duration_sec=0.5,
            )


    @staticmethod
    def _detect_ftr_config(config_path: str) -> bool:
        """Return True if config_path is an FTR training config (train_ftr.py)."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            return "ftr_obs_encoder_opts" in cfg or "task" in cfg
        except Exception:
            return False


def main(args=None):
    rclpy.init(args=args)
    node = FlipperPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
