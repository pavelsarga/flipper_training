#!/usr/bin/env python3
"""
ROS2 node for deploying the trained flipper control policy on the MARV robot.

Subscribes to robot state and elevation map, publishes track velocities and flipper commands.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, PointCloud2, PointField, Imu, Image
from geometry_msgs.msg import PoseStamped, Twist, Point
from std_msgs.msg import Float64, MultiArrayDimension, MultiArrayLayout, Float32MultiArray, Bool
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as TimeMsg
import struct

import cv2
import torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate as ndimage_rotate

from flipper_training.experiments.ppo.ftr_heightmap_window import HM_ROWS, HM_COLS, HM_RES


class FlipperPolicyNode(Node):
    """ROS2 node that runs the trained flipper control policy."""

    # Flipper joint name mapping (order expected by policy)
    FLIPPER_NAMES = ["front_left_flipper_j", "front_right_flipper_j", "rear_left_flipper_j", "rear_right_flipper_j"]

    # Track width for differential drive conversion (meters)
    TRACK_WIDTH = 0.36  # From robot config rover_bodyWidth

    # Gazebo's flipper joints are physically unbounded (marv_model.sdf's revolute
    # joint limits are +/-1e16 -- no hard stop at all), so every policy's flipper
    # commands must be soft-limited here or a runaway/erratic policy output can spin
    # a flipper indefinitely. +/-90 deg matches this project's own documented
    # fully-up/fully-down convention (CLAUDE.md's Flipper Angle Conventions table) and
    # is symmetric, so it applies the same regardless of which raw/FTR sign
    # convention a given policy's output happens to be in.
    FLIPPER_ANGLE_LIMIT = math.pi / 2

    def __init__(self):
        super().__init__("flipper_policy_node")

        # Declare parameters
        self.declare_parameter("config_path", "")
        self.declare_parameter("policy_weights_path", "")
        self.declare_parameter("vecnorm_weights_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("control_rate", 10.0)  # Hz
        # Temporal decay for the heightmap: new = decay*old + (1-decay)*incoming.
        # DEFAULT 0.0 = pure pass-through of the incoming map.
        # This used to default to 0.95, which is an ~20-frame exponential average
        # over an EGOCENTRIC map: as the robot moves, every obstacle smears a long
        # ghost trail behind (and beside) its true position, so the policy sees
        # obstacles that are no longer there and sees real ones at ~5% of their
        # true height for the first second. No policy here was trained on a
        # temporally filtered map -- /elevation_map_filtered is already fused and
        # inpainted by elevation_mapping, which is the smoothing that belongs in
        # this pipeline. Raise this only if the raw map is visibly noisy, and then
        # only slightly.
        self.declare_parameter("heightmap_decay", 0.0)
        self.declare_parameter("heightmap_layer", "elevation")  # Layer name in GridMap
        self.declare_parameter("flipper_velocity_scale", 1.0)  # Scale factor for flipper velocities
        self.declare_parameter("publish_debug_cloud", True)  # Publish heightmap as point cloud for debugging
        # When true, never publish angular.z from inference (FTR/native action
        # spaces only -- the rl_baselines family already never publishes any
        # /cmd_vel at all, fixed or otherwise); steering is left entirely to the
        # user/teleop while the policy still drives linear.x and flippers.
        # Default false: the policy steers, which is how it was trained.
        self.declare_parameter("disable_angular_output", False)
        # Odometry the POLICY steers on. Separate from ground truth on purpose: the
        # arena/scoring pipeline (arena_server, score.py, area_id_publisher) is
        # hardcoded to /ground_truth_odom and must stay that way regardless of this
        # setting, so obstacle-area detection and TQ_n/CL_n scoring stay accurate
        # even if the policy is evaluated on noisier, estimated localization.
        self.declare_parameter("odom_topic", "/ground_truth_odom")
        # Opt-in gate, matching the convention the standalone baseline controllers use
        # (marv_xu_hto/controller_base.py's drive_only_when_enabled/drive_enable):
        # while it is on, the node stays stationary until something publishes True on
        # drive_enable. It defaults OFF -- a policy spawns RUNNING. The gate used to
        # default on and be released by the startup-override pulse this node no longer
        # sends; leaving it on with no pulse would just wedge every policy at standstill.
        self.declare_parameter("drive_only_when_enabled", False)

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
        self.disable_angular_output = self.get_parameter("disable_angular_output").get_parameter_value().bool_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.drive_gated = self.get_parameter("drive_only_when_enabled").get_parameter_value().bool_value
        # If gated: overridden by default, must be explicitly enabled. If not gated
        # (opted out via param): always allowed to drive, same as before this existed.
        self._drive_enabled = not self.drive_gated
        self.create_subscription(Bool, f"/{self.get_name()}/drive_enable", self._on_drive_enable, 10)
        self.get_logger().info(
            f"drive_enable gate: {'ARMED (overridden until /'+self.get_name()+'/drive_enable=true)' if self.drive_gated else 'disabled (always drives)'}"
        )
        self.get_logger().info(
            f"angular output: {'DISABLED (steering left to teleop)' if self.disable_angular_output else 'enabled (policy steers)'}"
        )

        # The project's actual pre-existing "override" control -- the sim UI's
        # OVERRIDE button and the gamepad's L3 button (ui.py/_on_override_toggled,
        # pad_extra_controls.py) both toggle this same /flipper_override topic
        # (bridged to the Gazebo plugin at /model/marv/flipper_override). That plugin
        # only ever gated flipper POSITION commands, deliberately letting velocity
        # commands (and /cmd_vel) straight through as "the manual path"
        # (flipper_control_plugin.cpp:94-102) -- so toggling OVERRIDE never stopped a
        # velocity-driven policy's forward motion. Subscribing here closes that gap:
        # this node treats /flipper_override the same as drive_enable=False, but ALSO
        # zeroes flipper output (both /flippers_cmd_vel/* and /flippers_cmd_pos/*),
        # matching the button's own label ("OVERRIDE ON (manual flippers)").
        #
        # This node only ever LISTENS on /flipper_override -- it never raises it.
        # It used to burst-publish override=true at startup so a policy spawned
        # stationary and needed a button press to go, which meant every rl:* run
        # began overridden and the sweep had to pulse the flag back down to arm it.
        # A policy now spawns RUNNING, and the flag belongs entirely to the three
        # things that legitimately take the flippers away from it: the UI/gamepad
        # OVERRIDE button, the gap-zone auto-arm (pad_extra_controls' neutral
        # strips), and autodrive's unstick manoeuvre (stuck_override).
        #
        # The falling edge (override released) still means "go": re-arm driving and,
        # if nothing else owns /goal_pose, send a fresh goal -- so handing the
        # flippers back needs no separate drive_enable or goal_pose publish.
        self._flipper_override_active = False
        self.create_subscription(Bool, "/flipper_override", self._on_flipper_override, 10)
        self.declare_parameter("auto_goal_distance", 5.0)
        self.auto_goal_distance = self.get_parameter("auto_goal_distance").get_parameter_value().double_value
        # The one-shot auto-goal is a fallback for running this node bare. When
        # obstacle_goal_publisher is up (policy.launch.py's obstacle_goal:=true,
        # the default) it owns /goal_pose and keeps the goal on the current
        # obstacle area's exit, so this must stay out of its way -- two publishers
        # racing on /goal_pose would leave the goal wherever the last message
        # landed, which is exactly the "goal doesn't follow the obstacle" symptom.
        self.declare_parameter("auto_goal_on_release", True)
        self.auto_goal_on_release = self.get_parameter("auto_goal_on_release").get_parameter_value().bool_value
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        if not config_path or not policy_weights_path:
            self.get_logger().error("config_path and policy_weights_path parameters are required!")
            raise ValueError("Missing required parameters")

        # Auto-detect which family this config/checkpoint belongs to: one of the
        # rl_baselines modules (atd3qn/icmd3qn/hfc/mitriakov/creps, MARV_RL-trained,
        # rl_modules.* policy classes), the generic FTR/marv_rl path, or the legacy
        # native (non-FTR) path.
        from flipper_training.experiments.ppo.rl_baseline_inference_modules import (
            detect_rl_baseline_module_name,
            build_rl_baseline_inference_module,
        )

        self._rl_baseline_module_name = detect_rl_baseline_module_name(config_path)
        self._is_ftr = self._detect_ftr_config(config_path) if self._rl_baseline_module_name is None else True

        if self._rl_baseline_module_name is not None:
            self.get_logger().info(f"Loading rl_baselines module '{self._rl_baseline_module_name}' from {config_path}")
            self.policy = build_rl_baseline_inference_module(
                self._rl_baseline_module_name,
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                device=device,
            )
        elif self._is_ftr:
            self.get_logger().info(f"Loading FTR policy from {config_path}")
            from flipper_training.experiments.ppo.ftr_policy_inference_module import FtrPolicyInferenceModule
            self.policy = FtrPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                vecnorm_weights_path=vecnorm_weights_path if vecnorm_weights_path else None,
                device=device,
            )
        else:
            self.get_logger().info(f"Loading native policy from {config_path}")
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
        # Note: ground_truth_odom uses RELIABLE QoS, so we must match it. If
        # odom_topic is repointed at a different source (e.g. lidar ICP odometry),
        # confirm that source's QoS is also RELIABLE or this subscription won't connect.
        self.get_logger().info(f"Policy odometry source: {odom_topic}")
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, reliable_qos)
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
        # Position-mode rl_baselines policies (hfc/mitriakov/creps) output absolute
        # flipper-angle targets — a real, separately-actuated interface also used by
        # the geometric baselines (marv_xu_hto/controller_base.py, marv_oehler_baseline).
        self.flipper_pos_pubs = {
            "front_left": self.create_publisher(Float64, "/flippers_cmd_pos/front_left", 10),
            "front_right": self.create_publisher(Float64, "/flippers_cmd_pos/front_right", 10),
            "rear_left": self.create_publisher(Float64, "/flippers_cmd_pos/rear_left", 10),
            "rear_right": self.create_publisher(Float64, "/flippers_cmd_pos/rear_right", 10),
        }

        # Debug visualization publishers
        self.heightmap_cloud_pub = self.create_publisher(PointCloud2, "/policy_heightmap_debug", 10)
        self.heightmap_gridmap_pub = self.create_publisher(GridMap, "/policy_heightmap", 10)
        # Plain 2D image of the same heightmap array -- an RViz Image panel is
        # much easier to read at a glance than the 3D point cloud/GridMap
        # displays (readable even when the 3D view isn't framed on the robot).
        self.heightmap_image_pub = self.create_publisher(Image, "/policy_heightmap_image", 10)
        # Top-down flipper-command HUD: 4 spots (FL/FR/RL/RR) each showing an
        # up/down arrow or "-" for still, front-to-the-left, lengthwise layout --
        # a quick glance readout of what the policy is commanding right now,
        # independent of the 3D view's current camera angle.
        self.flipper_cmd_hud_pub = self.create_publisher(Image, "/policy_flipper_command_hud", 10)
        # rl_generic.rviz/pan_d3qn.rviz/azayev.rviz all already expect this topic
        # ("goal-direction arrow + roll/pitch") but nothing was ever publishing to it.
        self.policy_obs_markers_pub = self.create_publisher(MarkerArray, "/policy_obs_markers", 10)
        # Separate from policy_obs_markers (INPUTS): what the policy is actually
        # commanding right now -- velocity arrow + flipper values.
        self.policy_action_markers_pub = self.create_publisher(MarkerArray, "/policy_action_markers", 10)
        # Every FTR-trained policy (marv_rl/FTR PPO + all five rl_baselines) is fed
        # the SAME fixed-size 45x21 @ 0.05m/cell window (ftr_heightmap_window) --
        # this draws that window's extent as a rectangle around the robot, in the
        # style of the geometric baselines' spatial debug markers (SPHERE/ARROW/
        # LINE_STRIP in Visualization.cpp), not text.
        self.heightmap_extent_pub = self.create_publisher(MarkerArray, "/policy_heightmap_extent", 10)
        # CREPS-specific: unlike atd3qn/icmd3qn (which reduce the full 45x21 crop to
        # a 15-band terrain profile) CREPS reads exactly ONE heightmap cell plus
        # pitch. This draws that single cell as a square on the ground plane
        # (base_link frame, same as the extent rectangle above) -- a detection-point
        # marker, not text.
        self.creps_debug_pub = self.create_publisher(MarkerArray, "/creps_debug_markers", 10)

        # Live speed dial for the rl_baselines family (atd3qn/icmd3qn/hfc/mitriakov/
        # creps) -- these publish ONLY flipper commands (v/w are user-driven, see
        # the "not the network's output" comment in control_callback), so scaling
        # them is the only lever that slows/speeds up what those policies actually
        # do. 1.0 = policy's real output, 0.0 = frozen, up to 5x. Driven by a
        # QSlider on the CommandScale rviz panel (marv_rviz_panels); default 1.0
        # until a slider message arrives.
        self.command_scale = 1.0
        self.create_subscription(Float64, "/policy_command_scale", self._on_command_scale, 10)

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

            # GridMap stores each layer as a flattened COLUMN-MAJOR array:
            # dim[0] = column index (outer, size=cols), dim[1] = row index
            # (inner, size=rows) -- i.e. flat index k = colIdx*rows + rowIdx.
            cols = msg.data[layer_idx].layout.dim[0].size
            rows = msg.data[layer_idx].layout.dim[1].size

            # np.reshape(..., order='F') makes the FIRST axis of the target
            # shape vary fastest. Our flat data already has rowIdx varying
            # fastest (within each column-contiguous block), so reshaping to
            # (rows, cols) -- NOT (cols, rows) -- is what actually lines up
            # axis 0 = row (X, front-back) and axis 1 = col (Y, left-right).
            # (The previous code reshaped to (cols, rows) then '.T'd, which
            # for a square rows==cols crop silently produced the numerically
            # "right" shape while actually swapping X and Y -- an axis-swap
            # bug independent of robot heading, confirmed live 2026-08-08:
            # /policy_heightmap_debug was rotated relative to the chassis
            # even at near-zero yaw.)
            heightmap = data.reshape((rows, cols), order='F')

            # Apply circular buffer start indices if present. outer_start_index
            # is the ring-buffer offset for the COLUMN dimension (Y, axis 1),
            # inner_start_index for the ROW dimension (X, axis 0) -- matching
            # dim[0]=column/dim[1]=row above, not axis position.
            if msg.outer_start_index != 0 or msg.inner_start_index != 0:
                heightmap = np.roll(heightmap, -msg.inner_start_index, axis=0)
                heightmap = np.roll(heightmap, -msg.outer_start_index, axis=1)

            # Handle NaN values (unknown areas) -- must happen BEFORE the yaw
            # rotation below, since spline interpolation would otherwise smear
            # NaN across the rotated result.
            heightmap = np.nan_to_num(heightmap, nan=0.0)

            # msg.header.frame_id is "map" (elevation_mapping publishes in a
            # fixed, world-aligned frame -- it does NOT rotate with the robot).
            # Row 0 of the array above is therefore always "+X in map frame",
            # NOT "in front of the robot", unless the robot happens to be at
            # yaw=0. The policy was trained on an EGOCENTRIC heightmap ("as if
            # standing behind the robot looking forward" -- objects ahead in
            # the upper part), so we rotate the map-aligned crop here to
            # compensate for the robot's current heading before anything
            # downstream treats "up" as "forward". (Previously this rotation
            # was entirely missing -- see the old debug title's own "?": row 0
            # = front of robot?" -- so the heightmap was silently rotated by
            # whatever the robot's current yaw happened to be.)
            if self.current_odom is not None:
                q = self.current_odom.pose.pose.orientation
                yaw_deg = math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
                heightmap = ndimage_rotate(heightmap, angle=-yaw_deg, reshape=False, order=1, mode="constant", cval=0.0)

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
            self.publish_heightmap_image()

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

    def publish_heightmap_image(self):
        """Publish /policy_heightmap_image: the heightmap patch the network is
        ACTUALLY fed, as a plain sensor_msgs/Image viewable in an RViz Image panel.
        Row 0 = front of robot, top of image -- much easier to sanity-check at a
        glance than the 3D point cloud/GridMap displays, and stays readable
        regardless of where the 3D view camera is looking.

        Note this is deliberately the inference module's own post-crop/resample
        window (FTR: 45×21 over ±1.10 × ±0.50 m; native heightmap policies: the
        percep_shape/percep_extent patch), NOT self.current_heightmap. The latter
        is the full 8 m × 8 m elevation-map crop, which is square and therefore
        looks nothing like the policy's input -- showing it hid the fact that the
        FTR module was squeezing the whole 8 m map into its 2.25 m window. Falls
        back to the full map only until the first inference has run (and for the
        rl_baselines family, which consumes 1-D terrain bins, not a 2-D patch).
        """
        hm = getattr(self.policy, "last_policy_heightmap", None)
        if hm is None or getattr(hm, "ndim", 0) != 2:
            hm = self.current_heightmap
        if hm is None:
            return
        hm = np.asarray(hm, dtype=np.float32)
        z_min, z_max = float(hm.min()), float(hm.max())
        z_range = z_max - z_min if z_max > z_min else 1.0
        normalized = np.clip((hm - z_min) / z_range * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        # The policy window is tiny (45x21 for FTR) -- blow it up nearest-neighbour
        # so individual cells stay visible and countable in the RViz Image panel.
        scale = max(1, int(400 // max(colored.shape[0], colored.shape[1])))
        if scale > 1:
            colored = cv2.resize(colored, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        # This image is the module's whole post-crop window -- for atd3qn/icmd3qn
        # that's genuinely the full policy input (row-banded into 15 terrain means),
        # but CREPS's linear law reads exactly ONE cell of it (+ pitch, not shown
        # here). Draw that cell so the image doesn't overstate what CREPS actually
        # sees -- see publish_creps_debug_markers for the 3D equivalent.
        diag = getattr(self.policy, "last_diag", None) if self._rl_baseline_module_name == "creps" else None
        if diag is not None:
            r, c = int(diag["row"]), int(diag["col"])
            y0, y1 = r * scale, (r + 1) * scale
            x0, x1 = c * scale, (c + 1) * scale
            cv2.rectangle(colored, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), max(1, scale // 6))
        # Built by hand rather than via cv_bridge.cv2_to_imgmsg(): cv_bridge's
        # compiled half and its runtime type table disagree whenever the
        # interpreter's OpenCV differs from the one ROS was built against
        # (flipper_venv has cv2 5.x, whose type codes shift the channel count
        # by 5 bits instead of 3, so "bgr8" is 64 there but 16 in the C++
        # extension -> KeyError: 16 on every publish, which surfaced as a
        # 3 Hz "Error processing elevation map: 16"). A packed bgr8 Image
        # needs no conversion logic, so drop the dependency and stay
        # version-agnostic -- this also works where cv_bridge is fine.
        colored = np.ascontiguousarray(colored, dtype=np.uint8)
        img_msg = Image()
        img_msg.height, img_msg.width = colored.shape[:2]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = 3 * img_msg.width
        img_msg.data = colored.tobytes()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "base_link"
        self.heightmap_image_pub.publish(img_msg)

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
        cloud_msg.header.stamp = self._viz_stamp()
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

    @staticmethod
    def _viz_stamp() -> TimeMsg:
        """Stamp for base_link-attached RViz decorations: 0 == "latest
        available transform".

        These markers/clouds are rigidly attached to base_link (arrow at the
        robot origin, text above it, heightmap sampled in the robot frame), so
        RViz should draw them against the newest base_link pose it has.
        Stamping them with get_clock().now() instead asks for a TF time that
        does not exist yet: map->odom is published at ~20 Hz, so "now" is up
        to ~50 ms ahead of the newest sample and the lookup fails with
        "Lookup would require extrapolation into the future" -- which turns
        the whole display red in a map-fixed-frame view (measured live on the
        30-deg arena: 149/150 lookups at clock-now failed, 150/150 at a stamp
        tf2 can actually resolve succeeded).
        """
        return TimeMsg()

    def publish_policy_obs_markers(self, goal_vec_local: np.ndarray | None, roll_deg: float | None = None, pitch_deg: float | None = None):
        """Publish /policy_obs_markers: a goal-direction arrow using the SAME
        goal_vec_local the policy itself just consumed (real robot-frame vector, real
        length — not a static placeholder), plus a small roll/pitch text readout.
        rl_generic.rviz / pan_d3qn.rviz / azayev.rviz already display this topic; it
        just had nothing publishing to it.
        """
        markers = MarkerArray()
        now = self._viz_stamp()

        if goal_vec_local is not None:
            arrow = Marker()
            arrow.header.frame_id = "base_link"
            arrow.header.stamp = now
            arrow.ns = "policy_obs"
            arrow.id = 0
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                Point(x=0.0, y=0.0, z=0.3),
                Point(x=float(goal_vec_local[0]), y=float(goal_vec_local[1]), z=float(goal_vec_local[2]) + 0.3),
            ]
            arrow.scale.x = 0.06  # shaft diameter
            arrow.scale.y = 0.12  # head diameter
            arrow.scale.z = 0.0
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (0.1, 0.9, 0.2, 0.9)
            markers.markers.append(arrow)
        else:
            # No goal set: remove any previously-published arrow instead of leaving a
            # stale one pointing at the last goal.
            clear = Marker()
            clear.header.frame_id = "base_link"
            clear.header.stamp = now
            clear.ns = "policy_obs"
            clear.id = 0
            clear.action = Marker.DELETE
            markers.markers.append(clear)

        if roll_deg is not None and pitch_deg is not None:
            text = Marker()
            text.header.frame_id = "base_link"
            text.header.stamp = now
            text.ns = "policy_obs"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.z = 0.8
            text.scale.z = 0.15
            text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 1.0, 1.0, 0.9)
            text.text = f"roll={roll_deg:.1f} deg  pitch={pitch_deg:.1f} deg"
            markers.markers.append(text)

        self.policy_obs_markers_pub.publish(markers)

    def publish_policy_action_markers(self, twist: Twist | None, flipper_values, flipper_mode: str):
        """Publish /policy_action_markers: what the policy is actually commanding
        right now -- a velocity arrow (blue, distinct from the green goal-direction
        arrow on /policy_obs_markers) plus a flipper-command text readout.
        twist=None means /cmd_vel isn't being published at all right now (the
        rl_baselines family, which never publishes it; or overridden, where the
        caller should skip this entirely -- see the early-return in the main
        control callback).
        """
        markers = MarkerArray()
        now = self._viz_stamp()

        if twist is not None:
            arrow = Marker()
            arrow.header.frame_id = "base_link"
            arrow.header.stamp = now
            arrow.ns = "policy_action"
            arrow.id = 0
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            v = float(twist.linear.x)
            w = float(twist.angular.z)
            # Bend the arrow toward the commanded turn direction (small angle
            # approximation) so w is visible too, not just v as a straight shaft.
            end_x = v * math.cos(w * 0.3)
            end_y = v * math.sin(w * 0.3)
            arrow.points = [
                Point(x=0.0, y=0.0, z=0.5),
                Point(x=end_x, y=end_y, z=0.5),
            ]
            arrow.scale.x = 0.06  # shaft diameter
            arrow.scale.y = 0.12  # head diameter
            arrow.scale.z = 0.0
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (0.2, 0.45, 1.0, 0.9)
            markers.markers.append(arrow)
        else:
            clear = Marker()
            clear.header.frame_id = "base_link"
            clear.header.stamp = now
            clear.ns = "policy_action"
            clear.id = 0
            clear.action = Marker.DELETE
            markers.markers.append(clear)

        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp = now
        text.ns = "policy_action"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 1.1
        text.scale.z = 0.13
        text.color.r, text.color.g, text.color.b, text.color.a = (0.5, 0.75, 1.0, 0.9)
        vel_str = f"v={twist.linear.x:.2f} w={twist.angular.z:.2f}" if twist is not None else "v/w: user-driven"
        text.text = (
            f"{vel_str}\n"
            f"flippers ({flipper_mode}) [FL,FR,RL,RR]=[{flipper_values[0]:.2f}, {flipper_values[1]:.2f}, "
            f"{flipper_values[2]:.2f}, {flipper_values[3]:.2f}]"
        )
        markers.markers.append(text)

        self.policy_action_markers_pub.publish(markers)

    def _flipper_direction(self, key: str, flipper_mode: str, value: float) -> str:
        """"up" / "down" / "still" -- the PHYSICAL direction `value` commands for
        corner `key`, using the front/rear sign convention every limiter in this
        node already assumes (CLAUDE.md's Flipper Angle Conventions table: front
        negative = up, positive = down; rear positive = up, negative = down).

        flipper_mode="position": value is an absolute radian target -- compare
        against the live measured angle (self.flipper_positions) to get a
        direction of intended motion. Any other flipper_mode ("velocity"/"vel"):
        value is already a rad/s command, used directly.
        """
        is_front = key in ("front_left", "front_right")
        if flipper_mode == "position":
            delta = value - self.flipper_positions.get(key, 0.0)
            threshold = math.radians(1.0)
        else:
            delta = value
            threshold = 0.02  # rad/s
        if abs(delta) < threshold:
            return "still"
        physically_up = (delta < 0.0) if is_front else (delta > 0.0)
        return "up" if physically_up else "down"

    def publish_flipper_command_hud(self, flipper_values, flipper_mode: str):
        """Publish /policy_flipper_command_hud: a small top-down schematic (front
        to the LEFT, lengthwise) with one spot per corner (FL/FR/RL/RR), each
        drawn as an up arrow, down arrow, or a dash for "not moving" -- a
        quick-glance readout of what's actually being commanded right now,
        independent of the 3D view's current camera framing.
        """
        flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
        W, H = 360, 180
        img = np.full((H, W, 3), 32, dtype=np.uint8)

        body_x0, body_y0, body_x1, body_y1 = 70, 50, 290, 130
        cv2.rectangle(img, (body_x0, body_y0), (body_x1, body_y1), (90, 90, 90), 2, cv2.LINE_AA)
        # Front indicator: a triangle nose pointing left, off the front edge.
        nose = np.array([
            (body_x0 - 20, (body_y0 + body_y1) // 2),
            (body_x0, body_y0 + 8),
            (body_x0, body_y1 - 8),
        ], dtype=np.int32)
        cv2.fillPoly(img, [nose], (110, 110, 110))

        spots = {
            "front_left": (body_x0 + 32, body_y0),
            "front_right": (body_x0 + 32, body_y1),
            "rear_left": (body_x1 - 32, body_y0),
            "rear_right": (body_x1 - 32, body_y1),
        }
        labels = {"front_left": "FL", "front_right": "FR", "rear_left": "RL", "rear_right": "RR"}
        colors = {"up": (70, 210, 70), "down": (70, 130, 240), "still": (150, 150, 150)}

        for i, key in enumerate(flipper_keys):
            cx, cy = spots[key]
            direction = self._flipper_direction(key, flipper_mode, float(flipper_values[i]))
            color = colors[direction]
            cv2.circle(img, (cx, cy), 20, color, 2, cv2.LINE_AA)
            if direction == "up":
                cv2.arrowedLine(img, (cx, cy + 11), (cx, cy - 11), color, 3, cv2.LINE_AA, tipLength=0.5)
            elif direction == "down":
                cv2.arrowedLine(img, (cx, cy - 11), (cx, cy + 11), color, 3, cv2.LINE_AA, tipLength=0.5)
            else:
                cv2.line(img, (cx - 10, cy), (cx + 10, cy), color, 3, cv2.LINE_AA)
            label_y = cy - 30 if cy < (body_y0 + body_y1) // 2 else cy + 42
            cv2.putText(img, labels[key], (cx - 12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        img_msg = Image()
        img_msg.height, img_msg.width = img.shape[:2]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = 3 * img_msg.width
        img_msg.data = img.tobytes()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "base_link"
        self.flipper_cmd_hud_pub.publish(img_msg)

    def publish_heightmap_extent_marker(self):
        """Publish /policy_heightmap_extent: a rectangle on the ground plane around
        the robot showing the FTR heightmap window's real extent (+/-1.10m fore-aft x
        +/-0.50m lateral, base_link frame) -- what "the policy's heightmap input"
        actually spans, for every policy family that goes through
        ftr_heightmap_window (marv_rl/FTR PPO + all five rl_baselines), not just
        CREPS. No-op until the first successful inference (last_policy_heightmap set
        by _resample_heightmap/ftr_policy_inference_module).
        """
        hm = getattr(self.policy, "last_policy_heightmap", None)
        if hm is None or getattr(hm, "shape", None) != (HM_ROWS, HM_COLS):
            return
        half_x = (HM_ROWS // 2) * HM_RES
        half_y = (HM_COLS // 2) * HM_RES

        rect = Marker()
        rect.header.frame_id = "base_link"
        rect.header.stamp = TimeMsg()  # zero stamp -- see publish_policy_obs_markers for why
        rect.ns = "policy_heightmap_extent"
        rect.id = 0
        rect.type = Marker.LINE_STRIP
        rect.action = Marker.ADD
        rect.pose.orientation.w = 1.0  # required -- an all-zero quaternion is invalid and RViz drops the marker
        corners = [(half_x, half_y), (half_x, -half_y), (-half_x, -half_y), (-half_x, half_y), (half_x, half_y)]
        rect.points = [Point(x=cx, y=cy, z=0.02) for cx, cy in corners]
        rect.scale.x = 0.02
        rect.color.r, rect.color.g, rect.color.b, rect.color.a = (1.0, 1.0, 1.0, 0.8)
        self.heightmap_extent_pub.publish(MarkerArray(markers=[rect]))

    def publish_creps_debug_markers(self):
        """Publish /creps_debug_markers: the ONE heightmap cell CREPSInferenceModule's
        linear law actually reads (plus pitch, not spatially drawable), as a flat
        square on the ground plane -- a detection-point marker in the style of the
        geometric baselines' SPHERE/LINE_STRIP markers (Visualization.cpp), not text.
        Colored by height_ahead sign: green ~flat, red = rise ahead, blue = dip ahead
        -- the same sign CREPS's w2/w5 terms react to. No-op if this isn't a CREPS
        policy or infer_action hasn't run yet (last_diag is None until the first tick).

        The sample point is only ~0.20m ahead of base_link (matching training/the
        paper's own short lookahead) -- on this chassis that's still UNDER the body
        (the front flippers alone start ~0.26m out), so the square would be hidden
        beneath the robot mesh from most viewing angles. A vertical pole marks it so
        it's visible without moving/faking the actual sampled position.
        """
        diag = getattr(self.policy, "last_diag", None)
        if diag is None:
            return
        cell_half = HM_RES / 2.0
        x = diag["x_ahead"]
        z = diag["ground_ahead"]
        h = diag["height_ahead"]
        if abs(h) < 0.03:
            rgb = (0.2, 0.9, 0.3)
        elif h > 0:
            rgb = (1.0, 0.25, 0.15)
        else:
            rgb = (0.2, 0.55, 1.0)

        square = Marker()
        square.header.frame_id = "base_link"
        square.header.stamp = TimeMsg()  # zero stamp -- see publish_policy_obs_markers for why
        square.ns = "creps_debug"
        square.id = 0
        square.type = Marker.CUBE
        square.action = Marker.ADD
        square.pose.position.x = x
        square.pose.position.y = 0.0
        square.pose.position.z = z
        square.pose.orientation.w = 1.0  # required -- an all-zero quaternion is invalid and RViz drops the marker
        square.scale.x = square.scale.y = HM_RES * 1.8  # slightly oversized so it's visible against the grid
        square.scale.z = 0.02
        square.color.r, square.color.g, square.color.b = rgb
        square.color.a = 0.95

        ray = Marker()
        ray.header.frame_id = "base_link"
        ray.header.stamp = square.header.stamp
        ray.ns = "creps_debug"
        ray.id = 1
        ray.type = Marker.LINE_STRIP
        ray.action = Marker.ADD
        ray.pose.orientation.w = 1.0
        ray.points = [Point(x=0.0, y=0.0, z=z), Point(x=x - cell_half, y=0.0, z=z)]
        ray.scale.x = 0.015
        ray.color.r, ray.color.g, ray.color.b, ray.color.a = (0.9, 0.9, 0.9, 0.6)

        pole = Marker()
        pole.header.frame_id = "base_link"
        pole.header.stamp = square.header.stamp
        pole.ns = "creps_debug"
        pole.id = 2
        pole.type = Marker.LINE_STRIP
        pole.action = Marker.ADD
        pole.pose.orientation.w = 1.0
        pole.points = [Point(x=x, y=0.0, z=z), Point(x=x, y=0.0, z=z + 0.5)]
        pole.scale.x = 0.01
        pole.color.r, pole.color.g, pole.color.b, pole.color.a = (*rgb, 0.6)

        self.creps_debug_pub.publish(MarkerArray(markers=[square, ray, pole]))

    def _on_drive_enable(self, msg: Bool):
        was = self._drive_enabled
        self._drive_enabled = bool(msg.data)
        if was != self._drive_enabled:
            self.get_logger().info(f"drive_enable -> {self._drive_enabled}")

    def _on_flipper_override(self, msg: Bool):
        was = self._flipper_override_active
        self._flipper_override_active = bool(msg.data)
        if was != self._flipper_override_active:
            self.get_logger().info(f"/flipper_override -> {self._flipper_override_active}")
            if not was and self._flipper_override_active:
                # Rising edge: override just got turned ON. From here the control
                # loop stops publishing anything at all (see the early-return
                # there), so bring the robot to a stop now -- otherwise Gazebo
                # just holds whatever /cmd_vel and /flippers_cmd_vel/* happened to
                # be commanded at this instant forever (no dead-man's-switch).
                self.stop_all()
            elif was and not self._flipper_override_active:
                # Falling edge: override just got turned OFF -- auto-arm driving
                # (bypasses the separate drive_enable gate) and send a fresh goal
                # so the policy has something to navigate toward immediately.
                self._drive_enabled = True
                if self.auto_goal_on_release:
                    self._send_auto_goal()
                elif self.current_goal is None:
                    self.get_logger().warn(
                        "Override released with no goal yet -- waiting for "
                        "obstacle_goal_publisher (/obstacle_id must be != -1)")

    def _send_auto_goal(self):
        """Publish a goal auto_goal_distance metres ahead of the robot's current
        heading (map frame), mirroring send_goal.py's logic -- triggered by the
        override falling edge instead of requiring a separate manual step."""
        if self.current_odom is None:
            self.get_logger().warn("Cannot auto-send goal: no odometry yet")
            return
        p = self.current_odom.pose.pose.position
        q = self.current_odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = p.x + self.auto_goal_distance * math.cos(yaw)
        goal.pose.position.y = p.y + self.auto_goal_distance * math.sin(yaw)
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(goal)
        self.current_goal = goal
        self.get_logger().info(
            f"Auto-goal sent: {self.auto_goal_distance:.1f}m ahead at yaw={math.degrees(yaw):.1f} deg"
        )

    def _is_overridden(self) -> bool:
        """True if either override source says stop: drive_enable not yet armed, or
        the UI/gamepad OVERRIDE button is engaged. See the early-return in the main
        control callback -- all command publishing is skipped entirely (not just
        zeroed) while this is true, so teleop/manual_flipper_tilt own /cmd_vel and
        /flippers_cmd_vel/* uncontested."""
        return (not self._drive_enabled) or self._flipper_override_active

    def stop_all(self):
        """Publish a zero /cmd_vel + zero flipper velocities.

        Gazebo's diff-drive/flipper velocity actuators hold the LAST commanded value
        indefinitely once a publisher stops (there's no dead-man's-switch/timeout) --
        so overriding this policy (killing this node, or a launch escalating SIGINT to
        SIGTERM) previously left the robot driving forward forever on whatever /cmd_vel
        happened to be commanded at the moment this process died. Called from main()'s
        shutdown path so an override actually stops the robot instead of freezing its
        last command in place. Only zeroes velocity-mode topics (/cmd_vel,
        /flippers_cmd_vel/*) -- a stale POSITION target on /flippers_cmd_pos/* just
        holds the flipper still, so it doesn't need to be re-zeroed here.
        """
        try:
            self.cmd_vel_pub.publish(Twist())
            for pub in self.flipper_vel_pubs.values():
                pub.publish(Float64(data=0.0))
        except Exception as e:
            self.get_logger().error(f"stop_all() failed: {e}")

    # Order the per-corner limits are indexed in: matches FLIPPER_NAMES / the
    # [FL, FR, RL, RR] action layout every module here uses.
    _LIMIT_ORDER = ("front_left", "front_right", "rear_left", "rear_right")

    def _flipper_bounds(self, key: str) -> tuple[float, float]:
        """(low, high) angle limits in radians for one flipper corner.

        Read from the loaded policy's own training config when it exposes them
        (`flipper_limits_rad`, set by the FTR and rl_baselines inference modules
        from env_cfg_overrides' marv_flipper_{front,back}_{up,down}_deg -- e.g.
        front [-60, +80] deg, rear [-80, +60] deg for the marv_rl runs). These
        are asymmetric and per-pair, so the symmetric +/-90 fallback below let
        every flipper travel well past anything it saw in training -- 30 deg
        further up and 10 deg further down.

        Falls back to +/-FLIPPER_ANGLE_LIMIT only when the module publishes no
        limits (e.g. the native PPO path, whose configs carry no MARV flipper
        limits). The fallback still matters as a backstop: the Gazebo model's own
        joint limits are +/-1e16, i.e. no hard stop at all.
        """
        limits = getattr(self.policy, "flipper_limits_rad", None)
        if limits is not None:
            try:
                low, high = limits[self._LIMIT_ORDER.index(key)]
                return float(low), float(high)
            except (ValueError, IndexError):
                pass
        return -self.FLIPPER_ANGLE_LIMIT, self.FLIPPER_ANGLE_LIMIT

    def _on_command_scale(self, msg: Float64):
        self.command_scale = float(np.clip(msg.data, 0.0, 5.0))

    def _limit_flipper_velocity(self, key: str, velocity: float) -> float:
        """Zero a velocity command that would drive a flipper further past its
        angle limits, using the last measured joint_state angle for that corner.
        Lets the flipper move freely everywhere inside the limits and only clips
        the specific direction that would overshoot (not a general clamp on the
        command's magnitude), so normal motion is unaffected.
        """
        low, high = self._flipper_bounds(key)
        angle = self.flipper_positions.get(key, 0.0)
        if angle >= high and velocity > 0.0:
            return 0.0
        if angle <= low and velocity < 0.0:
            return 0.0
        return velocity

    def _limit_flipper_position(self, key: str, position: float) -> float:
        """Clamp an absolute flipper position target to that corner's limits."""
        low, high = self._flipper_bounds(key)
        return float(np.clip(position, low, high))

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

        # Clamp the MEASURED angles to the same limits the commands are clamped to,
        # so a flipper parked outside them (dragged there by terrain, or left over
        # from manual/teleop control, which does not go through this node's limiter)
        # is still reported to the policy as an in-distribution angle rather than
        # one it never saw in training.
        for i, key in enumerate(self._LIMIT_ORDER):
            low, high = self._flipper_bounds(key)
            angles[i] = np.clip(angles[i], low, high)
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
            self.publish_policy_obs_markers(goal_vec_local=None)
            return
        if thetas is None:
            return
        if quat is None:
            return

        # Log all inputs for debugging
        # Compute roll/pitch from quaternion for verification
        rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
        self.publish_policy_obs_markers(goal_vec_local=goal_vec_local, roll_deg=roll, pitch_deg=pitch)

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
        robot_world_z = self.current_odom.pose.pose.position.z if self.current_odom is not None else None
        try:
            if self._rl_baseline_module_name is not None:
                rl_action = self.policy.infer_action(
                    heightmap=self.current_heightmap,
                    heightmap_extent=self.heightmap_extent,
                    goal_vec_local=goal_vec_local,
                    xd_local=xd_local,
                    omega_local=omega_local,
                    thetas=thetas,
                    quat=quat,
                    robot_world_z=robot_world_z,
                    dt=self.dt,
                )
            else:
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

        # Overridden (OVERRIDE button/L3, or drive not yet armed): publish nothing
        # at all here. /cmd_vel and /flippers_cmd_vel/* belong to teleop and
        # manual_flipper_tilt's relay during override -- continuing to publish
        # anything from this node (even zeros) would fight their publishes on the
        # same topics, since Gazebo just acts on whichever message arrived last.
        # stop_all() (called once on the override rising edge, see
        # _on_flipper_override) already handles bringing the robot to a stop.
        if self._is_overridden():
            return

        # rl_baselines (atd3qn/icmd3qn/hfc/mitriakov/creps): uniform RLBaselineAction
        # contract, dispatched to the velocity or position flipper interface.
        if self._rl_baseline_module_name is not None:
            # v/w are NOT network output for this module family -- they're read
            # straight from config (env_cfg_overrides.fixed_forward_vel / a
            # hardcoded 0 for w; see rl_baseline_inference_modules.py). Publishing
            # them would silently drive the robot at a fixed speed the "policy"
            # never actually decided -- leave /cmd_vel to the user/teleop instead
            # and only publish the flipper commands these modules actually infer.
            flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
            pubs = self.flipper_vel_pubs if rl_action.flipper_mode == "velocity" else self.flipper_pos_pubs
            limit = self._limit_flipper_velocity if rl_action.flipper_mode == "velocity" else (
                self._limit_flipper_position)
            for i, key in enumerate(flipper_keys):
                raw = float(rl_action.flippers[i])
                if rl_action.flipper_mode == "velocity":
                    # Directly proportional to speed.
                    scaled = raw * self.command_scale
                else:
                    # Position mode: `raw` is an ABSOLUTE angle target, so scaling it
                    # directly would move the target itself, not the speed of motion.
                    # Scale the DELTA from the live measured angle instead -- 1x moves
                    # exactly as far as the policy intended this tick, 3x moves that
                    # far again 3x faster (bigger per-tick step), 0x holds still.
                    current = self.flipper_positions.get(key, 0.0)
                    scaled = current + self.command_scale * (raw - current)
                msg = Float64()
                msg.data = float(limit(key, scaled))
                pubs[key].publish(msg)

            self.get_logger().info(
                f"CMD({self._rl_baseline_module_name}): vel=(user-driven, not published) "
                f"flipper_{rl_action.flipper_mode}=[{rl_action.flippers[0]:.2f}, {rl_action.flippers[1]:.2f}, "
                f"{rl_action.flippers[2]:.2f}, {rl_action.flippers[3]:.2f}]",
                throttle_duration_sec=0.5,
            )
            self.publish_policy_action_markers(twist=None, flipper_values=rl_action.flippers, flipper_mode=rl_action.flipper_mode)
            self.publish_flipper_command_hud(rl_action.flippers, rl_action.flipper_mode)
            self.publish_heightmap_extent_marker()
            if self._rl_baseline_module_name == "creps":
                self.publish_creps_debug_markers()
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
            twist.angular.z = 0.0 if self.disable_angular_output else float(action[1])
            self.cmd_vel_pub.publish(twist)

            # FTR's flipper action is NOT rad/s: it is integrated as
            # `action * flipper_dt` degrees per control step (flipper_dt = 5), so a
            # unit action means 5 deg/step = 5*control_rate deg/s. Converting here
            # keeps the deployed flipper rate equal to the trained one; publishing
            # the raw action as rad/s (what this used to do) ran the flippers ~15%
            # fast at 10 Hz and would be off by the full ratio at any other rate.
            ftr_rate = self.policy.FLIPPER_RAD_PER_STEP_PER_ACTION / self.dt
            flipper_velocities = action[2:6] * ftr_rate * self.flipper_velocity_scale
            flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
            for i, key in enumerate(flipper_keys):
                msg = Float64()
                msg.data = self._limit_flipper_velocity(key, float(flipper_velocities[i]))
                self.flipper_pubs[key].publish(msg)

            self.get_logger().info(
                f"CMD(ftr): vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
                f"flipper_vel=[{flipper_velocities[0]:.2f}, {flipper_velocities[1]:.2f}, "
                f"{flipper_velocities[2]:.2f}, {flipper_velocities[3]:.2f}]",
                throttle_duration_sec=0.5,
            )
            self.publish_policy_action_markers(twist=twist, flipper_values=flipper_velocities, flipper_mode="vel")
            self.publish_flipper_command_hud(flipper_velocities, "vel")
            self.publish_heightmap_extent_marker()
        else:
            # Native action: [track_fl, track_fr, track_rl, track_rr, flip_fl, flip_fr, flip_rl, flip_rr] — 8-D
            track_velocities = action[:4]
            flipper_velocities = action[4:8] * self.flipper_velocity_scale

            twist = self.track_velocities_to_twist(track_velocities)
            if self.disable_angular_output:
                twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

            flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
            for i, key in enumerate(flipper_keys):
                msg = Float64()
                msg.data = self._limit_flipper_velocity(key, float(flipper_velocities[i]))
                self.flipper_pubs[key].publish(msg)

            self.get_logger().info(
                f"CMD: vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
                f"flipper_vel=[{flipper_velocities[0]:.2f}, {flipper_velocities[1]:.2f}, "
                f"{flipper_velocities[2]:.2f}, {flipper_velocities[3]:.2f}]",
                throttle_duration_sec=0.5,
            )
            self.publish_policy_action_markers(twist=twist, flipper_values=flipper_velocities, flipper_mode="vel")
            self.publish_flipper_command_hud(flipper_velocities, "vel")


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
        # Always stop the robot on the way out (SIGINT, SIGTERM, or an unhandled
        # exception in the spin loop) -- see stop_all()'s docstring for why this
        # matters: without it, overriding/killing this node leaves the robot driving
        # on its last command forever.
        node.stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
