#!/usr/bin/env python3
"""
ROS2 node for deploying the trained flipper control policy on the MARV robot.

Subscribes to robot state and elevation map, publishes track velocities and flipper commands.
"""
import os, sys
# Force Python to load the venv packages BEFORE the ROS 2 system packages -- but ONLY when
# this node was launched with a bare interpreter. When it is already running inside a
# virtualenv (marv_flipper_eval launches it as "$VENV/bin/python", VENV=marv_venv),
# prepending ~/.venv shadows the venv that was deliberately chosen: the two carry different
# torchrl majors (0.11.1 vs 0.8.1), and loading a checkpoint under the wrong one fails in
# ways that look like a broken policy rather than a broken path. sys.prefix diverges from
# sys.base_prefix exactly when a venv is active.
if sys.prefix == sys.base_prefix:
    venv_site = os.path.expanduser('~/.venv/lib/python3.12/site-packages')
    if os.path.isdir(venv_site):
        if venv_site in sys.path:
            sys.path.remove(venv_site)
        sys.path.insert(0, venv_site)
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, PointCloud2, PointField, Imu, Image as RosImage
from geometry_msgs.msg import PoseStamped, Point, Twist
from std_msgs.msg import Bool, Float64, MultiArrayDimension, MultiArrayLayout, Float32MultiArray
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as TimeMsg
import math
import struct

# The RViz configs shipped with marv_flipper_eval (rl_generic/azayev/pan_d3qn/creps)
# display the debug decorations published below. They live here, in the node that
# actually runs, rather than in the flipper_eval workspace's own copy of this file --
# that copy is no longer the one launched (run_flipper_policy_sim.sh execs THIS file),
# which is why those displays sat empty with "Ok" status and no data.
try:
    import cv2
except ImportError:  # HUD/colour image degrade to no-ops; markers are unaffected
    cv2 = None

import matplotlib
matplotlib.use('Agg')  # Forces Matplotlib to run in headless/thread-safe mode

import torch
from scipy.spatial.transform import Rotation


class FlipperPolicyNode(Node):
    """ROS2 node that runs the trained flipper control policy."""

    # Flipper joint name mapping (order expected by policy)
    FLIPPER_NAMES = ["front_left_flipper_j", "front_right_flipper_j", "rear_left_flipper_j", "rear_right_flipper_j"]

    # Track width for differential drive conversion (meters)
    TRACK_WIDTH = 0.36  # From robot config rover_bodyWidth

    # Cells of the window that never reach the observation are painted this flat
    # grey in the debug images (see _policy_view).
    _UNUSED_GREY = 60

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
        self.declare_parameter("track_velocity_scale", 1.0)   # Scale factor for FTR track velocity commands
        self.declare_parameter("publish_debug_cloud", True)  # Publish heightmap as point cloud for debugging
        self.declare_parameter("disable_turning", False)  # Force angular velocity to 0
        # /policy_heightmap_img orientation. True (default) = drawn the way the world looks,
        # so the panel lines up with /policy_heightmap_debug and Gazebo. False = the raw
        # policy-convention observation, which is deliberately mirrored left/right (training
        # puts col 0 at -y; see _publish_heightmap_image). This only affects the picture —
        # the policy builds its own observation in infer_action and never sees this.
        self.declare_parameter("heightmap_image_world_orientation", True)
        # Publish /cmd_vel at all. Off by default: this node is flipper control only and
        # the operator/autodrive owns track velocity (see the block in the policy setup).
        self.declare_parameter("publish_cmd_vel", False)
        # ---- robot branch ----------------------------------------------------
        # I/O topics are parameters with REAL-ROBOT defaults on this branch
        # (main hardcodes the sim names; run_flipper_policy_sim.sh passes these
        # explicitly in both profiles, so the defaults only matter for a bare
        # `ros2 run`). /icp_odom is the NUC's norlab ICP odometry -- there is
        # no /ground_truth_odom on hardware.
        self.declare_parameter("odom_topic", "/icp_odom")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("joint_topic", "/marv/joint_states")
        self.declare_parameter("elevation_topic", "/elevation_map_filtered")
        # Hard ceiling on |/cmd_vel.linear.x| [m/s] whenever publish_cmd_vel is
        # on. Carried over from the retired rodeo_rl_ws deployment (0.3 there);
        # a policy echoing its training-time fixed forward speed must never be
        # able to command more on hardware.
        self.declare_parameter("max_linear_velocity", 0.3)
        # Deadman + estop gate (the C++ reactive controller has the same gate,
        # Control.cpp:874-908; an RL node commanding actuators must not be the
        # one path without it). require_deadman=false is for bag replay /
        # bench runs only -- leave true on the robot.
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("deadman_topic", "/marv/teleop/deadman")
        self.declare_parameter("estop_topic", "/marv/estop")
        self.declare_parameter("deadman_timeout_sec", 0.2)
        # Previously passed by run_flipper_policy_sim.sh but never declared --
        # silent no-ops. disable_angular_output is the launch-facing alias of
        # disable_turning; auto_goal_on_release gates the one-shot
        # goal-ahead-on-override-release below.
        self.declare_parameter("disable_angular_output", False)
        self.declare_parameter("auto_goal_on_release", False)
        self.declare_parameter("auto_goal_ahead_m", 5.0)

        # Get parameters
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        policy_weights_path = self.get_parameter("policy_weights_path").get_parameter_value().string_value
        vecnorm_weights_path = self.get_parameter("vecnorm_weights_path").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value
        self.heightmap_decay = self.get_parameter("heightmap_decay").get_parameter_value().double_value
        self.heightmap_layer = self.get_parameter("heightmap_layer").get_parameter_value().string_value
        self.flipper_velocity_scale = self.get_parameter("flipper_velocity_scale").get_parameter_value().double_value
        self.track_velocity_scale = self.get_parameter("track_velocity_scale").get_parameter_value().double_value
        self.publish_debug_cloud = self.get_parameter("publish_debug_cloud").get_parameter_value().bool_value
        self.heightmap_image_world_orientation = self.get_parameter(
            "heightmap_image_world_orientation"
        ).get_parameter_value().bool_value
        self.disable_turning = (
            self.get_parameter("disable_turning").get_parameter_value().bool_value
            or self.get_parameter("disable_angular_output").get_parameter_value().bool_value)
        self.max_linear_velocity = abs(
            self.get_parameter("max_linear_velocity").get_parameter_value().double_value)
        self.require_deadman = self.get_parameter("require_deadman").get_parameter_value().bool_value
        self.deadman_timeout_sec = self.get_parameter("deadman_timeout_sec").get_parameter_value().double_value
        self.auto_goal_on_release = self.get_parameter("auto_goal_on_release").get_parameter_value().bool_value
        self.auto_goal_ahead_m = self.get_parameter("auto_goal_ahead_m").get_parameter_value().double_value

        if not config_path or not policy_weights_path:
            self.get_logger().error("config_path and policy_weights_path parameters are required!")
            raise ValueError("Missing required parameters")

        # Dispatch by env_cfg_overrides.module_name — authoritative and present in every
        # FTR-family config, unlike the old _detect_ftr_config() heuristic (task/
        # ftr_obs_encoder_opts presence), which is true for ALL FTR configs regardless of
        # actual policy family and mis-routed atd3qn/icmd3qn/creps into the PPO path.
        module_name = self._detect_module_name(config_path)
        if module_name in ("marv_rl", "hfc"):
            # hfc is a distinct module from hfcil (the separately-trained IL classifier,
            # deployed via marv_flipper_eval's dedicated hfcil_policy_node.py) — hfc is a
            # PPO-trained actor (HFCActionDecoder bakes in Eq. 7/8/9's composite action
            # already) and loads through the same generic FtrPolicyInferenceModule path
            # as marv_rl, via its own policy_config in config.yaml.
            self._kind = "ppo_ftr"
        elif module_name in ("atd3qn", "icmd3qn"):
            self._kind = "d3qn"
        elif module_name == "creps":
            self._kind = "creps"
        elif module_name == "ctrac":
            self._kind = "ctrac"
        elif module_name == "mitriakov":
            raise RuntimeError(
                f"policy module '{module_name}' is not supported by this node — mitriakov "
                "needs step-edge perception never implemented here."
            )
        else:
            self._kind = "native"
        # Backward-compat flag some helper methods still branch on. ctrac rides the same
        # path as the PPO/FTR families: it needs the goal vector (so the goal-carrying
        # _control_callback, not _control_callback_no_goal), emits the same 6-D
        # [v, w, fl, fr, rl, rr], and — being flipper_control_mode: velocity with
        # flipper_dt=5 deg/step — wants exactly the incremental-position-to-velocity
        # conversion that branch already does.
        self._is_ftr = self._kind in ("ppo_ftr", "ctrac")

        # A policy trained behind a FIXED forward speed never learned to control velocity:
        # its track command is a constant the reproduction supplies, not a network output
        # (AT-D3QN/ICM-D3QN state this outright — the paper assumes an operator or planner
        # drives the tracks and the network only moves flippers). Publishing that constant
        # on /cmd_vel would fight the operator/autodrive, so on the robot we stay off the
        # topic entirely and emit flipper commands only.
        #
        # Detected from env_cfg_overrides.fixed_forward_vel rather than from self._kind,
        # because it does not follow the policy family: marv_config_marv_rl_set_vel.yaml
        # pins 0.5 while marv_config_marv_rl.yaml learns velocity, and both are "ppo_ftr".
        # Configs pinning it today: atd3qn, icmd3qn, creps, hfc, mitriakov, marv_rl_set_vel.
        # This node is FLIPPER CONTROL ONLY: /cmd_vel is not published for any policy.
        # Track velocity belongs to the operator / autodrive, which owns the topic; a policy
        # publishing into it at the control rate would fight whatever is actually driving.
        #
        # `publish_cmd_vel` re-enables it, but note that for a policy trained behind a fixed
        # forward speed the velocity output is not a network decision at all — it is the
        # constant the reproduction fed in during training, echoed back (AT-D3QN/ICM-D3QN
        # say so outright: the paper assumes an operator or planner drives the tracks).
        # _fixed_forward_vel records which case this policy is, for the log below.
        self._fixed_forward_vel = self._detect_fixed_forward_vel(config_path)
        self.publish_cmd_vel = self.get_parameter("publish_cmd_vel").get_parameter_value().bool_value
        if not self.publish_cmd_vel:
            self.get_logger().info(
                "/cmd_vel disabled — flipper control only. Drive the tracks from the "
                "operator/autodrive. Set publish_cmd_vel:=true to override."
            )
        elif self._fixed_forward_vel is not None:
            self.get_logger().warn(
                f"publish_cmd_vel:=true, but this policy trained with "
                f"fixed_forward_vel={self._fixed_forward_vel} m/s — it never learned velocity "
                "control, so it will publish that constant regardless of state."
            )
        self.get_logger().info(f"Loading '{self._kind}' policy (module_name={module_name!r}) from {config_path}")

        if self._kind == "ppo_ftr":
            from marv_rl_training.training.ftr_policy_inference_module import FtrPolicyInferenceModule
            from omegaconf import OmegaConf
            _ftr_cfg = OmegaConf.load(config_path)
            _max_deg = _ftr_cfg.get("env_cfg_overrides", {}).get("flipper_pos_max_deg", 90.0)
            self._ftr_joint_limit_rad = float(np.deg2rad(_max_deg))
            self.get_logger().info(f"FTR flipper limit: ±{_max_deg}° (±{self._ftr_joint_limit_rad:.3f} rad)")
            self.policy = FtrPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                vecnorm_weights_path=vecnorm_weights_path if vecnorm_weights_path else None,
                device=device,
            )
        elif self._kind == "d3qn":
            from marv_rl_training.training.d3qn_policy_inference_module import D3QNPolicyInferenceModule
            self.policy = D3QNPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                device=device,
            )
        elif self._kind == "ctrac":
            from marv_rl_training.training.ctrac_policy_inference_module import CTRACPolicyInferenceModule
            self.policy = CTRACPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
                device=device,
            )
            # The FTR publish branch clamps against a single symmetric ±_ftr_joint_limit_rad,
            # but ctrac trains with MARV's ASYMMETRIC per-corner limits (front -60/+80,
            # rear -80/+60). Use the tightest magnitude so the scalar clamp can never
            # command past a real joint limit; _clamp_flipper_limits below applies the
            # true per-corner interval on top.
            self._ftr_joint_limit_rad = float(np.min(np.abs(self.policy.flipper_limits_rad)))
            # Per-corner limits in the RAW ROS frame, which is what self.flipper_positions
            # reports and what the publish branch compares against. Negating a front row
            # also REVERSES it: FTR [-front_up, +front_down] becomes ROS [-front_down,
            # +front_up]. Swapping is not optional — these are read as (low, high) and an
            # inverted interval would clamp against nothing.
            _lim = np.asarray(self.policy.flipper_limits_rad, dtype=np.float64).copy()
            _lim[:2] = -_lim[:2, ::-1]
            self._ftr_limits_ros = _lim
        elif self._kind == "creps":
            from marv_rl_training.training.creps_policy_inference_module import CREPSPolicyInferenceModule
            self.policy = CREPSPolicyInferenceModule(
                config_path=config_path,
                policy_weights_path=policy_weights_path,
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

        # Subscribers (topics are parameters on this branch; both odom sources
        # -- sim GT and the robot ICP -- publish RELIABLE, so we must match)
        _odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        _imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        _joint_topic = self.get_parameter("joint_topic").get_parameter_value().string_value
        _elevation_topic = self.get_parameter("elevation_topic").get_parameter_value().string_value
        self.get_logger().info(
            f"I/O: odom={_odom_topic} imu={_imu_topic} joints={_joint_topic} "
            f"elevation={_elevation_topic} (layer {self.heightmap_layer})")
        self.odom_sub = self.create_subscription(Odometry, _odom_topic, self.odom_callback, reliable_qos)
        self.imu_sub = self.create_subscription(Imu, _imu_topic, self.imu_callback, sensor_qos)
        self.joint_state_sub = self.create_subscription(JointState, _joint_topic, self.joint_state_callback, sensor_qos)
        # Goal uses VOLATILE to accept messages from any publisher (RViz, ros2 topic pub, etc.)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.goal_reset_sub = self.create_subscription(PoseStamped, "/goal_reset", self.goal_reset_callback, 10)
        self.elevation_map_sub = self.create_subscription(GridMap, _elevation_topic, self.elevation_map_callback, reliable_qos)
        # Deadman + estop gate state (see control_callback). A deadman message
        # older than deadman_timeout_sec counts as NOT held.
        self._deadman_held = False
        self._deadman_rx_time = None
        self._estop_latched = False
        self.create_subscription(
            Bool, self.get_parameter("deadman_topic").get_parameter_value().string_value,
            self._on_deadman, 10)
        self.create_subscription(
            Bool, self.get_parameter("estop_topic").get_parameter_value().string_value,
            self._on_estop, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.flipper_pubs = {
            "front_left": self.create_publisher(Float64, "/flippers_cmd_vel/front_left", 10),
            "front_right": self.create_publisher(Float64, "/flippers_cmd_vel/front_right", 10),
            "rear_left": self.create_publisher(Float64, "/flippers_cmd_vel/rear_left", 10),
            "rear_right": self.create_publisher(Float64, "/flippers_cmd_vel/rear_right", 10),
        }
        # FTR policy outputs position increments — use relative position commands
        self.flipper_pos_rel_pubs = {
            "front_left": self.create_publisher(Float64, "/flippers_cmd_pos_rel/front_left", 10),
            "front_right": self.create_publisher(Float64, "/flippers_cmd_pos_rel/front_right", 10),
            "rear_left": self.create_publisher(Float64, "/flippers_cmd_pos_rel/rear_left", 10),
            "rear_right": self.create_publisher(Float64, "/flippers_cmd_pos_rel/rear_right", 10),
        }
        # D3QN/CREPS policies output ABSOLUTE radian joint targets — same topic naming
        # hfcil_state_pose_driver.py/hfcil_policy_node.py (marv_flipper_eval) already
        # publish to successfully.
        self.flipper_pos_pubs = {
            "front_left": self.create_publisher(Float64, "/flippers_cmd_pos/front_left", 10),
            "front_right": self.create_publisher(Float64, "/flippers_cmd_pos/front_right", 10),
            "rear_left": self.create_publisher(Float64, "/flippers_cmd_pos/rear_left", 10),
            "rear_right": self.create_publisher(Float64, "/flippers_cmd_pos/rear_right", 10),
        }

        # Debug visualization publishers
        self.heightmap_cloud_pub = self.create_publisher(PointCloud2, "/policy_heightmap_debug", 10)
        self.heightmap_gridmap_pub = self.create_publisher(GridMap, "/policy_heightmap", 10)
        self.action_debug_pub = self.create_publisher(Float32MultiArray, "/policy_action_debug", 10)
        self.heightmap_img_pub = self.create_publisher(RosImage, "/policy_heightmap_img", 10)
        # Same heightmap window as /policy_heightmap_img, colour-mapped and scaled up:
        # this is the topic name every marv_flipper_eval RViz config asks for.
        self.heightmap_image_pub = self.create_publisher(RosImage, "/policy_heightmap_image", 10)
        # Top-down flipper-command HUD: 4 spots (FL/FR/RL/RR) each showing an up/down
        # arrow or "-" for still, front-to-the-left -- a glance readout of what the
        # policy commands right now, independent of the 3D view's camera angle.
        self.flipper_cmd_hud_pub = self.create_publisher(RosImage, "/policy_flipper_command_hud", 10)
        # INPUTS the policy just consumed: goal-direction arrow + roll/pitch text.
        self.policy_obs_markers_pub = self.create_publisher(MarkerArray, "/policy_obs_markers", 10)
        # OUTPUTS: velocity arrow + flipper-command text.
        self.policy_action_markers_pub = self.create_publisher(MarkerArray, "/policy_action_markers", 10)
        # The 45x21 @ 0.05 m window (ftr_heightmap_window) drawn as a rectangle on the
        # ground around the robot -- what "the policy's heightmap input" really spans.
        self.heightmap_extent_pub = self.create_publisher(MarkerArray, "/policy_heightmap_extent", 10)

        # /flipper_override: LISTEN ONLY -- this node never raises it. The flag
        # belongs to the three things that legitimately take the flippers away from
        # a running policy: the UI/gamepad OVERRIDE button, the gap-zone auto-arm
        # (pad_extra_controls' neutral strips) and autodrive's unstick manoeuvre
        # (stuck_override). A policy spawns RUNNING and is muted only by those.
        #
        # Subscribing matters because Gazebo's FlipperControlPlugin gates POSITION
        # commands only, letting /flippers_cmd_vel/* and /cmd_vel through as "the
        # manual path" -- so without this, pressing OVERRIDE would mute d3qn/creps
        # (position targets) but leave a velocity-mode policy commanding the
        # flippers straight past the operator.
        self._flipper_override_active = False
        self.create_subscription(Bool, "/flipper_override", self._on_flipper_override, 10)

        # Live 0..5x dial on the policy's flipper output, driven by the CommandScale
        # RViz panel (marv_rviz_panels/command_scale_panel.cpp). These policies publish
        # flipper commands only -- track speed belongs to the operator/autodrive -- so
        # scaling that output is the one lever that slows down or speeds up what they
        # visibly do. 1.0 = the policy's own output, 0.0 = frozen.
        #
        # TRANSIENT_LOCAL to match the panel's own QoS(4).transient_local(): the panel
        # is usually up before the policy relaunches, and a VOLATILE subscription
        # silently misses the already-published slider value, so the dial reads 3x
        # while the node runs at 1x until someone touches the slider again.
        self.command_scale = 1.0
        self.create_subscription(
            Float64, "/policy_command_scale", self._on_command_scale,
            QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

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
    def goal_reset_callback(self, msg: PoseStamped):
        self.current_goal = None

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

            # grid_map_ros packs the layer as the Eigen matrix in COLUMN-major order with
            # `rows` rows, so the matrix is recovered with reshape((rows, cols), order="F").
            # This previously read reshape((cols, rows), order="F"), which happens to be
            # identical for a SQUARE map (the usual elevation_mapping case, so it went
            # unnoticed) but silently transposes — or raises — as soon as length_x !=
            # length_y. dim[0] is "column_index" and dim[1] is "row_index", hence the
            # cols/rows names above.
            heightmap = data.reshape((rows, cols), order='F')

            # Apply circular buffer start indices if present
            if msg.outer_start_index != 0 or msg.inner_start_index != 0:
                heightmap = np.roll(heightmap, -msg.outer_start_index, axis=0)
                heightmap = np.roll(heightmap, -msg.inner_start_index, axis=1)

            # No transpose: the reshape above already yields grid_map's [row, col] matrix,
            # with row 0 = x_max (front) and col 0 = y_max (the robot's LEFT). Verified in
            # grid_map_core/GridMapMath.cpp: transformBufferOrderToMapFrame returns
            # {-index[0], -index[1]}, so index (0,0) is the front-left cell.

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

        if not hasattr(self, '_pointcloud_save_counter'):
                self._pointcloud_save_counter = 0
        self._pointcloud_save_counter += 1
        if self._pointcloud_save_counter % 100 == 1:
            if self._pointcloud_save_counter % 100 == 1:
                try:
                    # BYPASS MATPLOTLIB ENTIRELY
                    np.save('/tmp/pointcloud_debug.npy', points)
                    self.get_logger().info('Saved raw pointcloud data to /tmp/pointcloud_debug.npy')
                except Exception as e:
                    self.get_logger().warn(f'Could not save debug pointcloud data: {e}')

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
        twist.linear.x = float(np.clip((left_vel + right_vel) / 2.0,
                                       -self.max_linear_velocity, self.max_linear_velocity))
        twist.angular.z = float((right_vel - left_vel) / self.TRACK_WIDTH)

        return twist

    def control_callback(self):
        """Main control loop - infer action and publish commands."""
        if self._flipper_override_active:
            # Overridden: publish NOTHING (not even zeros -- that would fight the
            # manual/autodrive stream that now owns these topics). stop_all() ran
            # once on the rising edge.
            return

        # Deadman/estop gate (robot branch): without the operator's deadman
        # held (or with the estop latched / deadman stale) this node commands
        # nothing but zeros -- velocity zeros hold the flippers, position
        # targets are left alone. Mirrors the reactive controller's own gate.
        if not self._actuation_allowed():
            self.stop_all()
            self.get_logger().warn(
                "actuation gated: "
                + ("ESTOP latched" if self._estop_latched else "deadman not held/stale")
                + " -- commanding zeros (require_deadman:=false only for bag replay)",
                throttle_duration_sec=2.0)
            return

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
        quat = self.get_orientation_quat()

        # D3QN/CREPS don't use a goal vector at all (constant forward speed, no steering)
        # — drive them from a dedicated, unconditional branch instead of the goal-gated
        # PPO/native flow below.
        if self._kind in ("d3qn", "creps"):
            self._control_callback_no_goal(thetas, quat)
            return

        goal_vec_local = self.get_goal_vector_local()
        velocities = self.get_velocities_local()

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

        # Goal-reached check (2D distance, ignoring height)
        if goal_vec_local is not None and np.linalg.norm(goal_vec_local[:2]) < 0.5:
            self.get_logger().info("Goal reached (< 0.5 m). Stopping.")
            self.current_goal = None
            goal_vec_local = None

        # Check required inputs
        if goal_vec_local is None:
            # No goal set - publish zero velocities to stand in place
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.publish_cmd_vel:
                self.cmd_vel_pub.publish(twist)
            # Also publish zero flipper velocities
            for key in ["front_left", "front_right", "rear_left", "rear_right"]:
                msg = Float64()
                msg.data = 0.0
                self.flipper_pubs[key].publish(msg)
            self.publish_policy_debug_views(
                twist if self.publish_cmd_vel else None, [0.0] * 4, "vel",
                goal_vec_local=None)
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
                robot_z=float(self.current_odom.pose.pose.position.z) if self.current_odom is not None else 0.0,
            )
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        # Parse action and publish commands
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float64)

        # Publish raw action for debug UI (before any scaling/sign compensation)
        debug_msg = Float32MultiArray()
        debug_msg.data = action.astype(np.float32).tolist()
        self.action_debug_pub.publish(debug_msg)

        # Publish heightmap as grayscale image for debug UI
        self._publish_heightmap_image(self.current_heightmap)

        twist = Twist()
        flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
        if self._is_ftr:
            # FTR: 6-D [v, w, fl, fr, rl, rr]
            # Track: raw output is already in m/s and rad/s (clamped to 0.95 m/s, 1.0 rad/s in training)
            twist.linear.x = float(np.clip(action[0] * self.track_velocity_scale,
                                           -self.max_linear_velocity, self.max_linear_velocity))
            twist.angular.z = 0.0 if self.disable_turning else float(action[1] * self.track_velocity_scale)
            if self.publish_cmd_vel:
                self.cmd_vel_pub.publish(twist)
            # Flippers: FTR policy outputs incremental position (degrees/step).
            # Training (this config): sim_dt=0.005, decimation=5 → control rate = 40 Hz.
            # flipper_dt=5 deg/step → max = 5 deg × 40 Hz × π/180 = 3.49 rad/s.
            # Convert to velocity: action × 5_deg_per_step × control_rate × π/180.
            # control_rate must match training: set via launch arg or compute from sim_dt×decimation.
            # Apply FTR front-flipper sign convention (same as FtrWheelArticulation: [-1,-1,1,1])
            flipper_compensation = np.array([-1.0, -1.0, 1.0, 1.0])
            flipper_vel = (action[2:6] * flipper_compensation * np.deg2rad(5.0)
                           * (1.0 / self.dt) * self.flipper_velocity_scale * self.command_scale)

            # Enforce position limits. Symmetric ±flipper_pos_max_deg for the PPO/FTR
            # configs; ctrac carries MARV's asymmetric per-corner interval instead, so use
            # that when the active module supplied one.
            _limits = getattr(self, "_ftr_limits_ros", None)
            for i, key in enumerate(flipper_keys):
                pos = self.flipper_positions[key]
                if _limits is not None:
                    lo, hi = float(_limits[i, 0]), float(_limits[i, 1])
                else:
                    lo, hi = -self._ftr_joint_limit_rad, self._ftr_joint_limit_rad
                if pos >= hi and flipper_vel[i] > 0:
                    flipper_vel[i] = 0.0
                elif pos <= lo and flipper_vel[i] < 0:
                    flipper_vel[i] = 0.0

            for i, key in enumerate(flipper_keys):
                msg = Float64()
                msg.data = float(flipper_vel[i])
                self.flipper_pubs[key].publish(msg)
            self.get_logger().info(
                f"CMD: vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
                f"flipper_vel=[{flipper_vel[0]:.2f}, {flipper_vel[1]:.2f}, "
                f"{flipper_vel[2]:.2f}, {flipper_vel[3]:.2f}] rad/s",
                throttle_duration_sec=0.5,
            )
            self.publish_policy_debug_views(
                twist if self.publish_cmd_vel else None, flipper_vel, "vel",
                goal_vec_local=goal_vec_local, roll_deg=roll, pitch_deg=pitch)
        else:
            # Native: 8-D [track_vl, track_vr, track_vl2, track_vr2, fl, fr, rl, rr]
            track_velocities = action[:4]
            twist = self.track_velocities_to_twist(track_velocities)
            if self.disable_turning:
                twist.angular.z = 0.0
            flipper_velocities = action[4:8] * self.flipper_velocity_scale * self.command_scale
            if self.publish_cmd_vel:
                self.cmd_vel_pub.publish(twist)
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
            self.publish_policy_debug_views(
                twist if self.publish_cmd_vel else None, flipper_velocities, "vel",
                goal_vec_local=goal_vec_local, roll_deg=roll, pitch_deg=pitch)


    def _control_callback_no_goal(self, thetas: np.ndarray | None, quat: np.ndarray | None) -> None:
        """D3QN/CREPS control path: no goal vector, constant forward speed, absolute
        radian joint targets published to /flippers_cmd_pos/*."""
        if thetas is None or self.current_odom is None:
            return
        robot_z = float(self.current_odom.pose.pose.position.z)

        try:
            action = self.policy.infer_action(
                heightmap=self.current_heightmap,
                heightmap_extent=self.heightmap_extent,
                robot_z=robot_z,
                thetas=thetas,
                quat=quat,
            )
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float64)

        debug_msg = Float32MultiArray()
        debug_msg.data = action.astype(np.float32).tolist()
        self.action_debug_pub.publish(debug_msg)
        self._publish_heightmap_image(self.current_heightmap)

        twist = Twist()
        twist.linear.x = float(np.clip(action[0], -self.max_linear_velocity,
                                       self.max_linear_velocity))
        twist.angular.z = 0.0 if self.disable_turning else float(action[1])
        if self.publish_cmd_vel:
            self.cmd_vel_pub.publish(twist)

        flipper_keys = ["front_left", "front_right", "rear_left", "rear_right"]
        targets = np.array([self._scaled_flipper_target(key, action[2 + i])
                            for i, key in enumerate(flipper_keys)])
        for i, key in enumerate(flipper_keys):
            msg = Float64()
            msg.data = float(targets[i])
            self.flipper_pos_pubs[key].publish(msg)

        self.get_logger().info(
            f"CMD ({self._kind}): vel=({twist.linear.x:.2f}, {twist.angular.z:.2f}) "
            f"flipper_pos=[{targets[0]:.2f}, {targets[1]:.2f}, {targets[2]:.2f}, {targets[3]:.2f}] rad"
            + (f" (scale {self.command_scale:.2f}x)" if self.command_scale != 1.0 else ""),
            throttle_duration_sec=0.5,
        )
        # These families steer by nothing but the heightmap: no goal arrow to draw,
        # and the flipper values are absolute targets, not rates.
        roll_deg = pitch_deg = None
        if quat is not None:
            roll_deg, pitch_deg, _ = Rotation.from_quat(
                [quat[0], quat[1], quat[2], quat[3]]).as_euler('xyz', degrees=True)
        self.publish_policy_debug_views(
            twist if self.publish_cmd_vel else None, targets, "position",
            goal_vec_local=None, roll_deg=roll_deg, pitch_deg=pitch_deg)

    def _publish_heightmap_image(self, hmap: np.ndarray) -> None:
        """Publish the policy's heightmap window for RViz.

        Display only — this is a dead end. The policy builds its own observation inside
        infer_action (ftr_policy_inference_module._build_obs), from the same raw
        `current_heightmap` but independently of anything done here, so no choice made in
        this function can affect what the robot does.

        What is drawn is the real FTR window (2.25 x 1.05 m cropped and resampled to 45x21,
        via ftr_heightmap_window), not the whole map squashed with a bare cv2.resize as it
        used to be. Only the lateral direction is a display choice:

          heightmap_image_world_orientation = True  (default)
              Mirrored back to world orientation: row 0 (top) = FRONT, col 0 (left) = the
              robot's LEFT. Lines up with /policy_heightmap_debug and Gazebo, so the panel
              can be compared against reality at a glance.
          heightmap_image_world_orientation = False
              The observation exactly as the network receives it: col 0 = the robot's
              RIGHT, because training's MapHelper.get_obs + .flip(0) put col 0 at -y. This
              looks mirrored on screen, and that is correct.

        The mirror between the two conventions is what
        marv_rl_training/training/test_heightmap_orientation.py pins; it is not something
        this panel can verify either way.
        """
        hmap, used = self._policy_view(hmap)
        if hmap is None or hmap.size == 0:
            return
        if self.heightmap_image_world_orientation:
            # Undo training's lateral mirror for display only (see the docstring).
            hmap = hmap[:, ::-1]
            if used is not None:
                used = used[:, ::-1]
        h, w = hmap.shape
        # Contrast is set by the cells the policy actually reads: with creps'
        # single cell, stretching over the whole window would spend the entire
        # colour range on terrain the linear law never looks at.
        sampled = hmap[used] if used is not None and used.any() else hmap
        lo, hi = float(sampled.min()), float(sampled.max())
        norm = np.clip((hmap - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        if used is not None:
            norm[~used] = self._UNUSED_GREY
        msg = RosImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = h
        msg.width = w
        msg.encoding = "mono8"
        msg.step = w
        msg.data = norm.tobytes()
        self.heightmap_img_pub.publish(msg)

        # Same window, colour-mapped and blown up nearest-neighbour so individual
        # cells stay countable in an RViz Image panel (45x21 is a few pixels tall
        # otherwise). This is the topic the marv_flipper_eval RViz configs open.
        if cv2 is not None:
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            if used is not None:
                colored[~used] = (self._UNUSED_GREY,) * 3   # flat grey reads as "not an input"
            scale = max(1, int(400 // max(colored.shape[0], colored.shape[1])))
            if scale > 1:
                colored = cv2.resize(colored, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_NEAREST)
            if used is not None and used.any():
                # Outline the sampled region. Without it creps' single cell is 8x8
                # px in a 168x360 field of grey and simply cannot be found.
                rows, cols = np.where(used)
                cv2.rectangle(colored,
                              (int(cols.min()) * scale, int(rows.min()) * scale),
                              ((int(cols.max()) + 1) * scale - 1, (int(rows.max()) + 1) * scale - 1),
                              (0, 255, 255), max(1, scale // 6))
            color_msg = self._bgr_image_msg(colored)
            color_msg.header.stamp = msg.header.stamp
            self.heightmap_image_pub.publish(color_msg)

    # ------------------------------------------------------------------
    # RViz decorations (marv_flipper_eval's rl_generic/azayev/pan_d3qn/creps
    # configs). Display only -- nothing here feeds back into the policy.
    # ------------------------------------------------------------------
    @staticmethod
    def _viz_stamp() -> TimeMsg:
        """Stamp for base_link-attached decorations: 0 == "latest available
        transform".

        These markers are rigidly attached to base_link, so RViz should draw them
        against the newest base_link pose it has. Stamping with clock-now instead
        asks for a TF time that does not exist yet (map->odom is published at
        ~20 Hz, so "now" is ahead of the newest sample) and the lookup fails with
        "Lookup would require extrapolation into the future", turning the display
        red in a map-fixed-frame view.
        """
        return TimeMsg()

    def publish_policy_obs_markers(self, goal_vec_local, roll_deg=None, pitch_deg=None):
        """/policy_obs_markers: the goal-direction arrow built from the SAME
        goal_vec_local the policy just consumed, plus a roll/pitch readout."""
        markers = MarkerArray()
        now = self._viz_stamp()

        arrow = Marker()
        arrow.header.frame_id = "base_link"
        arrow.header.stamp = now
        arrow.ns = "policy_obs"
        arrow.id = 0
        if goal_vec_local is not None:
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                Point(x=0.0, y=0.0, z=0.3),
                Point(x=float(goal_vec_local[0]), y=float(goal_vec_local[1]),
                      z=float(goal_vec_local[2]) + 0.3),
            ]
            arrow.scale.x = 0.06   # shaft diameter
            arrow.scale.y = 0.12   # head diameter
            arrow.scale.z = 0.0
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (0.1, 0.9, 0.2, 0.9)
        else:
            # No goal (or a goal-free family like d3qn/creps): drop any previous
            # arrow rather than leaving a stale one pointing at the last goal.
            arrow.action = Marker.DELETE
        markers.markers.append(arrow)

        if roll_deg is not None and pitch_deg is not None:
            text = Marker()
            text.header.frame_id = "base_link"
            text.header.stamp = now
            text.ns = "policy_obs"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.z = 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.15
            text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 1.0, 1.0, 0.9)
            text.text = f"roll={roll_deg:.1f} deg  pitch={pitch_deg:.1f} deg"
            markers.markers.append(text)

        self.policy_obs_markers_pub.publish(markers)

    def publish_policy_action_markers(self, twist, flipper_values, flipper_mode: str):
        """/policy_action_markers: what is being commanded right now -- a blue
        velocity arrow (distinct from the green goal arrow) plus a text readout.
        twist=None means /cmd_vel is not being published at all (publish_cmd_vel
        off: the speed is user-driven, not the policy's)."""
        markers = MarkerArray()
        now = self._viz_stamp()

        arrow = Marker()
        arrow.header.frame_id = "base_link"
        arrow.header.stamp = now
        arrow.ns = "policy_action"
        arrow.id = 0
        if twist is not None:
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            v = float(twist.linear.x)
            w = float(twist.angular.z)
            # Bend the arrow toward the commanded turn (small-angle) so w shows too.
            arrow.points = [
                Point(x=0.0, y=0.0, z=0.5),
                Point(x=v * math.cos(w * 0.3), y=v * math.sin(w * 0.3), z=0.5),
            ]
            arrow.scale.x = 0.06
            arrow.scale.y = 0.12
            arrow.scale.z = 0.0
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (0.2, 0.45, 1.0, 0.9)
        else:
            arrow.action = Marker.DELETE
        markers.markers.append(arrow)

        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp = now
        text.ns = "policy_action"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 1.1
        text.pose.orientation.w = 1.0
        text.scale.z = 0.13
        text.color.r, text.color.g, text.color.b, text.color.a = (0.5, 0.75, 1.0, 0.9)
        vel_str = (f"v={twist.linear.x:.2f} w={twist.angular.z:.2f}"
                   if twist is not None else "v/w: user-driven")
        text.text = (
            f"{vel_str}\n"
            f"flippers ({flipper_mode}) [FL,FR,RL,RR]="
            f"[{float(flipper_values[0]):.2f}, {float(flipper_values[1]):.2f}, "
            f"{float(flipper_values[2]):.2f}, {float(flipper_values[3]):.2f}]"
        )
        markers.markers.append(text)

        self.policy_action_markers_pub.publish(markers)

    def _flipper_direction(self, key: str, flipper_mode: str, value: float) -> str:
        """"up" / "down" / "still" -- the PHYSICAL direction `value` commands for
        corner `key`, in MARV's ROS sign convention (front negative = up, rear
        positive = up).

        flipper_mode="position": `value` is an absolute radian target, so the
        direction is its offset from the live measured angle. Otherwise `value`
        is already a rad/s command and is used directly.
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
        """/policy_flipper_command_hud: a small top-down schematic (front to the
        LEFT) with one spot per corner, each an up arrow, down arrow, or a dash
        for "not moving"."""
        if cv2 is None:
            return
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
            cv2.putText(img, labels[key], (cx - 12, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (170, 170, 170), 1, cv2.LINE_AA)

        self.flipper_cmd_hud_pub.publish(self._bgr_image_msg(img))

    def publish_heightmap_extent_marker(self):
        """/policy_heightmap_extent: the FTR window's real extent (+/-1.10 m
        fore-aft x +/-0.50 m lateral, base_link) as a rectangle on the ground."""
        from marv_rl_training.training.ftr_heightmap_window import HM_ROWS, HM_COLS, HM_RES
        half_x = (HM_ROWS // 2) * HM_RES
        half_y = (HM_COLS // 2) * HM_RES

        rect = Marker()
        rect.header.frame_id = "base_link"
        rect.header.stamp = self._viz_stamp()
        rect.ns = "policy_heightmap_extent"
        rect.id = 0
        rect.type = Marker.LINE_STRIP
        rect.action = Marker.ADD
        rect.pose.orientation.w = 1.0  # an all-zero quaternion is invalid; RViz drops the marker
        corners = [(half_x, half_y), (half_x, -half_y), (-half_x, -half_y),
                   (-half_x, half_y), (half_x, half_y)]
        rect.points = [Point(x=cx, y=cy, z=0.02) for cx, cy in corners]
        rect.scale.x = 0.02
        rect.color.r, rect.color.g, rect.color.b, rect.color.a = (1.0, 1.0, 1.0, 0.8)
        self.heightmap_extent_pub.publish(MarkerArray(markers=[rect]))

    def publish_policy_debug_views(self, twist, flipper_values, flipper_mode,
                                   goal_vec_local=None, roll_deg=None, pitch_deg=None):
        """Every per-tick RViz decoration in one call, so both control paths
        (goal-driven and d3qn/creps) publish the same set."""
        self.publish_policy_obs_markers(goal_vec_local, roll_deg, pitch_deg)
        self.publish_policy_action_markers(twist, flipper_values, flipper_mode)
        self.publish_flipper_command_hud(flipper_values, flipper_mode)
        self.publish_heightmap_extent_marker()

    @staticmethod
    def _bgr_image_msg(img: np.ndarray) -> RosImage:
        """Packed bgr8 Image, built by hand rather than through cv_bridge: the
        compiled half of cv_bridge and its runtime type table disagree whenever
        the interpreter's OpenCV differs from the one ROS was built against, which
        surfaces as "KeyError: 16" on every publish."""
        img = np.ascontiguousarray(img, dtype=np.uint8)
        msg = RosImage()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = 3 * msg.width
        msg.data = img.tobytes()
        msg.header.frame_id = "base_link"
        return msg

    def _on_deadman(self, msg: Bool):
        self._deadman_held = bool(msg.data)
        self._deadman_rx_time = self.get_clock().now()

    def _on_estop(self, msg: Bool):
        if bool(msg.data) and not self._estop_latched:
            self.get_logger().warn("/marv/estop latched -- policy output gated")
        self._estop_latched = bool(msg.data)

    def _actuation_allowed(self) -> bool:
        if self._estop_latched:
            return False
        if not self.require_deadman:
            return True
        if not self._deadman_held or self._deadman_rx_time is None:
            return False
        age = (self.get_clock().now() - self._deadman_rx_time).nanoseconds * 1e-9
        return age <= self.deadman_timeout_sec

    def _on_flipper_override(self, msg: Bool):
        was = self._flipper_override_active
        self._flipper_override_active = bool(msg.data)
        if was == self._flipper_override_active:
            return
        self.get_logger().info(f"/flipper_override -> {self._flipper_override_active}")
        if self._flipper_override_active:
            # Rising edge. From here the control loop publishes nothing at all, and
            # Gazebo's velocity actuators hold the LAST command indefinitely (there
            # is no dead-man's-switch), so whatever was commanded at this instant
            # would otherwise be held forever. Zero it once, now.
            self.stop_all()
        elif self.auto_goal_on_release and self.current_goal is None and self.current_odom is not None:
            # Falling edge: hand the policy a one-shot goal straight ahead so a
            # release immediately resumes progress without an external goal
            # source. Gated by auto_goal_on_release (off when
            # obstacle_goal_publisher or an operator owns /goal_pose).
            pose = self.current_odom.pose.pose
            yaw = Rotation.from_quat([pose.orientation.x, pose.orientation.y,
                                      pose.orientation.z, pose.orientation.w]).as_euler("xyz")[2]
            goal = PoseStamped()
            goal.header = self.current_odom.header
            goal.pose.position.x = pose.position.x + self.auto_goal_ahead_m * float(np.cos(yaw))
            goal.pose.position.y = pose.position.y + self.auto_goal_ahead_m * float(np.sin(yaw))
            goal.pose.position.z = pose.position.z
            goal.pose.orientation = pose.orientation
            self.get_logger().info(
                f"override released -> auto goal {self.auto_goal_ahead_m:.1f} m ahead "
                f"({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})")
            self.goal_callback(goal)

    def stop_all(self):
        """Zero /cmd_vel and the flipper VELOCITY topics.

        Position targets are left alone: a stale /flippers_cmd_pos/* target just
        holds the flipper where it is, and the plugin drops those while the
        override is up anyway."""
        try:
            if self.publish_cmd_vel:
                self.cmd_vel_pub.publish(Twist())
            for pub in self.flipper_pubs.values():
                pub.publish(Float64(data=0.0))
        except Exception as e:
            self.get_logger().error(f"stop_all() failed: {e}")

    def _on_command_scale(self, msg: Float64):
        scale = float(np.clip(msg.data, 0.0, 5.0))
        if scale != self.command_scale:
            self.get_logger().info(f"command scale: {self.command_scale:.2f}x -> {scale:.2f}x")
        self.command_scale = scale

    def _scaled_flipper_target(self, key: str, target: float) -> float:
        """A position command with the command scale applied.

        `target` is an ABSOLUTE angle, so multiplying it would move the target
        itself -- 0.5x would aim at half the angle, not move there half as fast.
        Scale the DELTA from the live measured angle instead: 1x lands exactly
        where the policy aimed this tick, 3x takes a step 3x bigger toward it,
        0x holds still. The result is clamped back into the module's own joint
        interval so a >1x step cannot command past a real limit.
        """
        current = self.flipper_positions.get(key, 0.0)
        scaled = current + self.command_scale * (float(target) - current)
        limits = self._flipper_limits_ros()
        if limits is not None:
            i = ["front_left", "front_right", "rear_left", "rear_right"].index(key)
            scaled = float(np.clip(scaled, limits[i, 0], limits[i, 1]))
        return scaled

    def _flipper_limits_ros(self) -> np.ndarray | None:
        """Per-corner (low, high) joint limits in the RAW ROS frame, or None.

        ctrac already carries them as _ftr_limits_ros. d3qn/creps keep theirs in
        the FTR logical frame as front_low/high + rear_low/high, and the front
        pair has to be negated AND reversed on the way back (the module itself
        returns `-front_target` for FL/FR): FTR [-front_up, +front_down] is ROS
        [-front_down, +front_up]. An inverted interval would clamp against
        nothing, so the swap is not optional.
        """
        cached = getattr(self, "_ftr_limits_ros", None)
        if cached is not None:
            return cached
        p = self.policy
        if not all(hasattr(p, a) for a in ("front_low", "front_high", "rear_low", "rear_high")):
            return None
        limits = np.array([
            [-p.front_high, -p.front_low],
            [-p.front_high, -p.front_low],
            [p.rear_low, p.rear_high],
            [p.rear_low, p.rear_high],
        ], dtype=np.float64)
        self._ftr_limits_ros = limits
        return limits

    def _policy_view(self, hmap: np.ndarray | None):
        """The heightmap AS THE ACTIVE POLICY CONSUMES IT.

        Returns ``(view, used)``: the 2-D array to draw, and a boolean mask of the
        cells the network is actually fed (``None`` = all of them). The caller
        greys out everything outside the mask, so the panel never shows terrain
        detail that no policy input depends on.

        Not ``self.current_heightmap``: that is the full 8 x 8 m elevation-map crop
        -- square, and 3.5x too wide fore-aft. Publishing it made the panel square
        for every family whose module (not this node) owns the crop, hiding the
        fact that the policy only sees a 2.25 x 1.05 m window. Each module records
        its own crop in ``last_policy_heightmap``, so that is preferred; the node
        crops for itself only when the module kept none.

        Per family, on top of that window:

        ``ppo_ftr``   all 945 cells, flattened straight into the 966-D observation
                      (FtrPolicyInferenceModule._build_obs) -- the whole window,
                      unmasked.
        ``d3qn``      15 row-band means over the FULL width
                      (D3QNPolicyInferenceModule._terrain_bands), so the honest
                      picture is 15 uniform stripes laid back over the same
                      rectangle, not 45 rows of detail the net never receives. The
                      reduction is repeated here rather than read back because the
                      module keeps only the window -- and a mean is offset
                      invariant, so these bands and the net's differ by the
                      constant ``robot_z - track_wheel_radius``, which the display
                      normalisation removes anyway.
        ``ctrac``     one 12 x 20 strip 0.4-1.0 m ahead (_crop_local_window); the
                      rest of the window is carried by the module but never
                      reaches the observation.
        ``creps``     exactly ONE cell -- ``hm[height_ahead_row, HM_COLS // 2]`` --
                      plus pitch, which is not spatial and cannot be drawn.
        ``native``    the module's own percep_shape/percep_extent patch, whatever
                      size that config gives it, used whole.
        """
        hm = getattr(self.policy, "last_policy_heightmap", None)
        if hm is None or getattr(hm, "ndim", 0) != 2:
            hm = hmap
            if hm is None or hm.size == 0:
                return None, None
            if self._is_ftr:
                from marv_rl_training.training.ftr_heightmap_window import ftr_heightmap_window
                hm = ftr_heightmap_window(np.asarray(hm, dtype=np.float32), self.heightmap_extent)
        hm = np.asarray(hm, dtype=np.float32)

        if self._kind == "d3qn":
            from marv_rl_training.training.d3qn_policy_inference_module import (
                D3QNPolicyInferenceModule as _D3QN)
            rows = _D3QN._HM_DIM
            group = hm.shape[0] // rows if rows else 0
            if group and hm.shape[0] == rows * group:
                bands = hm.reshape(rows, group, hm.shape[1]).mean(axis=(1, 2))
                hm = np.repeat(bands, group)[:, None].repeat(hm.shape[1], axis=1)
            return hm, None

        if self._kind == "ctrac":
            # Marked by running the module's OWN crop over a grid of cell ids, so
            # the mask cannot drift from _crop_local_window's index arithmetic
            # (which row direction is front is a bug this project has shipped
            # before). Ids are 1-based: edge padding repeats a real id, and 0 then
            # unambiguously means "never sampled".
            from marv_rl_training.training.ctrac_policy_inference_module import _crop_local_window
            ids = np.arange(1, hm.size + 1, dtype=np.float32).reshape(hm.shape)
            used = np.isin(ids, _crop_local_window(ids))
            return hm, used

        if self._kind == "creps":
            from marv_rl_training.training.ftr_heightmap_window import HM_COLS
            diag = getattr(self.policy, "last_diag", None)
            row = int(diag["row"]) if diag else int(getattr(self.policy, "height_ahead_row", hm.shape[0] // 2))
            col = int(diag["col"]) if diag else min(HM_COLS // 2, hm.shape[1] - 1)
            used = np.zeros(hm.shape, dtype=bool)
            if 0 <= row < hm.shape[0] and 0 <= col < hm.shape[1]:
                used[row, col] = True
            return hm, used

        return hm, None   # ppo_ftr / native: the whole window is the observation

    @staticmethod
    def _detect_fixed_forward_vel(config_path: str) -> float | None:
        """env_cfg_overrides.fixed_forward_vel, or None when the policy controls its own speed.

        Non-None means the track speed was a constant held fixed throughout training, so the
        policy's velocity output carries no learned information and must not be published.
        """
        try:
            from omegaconf import OmegaConf

            v = OmegaConf.load(config_path).get("env_cfg_overrides", {}).get("fixed_forward_vel", None)
            return None if v is None else float(v)
        except Exception:
            return None

    @staticmethod
    def _detect_module_name(config_path: str) -> str | None:
        """Return env_cfg_overrides.module_name, or None (native/non-FTR config)."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            return cfg.get("env_cfg_overrides", {}).get("module_name")
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = FlipperPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only shutdown if context is still active (avoid double shutdown)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
