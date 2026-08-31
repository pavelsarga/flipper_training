#!/usr/bin/env python3
"""
ROS2 node for deploying the mitriakov rl_baselines policy on the MARV robot --
Mitriakov et al. 2021, "Reinforcement Learning Based, Staircase Negotiation Learning".

flipper_policy_node.py (this directory's generic FTR inference node) deliberately
refuses env_cfg_overrides.module_name == "mitriakov": "mitriakov needs step-edge
perception never implemented here". This node is that perception plus the trained
policy, as a separate file rather than another branch in that one, because mitriakov's
whole observation (Eq. 2's 8-D state vector -- local step-edge offsets + velocity +
flipper angles) shares nothing with the generic node's heightmap-window contract: no
heightmap, no goal vector, a different tiny MLP, and per-direction action bounds that
have no counterpart in any other module here. run_flipper_policy_sim.sh dispatches to
whichever of the two files matches a run's module_name, so `policy:=rl:<experiment>`
in marv_flipper_eval keeps working uniformly for every rl_baselines family including
this one.

Why the actor is reimplemented here in plain torch rather than importing
rl_modules.mitriakov.mitriakov_policy.MitriakovMLP/MitriakovFlipperBounds directly:
that module unconditionally imports tensordict/torchrl at the top level, and this
node's actual runtime is deliberately kept to torch + scipy + numpy (see the venv-
priority shim below, shared with flipper_policy_node.py, and marv_flipper_eval's own
hfcil_policy_node.py, which hit and solved the identical problem for rl_modules.hfc/
hfcil). MitriakovActorMLP below reproduces MitriakovMLP's exact
`self.mlp = nn.Sequential(...)` state_dict key layout, so a checkpoint written by
mitriakov_policy.py's MitriakovPolicyConfig.create() loads here with strict=True --
verified against a real logs/train_mitriakov_*/attempt_*/weights/policy_final.pth (the
ActorCriticWrapper's state_dict nests the actor under
"module.0....mlp.{0,2,4}.{weight,bias}"; _extract_actor_state locates that path
dynamically rather than hardcoding its depth, so a torchrl version bump elsewhere in
the training stack can't silently break the load here).

Step-edge geometry (p_x/p_y front/rear -- Fig. 2a's local step-edge offsets) comes from
StepEdgeSampler below: a TF-based sample of the incoming elevation map along the
robot's own local forward axis, finding the nearest riser ahead and behind. This is
deliberately NOT the "read row/col directly, no TF" shortcut flipper_policy_node.py
takes for its heightmap window -- that shortcut is fine for an MLP feature that
tolerates a residual frame error, but wrong here: p_x/p_y feed a hand-derived formula
whose entire point is precise geometry, and whether the incoming /elevation_map* is
already yaw-rotated into base_link (vs. published axis-aligned in a fixed frame with
the robot moving through it) isn't something this file can assume either way without
risk. The TF lookup (map frame -> base_frame) and yaw-rotated sampling degrade to the
same answer if the map already happens to be robot-aligned, so it is correct either
way -- marv_flipper_control_research/marv_flipper_eval's terrain_bands.py already
established this exact approach for the same elevation-map topics.

GT switching: no dedicated launch plumbing here either. marv_env.sh exports
MARV_GT_ELEV for the whole tmuxinator session (`tmuxinator start marv_flipper_eval
gt=1`), which the reactive controller's own launch already keys its odom_topic /
elevation-map choice off via the reactive_sim_gt.yaml / reactive_sim_icp_odom.yaml
overlay pair. This node reads that SAME env var as the default for its own `gt`
parameter (overridable with -p gt:=true/false, or elevation_topic/odom_topic
directly) instead of adding a second switch a user would have to remember to keep in
sync with the session's:
    gt=0 (default): /elevation_map_filtered (ICP-localized policy branch) + /icp_odom
    gt=1:            /elevation_map_gt_filtered (GT-pose branch)         + /ground_truth_odom

The action's track_v/track_w dims are NEVER published, matching the D3QN-family
rl_baselines convention documented in flipper_policy_node.py's own publish_cmd_vel
handling: "A policy trained behind a FIXED forward speed never learned to control
velocity... Publishing that constant on /cmd_vel would fight the operator/autodrive,
so on the robot we stay off the topic entirely and emit flipper commands only."
marv_config_mitriakov.yaml sets env_cfg_overrides.fixed_forward_vel: 0.6 for exactly
this reason (see that config's own comment: the trained actor's a[0] output was found
to be a step-to-step sign-flipping oscillation, not a driving signal), and the paper's
own task deliberately excludes yaw control ("a robot maintains a fixed orthogonal
orientation" to the staircase) -- so this node only ever writes /flippers_cmd_pos/*,
never /cmd_vel. Drive with teleop / auto_ride.
"""
import math
import os
import sys

# Force Python to load the venv packages BEFORE the ROS2 system packages -- see
# flipper_policy_node.py's identical block for why (torchrl version mismatch risk does
# not apply here since this node never imports torchrl, but keeping torch/scipy/numpy
# resolved from the same venv as every other node in this directory avoids a second,
# independent source of version drift).
if sys.prefix == sys.base_prefix:
    venv_site = os.path.expanduser("~/.venv/lib/python3.12/site-packages")
    if os.path.isdir(venv_site):
        if venv_site in sys.path:
            sys.path.remove(venv_site)
        sys.path.insert(0, venv_site)

import numpy as np
import rclpy
import torch
import torch.nn as nn
import yaml

try:
    import cv2
except ImportError:  # the HUD image degrades to a no-op; markers are unaffected
    cv2 = None
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image as RosImage, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
FLIPPER_NAMES = ["front_left_flipper_j", "front_right_flipper_j", "rear_left_flipper_j", "rear_right_flipper_j"]
DEFAULT_TRACK_WHEEL_RADIUS = 0.1165  # matches flipper_policy_node.py / ftr_env.py's MARV default

# MitriakovObservation's exact layout (mitriakov_observation.py):
# (p_x_front, p_y_front, p_x_rear, p_y_rear, v_s, psi_front, psi_rear, is_ascending)
OBS_DIM = 8
ACTION_DIM = 4  # (track_v, track_w, front, rear) -- only front/rear (indices 2, 3) are used here


def _gt_env_default() -> bool:
    return os.environ.get("MARV_GT_ELEV", "0").strip().lower() in ("1", "true", "yes", "on")


class MitriakovActorMLP(nn.Module):
    """Dense(8)->64->Tanh->64->Tanh->Dense(8), state_dict-key-identical to
    rl_modules.mitriakov.mitriakov_policy.MitriakovMLP (Fig. 1's architecture) -- see
    module docstring for why this is a plain-torch reproduction rather than an import."""

    def __init__(self, in_dim: int = OBS_DIM, out_dim: int = 2 * ACTION_DIM, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def _extract_actor_state(full_sd: dict) -> dict:
    """Locate the actor MitriakovMLP's mlp.{0,2,4}.{weight,bias} inside an
    ActorCriticWrapper checkpoint's state_dict, under whatever exact torchrl wrapping
    depth put them there (verified 4 levels deep -- "module.0.module.0.module.0.
    module.0.mlp.*" -- against a real checkpoint at the time this was written;
    located dynamically here instead of hardcoding that path so a torchrl version
    bump elsewhere can't silently break the load). policy_operator lives under the
    "module.0." prefix (ActorCriticWrapper's own state_dict indexes
    policy_operator=0, value_operator=1); value_operator's MitriakovMLP has
    out_features=1 and would also match "mlp.0.weight" un-prefixed, so restricting to
    "module.0." is what keeps this unambiguous.
    """
    prefix = "module.0."
    candidates = [k for k in full_sd if k.startswith(prefix) and k.endswith("mlp.0.weight")]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one actor MLP first-layer weight under '{prefix}*mlp.0.weight' "
            f"in the checkpoint, found {len(candidates)}: {candidates}")
    base = candidates[0][: -len("mlp.0.weight")]
    out = {}
    for i in (0, 2, 4):
        for p in ("weight", "bias"):
            out[f"mlp.{i}.{p}"] = full_sd[f"{base}mlp.{i}.{p}"]
    if out["mlp.0.weight"].shape[1] != OBS_DIM or out["mlp.4.weight"].shape[0] != 2 * ACTION_DIM:
        raise RuntimeError(
            f"actor MLP shape mismatch: got in_dim={out['mlp.0.weight'].shape[1]}, "
            f"out_dim={out['mlp.4.weight'].shape[0]}, expected {OBS_DIM}/{2 * ACTION_DIM} -- "
            "wrong checkpoint / not a mitriakov policy?")
    return out


class FlipperBounds:
    """Pure-python port of mitriakov_policy.py's MitriakovFlipperBounds -- computes the
    same per-direction (low, high) TanhNormal bounds (in [-1, 1] units relative to the
    OUTER marv_flipper_*_deg envelope), read from the SAME two config sections
    training used (env_cfg_overrides.marv_flipper_*_deg, policy_opts.
    mitriakov_limit_*_deg / mitriakov_trail_down_*_deg) so a config change is picked
    up here automatically. See that class's docstring for the physical rationale
    (paper's tip-over-safety "never raise the trailing flipper above neutral" vs.
    MARV's trail_down relaxation)."""

    def __init__(self, cfg: dict):
        env_ov = cfg.get("env_cfg_overrides", {}) or {}
        pol = cfg.get("policy_opts", {}) or {}

        front_up = math.radians(float(pol.get("front_up_deg", env_ov.get("marv_flipper_front_up_deg", 90.0))))
        front_down = math.radians(float(pol.get("front_down_deg", env_ov.get("marv_flipper_front_down_deg", 90.0))))
        back_up = math.radians(float(pol.get("back_up_deg", env_ov.get("marv_flipper_back_up_deg", 90.0))))
        back_down = math.radians(float(pol.get("back_down_deg", env_ov.get("marv_flipper_back_down_deg", 90.0))))
        self.front_low, self.front_high = -front_up, front_down    # OUTER envelope, radians
        self.rear_low, self.rear_high = -back_down, back_up        # OUTER envelope, radians

        rear_limit = math.radians(float(pol.get("mitriakov_limit_rear_deg", 45.0)))
        front_limit = math.radians(float(pol.get("mitriakov_limit_front_deg", 45.0)))
        rear_trail_down = math.radians(float(pol.get("mitriakov_trail_down_rear_deg", 0.0)))
        front_trail_down = math.radians(float(pol.get("mitriakov_trail_down_front_deg", 0.0)))

        def inv_map(target, low, high):
            return 2.0 * (target - low) / (high - low) - 1.0

        self.rear_tight_low = inv_map(-rear_trail_down, self.rear_low, self.rear_high)
        self.rear_tight_high = inv_map(rear_limit, self.rear_low, self.rear_high)
        self.front_tight_low = inv_map(-front_limit, self.front_low, self.front_high)
        self.front_tight_high = inv_map(front_trail_down, self.front_low, self.front_high)

    def bounds(self, is_ascending: bool):
        """-> ((front_lo, front_hi), (rear_lo, rear_hi)), in outer-envelope-normalised
        [-1, 1] units -- the tight sub-range applies to rear during ascent, front
        during descent (the other stays the full untouched outer range), matching the
        paper's own per-direction asymmetry."""
        if is_ascending:
            return (-1.0, 1.0), (self.rear_tight_low, self.rear_tight_high)
        return (self.front_tight_low, self.front_tight_high), (-1.0, 1.0)

    @staticmethod
    def to_angle(raw_loc: float, lo: float, hi: float, outer_low: float, outer_high: float) -> float:
        """Deterministic TanhNormal mode (tanh(loc) rescaled into [lo, hi]), then that
        outer-envelope-normalised command mapped to an absolute radian target the same
        way ftr_env.py's position control mode does (target = low + unit*(high-low))
        -- two rescales in a row, since [lo, hi] is itself expressed in outer-envelope
        units, not radians."""
        unit = (math.tanh(raw_loc) + 1.0) / 2.0
        tight_cmd = lo + unit * (hi - lo)
        unit2 = (tight_cmd + 1.0) / 2.0
        return outer_low + unit2 * (outer_high - outer_low)


class StepEdgeSampler:
    """TF + GridMap -> Mitriakov Eq. 2's local step-edge geometry. GridMap decode
    (index convention, bilinear interpolation) mirrors marv_flipper_control_research's
    terrain_bands.py, re-derived here rather than imported so this node has no
    dependency on that package (this file is the single source for the inference
    node, matching flipper_policy_node.py's own convention -- see this module's
    docstring)."""

    def __init__(self, tf_buffer: Buffer, base_frame: str, elevation_layer: str, track_wheel_radius: float):
        self._tf_buffer = tf_buffer
        self._base_frame = base_frame
        self.elevation_layer = elevation_layer
        self.track_wheel_radius = track_wheel_radius

    def robot_pose(self, map_frame: str, t: float):
        """-> (x, y, z, yaw) of base_frame in the map's frame, or None."""
        stamp = rclpy.time.Time(seconds=int(t), nanoseconds=int((t - int(t)) * 1e9))
        for query in (stamp, rclpy.time.Time()):
            try:
                tf = self._tf_buffer.lookup_transform(map_frame, self._base_frame, query)
            except Exception:
                continue
            tr = tf.transform.translation
            q = tf.transform.rotation
            yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            return float(tr.x), float(tr.y), float(tr.z), float(yaw)
        return None

    def _grid_array(self, msg: GridMap):
        """-> (H, W) float32, axis 0 = row (map +x at index 0), axis 1 = col (map +y).
        grid_map packs each layer column-major with dim[0]=columns (outer),
        dim[1]=rows (inner) -- see flipper_policy_node.py's elevation_map_callback for
        the same convention, verified there against grid_map_core."""
        if self.elevation_layer not in msg.layers:
            return None
        li = msg.layers.index(self.elevation_layer)
        layer = msg.data[li]
        cols = layer.layout.dim[0].size
        rows = layer.layout.dim[1].size
        arr = np.asarray(layer.data, dtype=np.float32)
        if arr.size != rows * cols:
            return None
        grid = arr.reshape((rows, cols), order="F")
        if msg.outer_start_index != 0 or msg.inner_start_index != 0:
            grid = np.roll(grid, -msg.inner_start_index, axis=0)
            grid = np.roll(grid, -msg.outer_start_index, axis=1)
        return grid

    def _sample(self, grid, msg: GridMap, px, py):
        """Bilinear-sample the map at world points (px, py) -> heights, NaN where
        unknown. grid_map indexing: index 0 is the MAXIMUM coordinate and grows as
        the coordinate shrinks, with a half-cell inset (grid_map_core's
        getIndexFromPosition)."""
        res = float(msg.info.resolution)
        lx, ly = float(msg.info.length_x), float(msg.info.length_y)
        cx, cy = float(msg.info.pose.position.x), float(msg.info.pose.position.y)
        oq = msg.info.pose.orientation
        map_yaw = Rotation.from_quat([oq.x, oq.y, oq.z, oq.w]).as_euler("xyz")[2]
        if abs(map_yaw) > 1e-6:
            c, s = math.cos(-map_yaw), math.sin(-map_yaw)
            dx, dy = px - cx, py - cy
            px, py = cx + c * dx - s * dy, cy + s * dx + c * dy

        rows, cols = grid.shape
        fi = (cx + lx / 2.0 - res / 2.0 - px) / res
        fj = (cy + ly / 2.0 - res / 2.0 - py) / res

        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        ti = fi - i0
        tj = fj - j0
        out = np.full(fi.shape, np.nan, dtype=np.float64)
        acc = np.zeros(fi.shape, dtype=np.float64)
        wsum = np.zeros(fi.shape, dtype=np.float64)
        for di, wi in ((0, 1.0 - ti), (1, ti)):
            for dj, wj in ((0, 1.0 - tj), (1, tj)):
                ii, jj = i0 + di, j0 + dj
                ok = (ii >= 0) & (ii < rows) & (jj >= 0) & (jj < cols)
                if not ok.any():
                    continue
                v = np.full(fi.shape, np.nan, dtype=np.float64)
                v[ok] = grid[ii[ok], jj[ok]]
                w = wi * wj
                good = ok & np.isfinite(v)
                acc[good] += w[good] * v[good]
                wsum[good] += w[good]
        has = wsum > 1e-9
        out[has] = acc[has] / wsum[has]
        return out

    def step_edges(self, msg, rx, ry, rz, yaw,
                    window_m: float = 3.0, res_m: float = 0.03,
                    dead_zone_m: float = 0.08, riser_threshold_m: float = 0.07):
        """-> (p_x_front, p_y_front, p_x_rear, p_y_rear, n_found) or None.

        None ONLY when the map has no usable layer -- a distinct, actionable failure
        (wrong elevation_layer / map not arriving) that must not look the same as
        "standing on flat ground", which is a perfectly normal state with a perfectly
        well-defined observation. n_found (0/1/2) says how many real risers backed
        the answer, for diagnostics.

        Samples height along the robot's local +x axis (y=0) from -window_m to
        +window_m at res_m spacing, relative to the track contact plane. dead_zone_m
        excludes samples directly under the robot (own-body occlusion) from edge
        detection. A "step edge" is a consecutive-sample |height jump| clearing
        riser_threshold_m, taken nearest the robot ahead of the dead zone (next_edge,
        Fig. 2a's front reference point) and nearest behind it (prev_edge, rear
        reference point).

        MISSING EDGES MIRROR TRAINING'S CLAMP RATHER THAN ABORTING. mitriakov_module.
        py's _progress_and_edges() indexes an analytic edge table with
        `next_idx = clamp(searchsorted(edges, progress_x), max=n-1)` and
        `prev_idx = clamp(next_idx - 1, min=0)`, so the observation is ALWAYS defined:
        before the staircase both indices collapse onto edges[0], making the front and
        rear pair an exact mirror (p_x_rear = -p_x_front, p_y_rear = -p_y_front) whose
        magnitude is simply the distance still to travel. The whole approach phase of
        every training episode looks like that, so it is squarely in-distribution -- and
        the policy has to be able to act during the approach, since that is when it
        chooses the pose it will hit the first step with. Reproduced here:
          * riser ahead, none behind (the normal approach) -> rear mirrors front;
          * riser behind, none ahead (rolled off the last step) -> front mirrors rear;
          * neither (flat ground, or staircase still beyond the map) -> saturate at
            +-window_m with zero height, i.e. "nothing anywhere near", still mirrored.
        Returning None here instead -- as the first version of this node did -- meant
        the node held its (nonexistent) last command and never wrote a flipper target
        at all until the robot was already straddling a step, which on an 8 m map with
        the obstacle 4 m out is never.
        """
        grid = self._grid_array(msg)
        if grid is None:
            return None

        xs = np.arange(-window_m, window_m + 1e-9, res_m)
        c, s = math.cos(yaw), math.sin(yaw)
        px = rx + xs * c
        py = ry + xs * s
        h = self._sample(grid, msg, px, py)
        h = h - (rz - self.track_wheel_radius)

        ahead_mask = xs > dead_zone_m
        behind_mask = xs < -dead_zone_m
        if not ahead_mask.any() or not behind_mask.any():
            return None

        edge_ahead = self._nearest_riser(xs[ahead_mask], h[ahead_mask], riser_threshold_m, from_start=True)
        edge_behind = self._nearest_riser(xs[behind_mask], h[behind_mask], riser_threshold_m, from_start=False)
        n_found = int(edge_ahead is not None) + int(edge_behind is not None)

        # (x_next, h_next) / (x_prev, h_prev) are the next/previous edge in signed
        # robot-local x (negative = behind), mirroring mitriakov_module.py's
        # next_edge/prev_edge exactly -- INCLUDING the clamp, where prev_idx == next_idx
        # makes them literally the same edge (not a reflected one), which is what turns
        # the p_front/p_rear pair into exact negatives of each other.
        if edge_ahead is not None and edge_behind is not None:
            x_next, h_next = edge_ahead
            x_prev, h_prev = edge_behind
        elif edge_ahead is not None:
            x_next, h_next = edge_ahead
            x_prev, h_prev = x_next, h_next
        elif edge_behind is not None:
            x_prev, h_prev = edge_behind
            x_next, h_next = x_prev, h_prev
        else:
            x_next, h_next = window_m, 0.0
            x_prev, h_prev = x_next, h_next

        # Training's exact four expressions, with h_* already expressed as
        # (edge height - robot height) by the contact-plane subtraction above:
        #   p_x_front = next_edge - progress   p_y_front = next_height - height_rel
        #   p_x_rear  = progress - prev_edge   p_y_rear  = height_rel - prev_height
        p_x_front, p_y_front = x_next, h_next
        p_x_rear, p_y_rear = -x_prev, -h_prev
        return p_x_front, p_y_front, p_x_rear, p_y_rear, n_found

    @staticmethod
    def _nearest_riser(xs, h, riser_threshold_m, from_start):
        """-> (edge_x, edge_height) for the riser CLOSEST to xs[0] (from_start=True,
        used ahead of the robot) or to xs[-1] (from_start=False, behind), or None if
        no consecutive-sample jump within the segment clears riser_threshold_m.
        edge_height is sampled a couple of cells past the jump (the tread the robot
        is stepping onto/off of) so it is not itself on the transition slope a
        coarse/interpolated map can leave at a sharp riser."""
        finite = np.isfinite(h)
        if finite.sum() < 3:
            return None
        order = np.arange(len(xs)) if from_start else np.arange(len(xs))[::-1]
        prev_i = None
        for i in order:
            if not finite[i]:
                continue
            if prev_i is None:
                prev_i = i
                continue
            if abs(h[i] - h[prev_i]) >= riser_threshold_m:
                far_i = min(i + 2, len(xs) - 1) if from_start else max(i - 2, 0)
                far_i = far_i if finite[far_i] else i
                return float(xs[i]), float(h[far_i])
            prev_i = i
        return None


class MitriakovPolicyNode(Node):
    def __init__(self):
        super().__init__("mitriakov_policy_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("policy_weights_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("control_rate", 20.0)  # Hz
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("track_wheel_radius", DEFAULT_TRACK_WHEEL_RADIUS)
        self.declare_parameter("gt", _gt_env_default())
        # Empty sentinel: resolved from `gt` below unless explicitly overridden.
        self.declare_parameter("elevation_topic", "")
        self.declare_parameter("odom_topic", "")
        # /marv/joint_states, NOT plain /joint_states. Both are throttles off the gz
        # bridge's /joint_states_raw (sim_foundation.launch.py), but /marv/joint_states is
        # the "robot-parity" feed that file deliberately added and that the reactive
        # controller, robot_state_publisher, flipper_eval_node, xu_hto and oehler_baseline
        # all subscribe to -- so it is the one kept alive and watched. Plain /joint_states
        # was observed with ZERO publishers on a fully-running session (its throttle
        # process had died while the marv one kept going), which starved this node
        # completely: it sat on "waiting for elevation map / joint states" forever and
        # never published a flipper command, a marker or a HUD frame.
        self.declare_parameter("joint_topic", "/marv/joint_states")
        # Deadman + estop gate (robot branch; same rationale as
        # flipper_policy_node). require_deadman:=false only for bag replay.
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("deadman_topic", "/marv/teleop/deadman")
        self.declare_parameter("estop_topic", "/marv/estop")
        self.declare_parameter("deadman_timeout_sec", 0.2)
        self.declare_parameter("elevation_layer", "elevation_inpainted")
        # 3.0 m: the sim's elevation map is 8x8 m robot-centred (elevation_sim.yaml's
        # length_in_x/y), so +-3 m stays well inside it with margin for the map lagging
        # the robot. It has to be this big -- the arena's obstacles sit ~4 m from spawn,
        # and a window that cannot see the staircase reports flat ground all the way in.
        self.declare_parameter("edge_window_m", 3.0)
        self.declare_parameter("edge_riser_threshold_m", 0.07)
        # Deadband on p_y_front for the ascending/descending latch (see _tick).
        self.declare_parameter("ascend_deadband_m", 0.04)

        gp = lambda n: self.get_parameter(n).get_parameter_value()  # noqa: E731
        config_path = gp("config_path").string_value
        weights_path = gp("policy_weights_path").string_value
        device_str = gp("device").string_value
        control_rate = gp("control_rate").double_value
        self.base_frame = gp("base_frame").string_value
        track_wheel_radius = gp("track_wheel_radius").double_value
        gt = gp("gt").bool_value
        elevation_topic = gp("elevation_topic").string_value
        odom_topic = gp("odom_topic").string_value
        joint_topic = gp("joint_topic").string_value
        self.elevation_layer = gp("elevation_layer").string_value
        self.edge_window_m = gp("edge_window_m").double_value
        self.edge_riser_threshold_m = gp("edge_riser_threshold_m").double_value
        self.ascend_deadband_m = gp("ascend_deadband_m").double_value

        if not config_path or not weights_path:
            self.get_logger().error("config_path and policy_weights_path parameters are required!")
            raise ValueError("Missing required parameters")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        module_name = ((cfg.get("env_cfg_overrides") or {}).get("module_name"))
        if module_name != "mitriakov":
            raise RuntimeError(
                f"config_path's env_cfg_overrides.module_name is {module_name!r}, expected "
                "'mitriakov' -- wrong config for this node (use flipper_policy_node.py instead).")
        self.bounds = FlipperBounds(cfg)
        hidden_dim = int((cfg.get("policy_opts") or {}).get("hidden_dim", 64))

        device = torch.device(device_str)
        full_sd = torch.load(weights_path, map_location=device)
        actor_sd = _extract_actor_state(full_sd)
        self.actor = MitriakovActorMLP(hidden_dim=hidden_dim).to(device)
        self.actor.load_state_dict(actor_sd, strict=True)
        self.actor.eval()
        self.device = device

        elevation_topic = elevation_topic or ("/elevation_map_gt_filtered" if gt else "/elevation_map_filtered")
        odom_topic = odom_topic or ("/ground_truth_odom" if gt else "/icp_odom")

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._terrain = StepEdgeSampler(self.tf_buffer, self.base_frame, self.elevation_layer, track_wheel_radius)

        self.latest_map = None
        self.latest_joints = None
        self.fwd_vel = 0.0
        self.is_ascending = True  # latched, see _tick
        self._last_n_found = None

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(GridMap, elevation_topic, self._on_map, 2)
        self.create_subscription(JointState, joint_topic, self._on_joints, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self._elevation_topic, self._joint_topic = elevation_topic, joint_topic

        self.pubs = {c: self.create_publisher(Float64, f"/flippers_cmd_pos/{c}", 4) for c in CORNERS}
        self._deadman_held = False
        self._deadman_rx_time = None
        self._estop_latched = False
        self.require_deadman = self.get_parameter("require_deadman").get_parameter_value().bool_value
        self.deadman_timeout_sec = self.get_parameter("deadman_timeout_sec").get_parameter_value().double_value
        self.create_subscription(
            Bool, self.get_parameter("deadman_topic").get_parameter_value().string_value,
            self._on_deadman, 10)
        self.create_subscription(
            Bool, self.get_parameter("estop_topic").get_parameter_value().string_value,
            self._on_estop, 10)
        self.pub_obs = self.create_publisher(Float64MultiArray, "~/obs", 5)
        # /policy_obs_markers is an existing display in marv_flipper_eval's rl_generic.rviz
        # (the config policy.launch.py falls back to for any rl run without a dedicated
        # one). Publishing the detected step edges there means that config shows this
        # node's actual perception instead of sitting empty -- its /policy_heightmap*
        # displays stay blank on purpose, since a mitriakov policy has no heightmap input
        # at all (mitriakov_observation.py). These two markers ARE the observation's
        # geometry, and the thing most worth watching: everything downstream is a
        # deterministic function of them.
        self.pub_markers = self.create_publisher(MarkerArray, "/policy_obs_markers", 4)
        # rl_generic.rviz's "Flipper Command HUD" Image display. flipper_policy_node.py
        # fills this for the families it serves; without it the panel reads "No Image"
        # for the whole run. Its sibling display, "Policy Heightmap Image", is left
        # unpublished ON PURPOSE and will keep saying "No Image": a mitriakov policy has
        # no heightmap input at all (mitriakov_observation.py's 8-D Eq. 2 state vector),
        # so there is nothing truthful to draw there -- better an obviously empty panel
        # than a synthesised picture implying an input the network never receives.
        self.pub_hud = self.create_publisher(RosImage, "/policy_flipper_command_hud", 4)

        self.create_timer(1.0 / control_rate, self._tick)

        self.get_logger().info(
            f"mitriakov_policy_node ready: {weights_path}\n"
            f"  gt={gt} -> elevation={elevation_topic}, odom={odom_topic}, joints={joint_topic}\n"
            "  -> /flippers_cmd_pos/* only, no /cmd_vel (fixed_forward_vel policy -- "
            "drive with teleop/auto_ride)\n"
            f"  @ {control_rate:g} Hz, edge_window_m={self.edge_window_m:g}, "
            f"riser_threshold_m={self.edge_riser_threshold_m:g}\n"
            "  ~/obs (Float64MultiArray, 8-D) mirrors MitriakovObservation exactly.")

    def _on_map(self, msg: GridMap):
        self.latest_map = msg

    def _on_joints(self, msg: JointState):
        self.latest_joints = msg

    def _on_odom(self, msg: Odometry):
        self.fwd_vel = float(msg.twist.twist.linear.x)

    def _flipper_angles(self, js: JointState):
        if len(js.name) != len(js.position):
            return None
        idx = {n: k for k, n in enumerate(js.name)}
        try:
            fl, fr, rl, rr = (js.position[idx[n]] for n in FLIPPER_NAMES)
        except KeyError:
            return None
        return (fl + fr) / 2.0, (rl + rr) / 2.0

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_deadman(self, msg: Bool):
        self._deadman_held = bool(msg.data)
        self._deadman_rx_time = self.get_clock().now()

    def _on_estop(self, msg: Bool):
        if bool(msg.data) and not self._estop_latched:
            self.get_logger().warning("/marv/estop latched -- policy output gated")
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

    def _tick(self):
        # Deadman/estop gate: this node emits POSITION targets, so gating means
        # publishing NOTHING (the last target holds the flippers where they are).
        if not self._actuation_allowed():
            self.get_logger().warning(
                "actuation gated: "
                + ("ESTOP latched" if self._estop_latched else "deadman not held/stale")
                + " -- holding last flipper positions", throttle_duration_sec=2)
            return
        # Name the topic that is actually silent. The original combined message ("waiting
        # for elevation map / joint states") could not distinguish a missing map from
        # missing joints, and the real cause -- a dead /joint_states throttle while the
        # map streamed fine -- was invisible in it.
        missing = []
        if self.latest_map is None:
            missing.append(self._elevation_topic)
        if self.latest_joints is None:
            missing.append(self._joint_topic)
        if missing:
            self.get_logger().warning(
                f"no messages yet on: {', '.join(missing)} -- check that topic has a publisher "
                "(ros2 topic info <topic>)", throttle_duration_sec=5)
            return

        pose = self._terrain.robot_pose(self.latest_map.header.frame_id, self._now())
        if pose is None:
            self.get_logger().warning("no TF for robot pose yet", throttle_duration_sec=5)
            return
        rx, ry, rz, yaw = pose

        edges = self._terrain.step_edges(
            self.latest_map, rx, ry, rz, yaw,
            window_m=self.edge_window_m, riser_threshold_m=self.edge_riser_threshold_m)
        if edges is None:
            self.get_logger().warning(
                f"elevation layer {self.elevation_layer!r} missing from the incoming map "
                f"(layers: {list(self.latest_map.layers)}) -- holding last flipper command",
                throttle_duration_sec=5)
            return
        p_x_front, p_y_front, p_x_rear, p_y_rear, n_found = edges

        # is_ascending: in training this is a FIXED per-episode property of the course
        # (mitriakov_module.py's _ascending_mask(): target_z > start_z, decided once at
        # spawn). Nothing hands us that on a real robot, so it is derived from the only
        # local evidence there is -- whether the next edge ahead is above or below the
        # robot -- which agrees with the training signal wherever the training signal is
        # defined: while ascending a staircase the next edge ahead is always higher,
        # while descending it is always the lip of a drop, hence lower. The deadband +
        # latch matter because p_y_front passes through ~0 on every flat approach and on
        # every tread, and this bit SWITCHES WHICH FLIPPER IS ANGLE-LIMITED
        # (MitriakovFlipperBounds) -- letting it chatter there would swap the tight bound
        # between front and rear several times per second mid-climb. Latched to whatever
        # was last unambiguous, starting from ascending.
        if p_y_front > self.ascend_deadband_m:
            self.is_ascending = True
        elif p_y_front < -self.ascend_deadband_m:
            self.is_ascending = False
        is_ascending = self.is_ascending

        flippers = self._flipper_angles(self.latest_joints)
        if flippers is None:
            self.get_logger().warning(
                f"none of {FLIPPER_NAMES} found in {self._joint_topic}", throttle_duration_sec=5)
            return
        psi_front, psi_rear = flippers

        obs = [p_x_front, p_y_front, p_x_rear, p_y_rear, self.fwd_vel, psi_front, psi_rear, float(is_ascending)]
        obs_t = torch.tensor([obs], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.actor(obs_t)[0]  # (2*ACTION_DIM,) = (loc[4], raw_scale[4])
        loc_front, loc_rear = float(out[2]), float(out[3])

        (front_lo, front_hi), (rear_lo, rear_hi) = self.bounds.bounds(is_ascending)
        front_angle = self.bounds.to_angle(loc_front, front_lo, front_hi, self.bounds.front_low, self.bounds.front_high)
        rear_angle = self.bounds.to_angle(loc_rear, rear_lo, rear_hi, self.bounds.rear_low, self.bounds.rear_high)

        # Log on CHANGE, not on a throttle: how many real risers back the observation is
        # the single thing that separates "tracking a staircase" from "extrapolating over
        # flat ground", and a periodic reprint of an unchanged value buries that.
        if n_found != self._last_n_found:
            what = {0: "no risers in window (flat/unmapped -- mirrored saturation)",
                    1: "one riser (other side mirrored, as in training's approach phase)",
                    2: "both risers (straddling a step)"}[n_found]
            self.get_logger().info(
                f"step-edge perception: {what}; p_front=({p_x_front:+.2f} m, {p_y_front:+.2f} m) "
                f"p_rear=({p_x_rear:+.2f} m, {p_y_rear:+.2f} m) ascending={is_ascending}")
            self._last_n_found = n_found

        for corner in ("front_left", "front_right"):
            self.pubs[corner].publish(Float64(data=front_angle))
        for corner in ("rear_left", "rear_right"):
            self.pubs[corner].publish(Float64(data=rear_angle))

        self.pub_obs.publish(Float64MultiArray(data=[float(x) for x in obs]))
        self._publish_markers(p_x_front, p_y_front, p_x_rear, p_y_rear, is_ascending, n_found,
                               rx, ry, rz, yaw, self.latest_map.header.frame_id)
        self._publish_hud(front_angle, rear_angle, psi_front, psi_rear, is_ascending)

    def _publish_hud(self, front_cmd, rear_cmd, psi_front, psi_rear, is_ascending):
        """rl_generic.rviz's Flipper Command HUD, drawn as a SIDE view (front to the
        LEFT) rather than the top-down four-corner schematic the velocity-mode families
        use, for two reasons specific to this policy:

        * `sync_flipper_control: true` means the front pair and the rear pair each move
          as one, so a four-corner layout necessarily prints two values twice -- pure
          duplication, and the main thing making the panel crowded.
        * A flipper ANGLE is a side-view quantity, and its sign convention is MIRRORED
          between the ends (front: negative = up; rear: positive = up -- see this repo's
          CLAUDE.md table). A bare signed number therefore cannot be read as up or down
          without remembering which end you are looking at, which is exactly the
          confusion this panel should be removing. Drawing each flipper as a line at its
          real angle makes up/down unambiguous without knowing the convention at all.

        Commanded angle is the bright solid line; the measured angle is a dim ghost line
        behind it. The GAP between them is the point of the panel: it is what
        "commanding hard into terrain that will not yield" looks like, which is the
        failure mode this baseline kept hitting (see marv_config_mitriakov.yaml's note on
        flipper torque pinned at the effort limit).
        """
        if cv2 is None:
            return
        W, H = 470, 276
        img = np.full((H, W, 3), 28, dtype=np.uint8)
        DIM, MID, BRIGHT = (110, 110, 110), (150, 150, 150), (215, 215, 215)

        # --- header: name + climb-direction badge -------------------------------
        cv2.putText(img, "MITRIAKOV", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, BRIGHT, 1, cv2.LINE_AA)
        badge, bcol = ("ASCEND", (90, 200, 90)) if is_ascending else ("DESCEND", (230, 170, 60))
        cv2.rectangle(img, (W - 132, 8), (W - 12, 34), bcol, 1, cv2.LINE_AA)
        cv2.putText(img, badge, (W - 122, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcol, 1, cv2.LINE_AA)

        # --- side view ------------------------------------------------------------
        cx0, cx1, cyt, cyb = 165, 305, 92, 116       # chassis box
        pivot_y = (cyt + cyb) // 2
        L = 58                                        # drawn flipper length, px
        # 0 deg datum: a faint horizontal through both pivots, so "level" is visible as
        # a reference rather than something you infer from the number.
        cv2.line(img, (56, pivot_y), (W - 56, pivot_y), (58, 58, 58), 1, cv2.LINE_AA)
        cv2.rectangle(img, (cx0, cyt), (cx1, cyb), (95, 95, 95), -1)
        # End labels sit OUTSIDE the tip arc (radius L about each pivot) rather than
        # above or below the chassis, where a fully raised or fully lowered flipper
        # would draw straight through them.
        cv2.putText(img, "front", (12, pivot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "rear", (W - 48, pivot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)

        def tip(end, ang):
            """Flipper tip in screen coords. Front extends LEFT (-x) and rear RIGHT
            (+x); screen y grows downward, so the two ends' opposite sign conventions
            (front up = negative, rear up = positive) both come out as "tip above the
            pivot" with these signs -- which is the whole point of drawing it."""
            if end == "front":
                return int(cx0 - L * math.cos(ang)), int(pivot_y + L * math.sin(ang))
            return int(cx1 + L * math.cos(ang)), int(pivot_y - L * math.sin(ang))

        for end, px, cmd, meas in (("front", cx0, front_cmd, psi_front),
                                    ("rear", cx1, rear_cmd, psi_rear)):
            gx, gy = tip(end, meas)
            cv2.line(img, (px, pivot_y), (gx, gy), (85, 85, 85), 5, cv2.LINE_AA)   # measured ghost
            tx, ty = tip(end, cmd)
            err = abs(math.degrees(cmd - meas))
            col = (90, 215, 90) if err < 12.0 else (60, 170, 240)
            cv2.line(img, (px, pivot_y), (tx, ty), col, 2, cv2.LINE_AA)            # commanded
            cv2.circle(img, (tx, ty), 4, col, -1, cv2.LINE_AA)
            cv2.circle(img, (px, pivot_y), 4, (140, 140, 140), -1, cv2.LINE_AA)

        # --- numbers --------------------------------------------------------------
        # UP/DOWN spelled out, resolving the mirrored convention in words so the reader
        # never has to. Deadband so a near-level flipper does not flicker UP/DOWN.
        def updown(end, ang):
            up = ang < 0 if end == "front" else ang > 0
            return "level" if abs(math.degrees(ang)) < 3.0 else ("UP" if up else "DOWN")

        cv2.putText(img, "cmd", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "now", (232, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)
        for row, (end, cmd, meas) in enumerate((("front", front_cmd, psi_front),
                                                 ("rear", rear_cmd, psi_rear))):
            y = 216 + row * 26
            err = abs(math.degrees(cmd - meas))
            col = (90, 215, 90) if err < 12.0 else (60, 170, 240)
            cv2.putText(img, f"{end.upper():<5s}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MID, 1, cv2.LINE_AA)
            cv2.putText(img, f"{math.degrees(cmd):+7.1f}", (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(img, f"{math.degrees(meas):+7.1f}", (196, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MID, 1, cv2.LINE_AA)
            cv2.putText(img, updown(end, cmd), (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if err >= 12.0:
                cv2.putText(img, f"gap {err:.0f}", (370, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                            (60, 170, 240), 1, cv2.LINE_AA)

        cv2.putText(img, "deg, 0 = level   bright = commanded, grey = measured", (14, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        msg = RosImage()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = 3 * msg.width
        msg.data = img.tobytes()
        msg.header.frame_id = self.base_frame
        self.pub_hud.publish(msg)

    def _publish_markers(self, p_x_front, p_y_front, p_x_rear, p_y_rear, is_ascending,
                          n_found, rx, ry, rz, yaw, map_frame):
        """The two step-edge reference points plus a ground-level polyline through the
        robot, published in the MAP frame.

        NOT in base_link, which is what the first version did and what made the markers
        sink underground whenever the chassis pitched. The observation's geometry is
        built in a yaw-only, gravity-aligned frame -- step_edges() walks the map's XY
        plane along the robot's heading and measures every height against the track
        contact plane, so p_y_* are world-vertical offsets, not chassis-relative ones.
        Placing them in base_link re-interprets those numbers in the PITCHED chassis
        frame, which tips the whole pair with the robot and buries it on any slope or
        stair. Converting to map coordinates here puts each point back where it was
        actually measured, so the markers stay pinned on the terrain regardless of
        attitude -- and, as a bonus, no TF transform is needed at all when RViz's fixed
        frame is already map.

        Heights are the terrain height at each edge: the contact plane sits at
        rz - track_wheel_radius, and h (= p_y_front, or -p_y_rear for the previous edge)
        is the edge's height above that plane, so their sum is ground level in map z.
        """
        # ZERO stamp, deliberately -- "use the latest transform you have", not "the
        # transform at this instant". map->base_link comes off the ICP/odom chain a few
        # tens of ms behind the clock, so stamping with now() asks TF to extrapolate into
        # the future; RViz refuses ("Lookup would require extrapolation into the future")
        # and DROPS the markers, which at 20 Hz reads as violent flicker rather than as
        # an error. Same convention (and reasoning) as flipper_policy_node.py's
        # _viz_stamp().
        now = TimeMsg()
        arr = MarkerArray()
        r = self._terrain.track_wheel_radius
        solid = n_found == 2
        c, s = math.cos(yaw), math.sin(yaw)
        ground_z = rz - r                      # track contact plane, map z

        def world(local_x, h):
            """Yaw-only local (along-heading, height-above-contact-plane) -> map xyz."""
            return rx + local_x * c, ry + local_x * s, ground_z + h

        next_pt = world(p_x_front, p_y_front)
        prev_pt = world(-p_x_rear, -p_y_rear)
        robot_pt = (rx, ry, ground_z)

        for i, (name, pt, rgb) in enumerate((("next_edge", next_pt, (0.1, 0.9, 0.2)),
                                              ("prev_edge", prev_pt, (1.0, 0.6, 0.1)))):
            mk = Marker()
            mk.header.frame_id = map_frame
            mk.header.stamp = now
            mk.ns = f"mitriakov_{name}"
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = (float(v) for v in pt)
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.12
            mk.color.r, mk.color.g, mk.color.b = rgb
            # Translucent whenever the point is an extrapolation rather than a measured
            # riser, so a glance separates "tracking a real step" from "mirroring across
            # flat ground".
            mk.color.a = 0.95 if solid else 0.35
            arr.markers.append(mk)

        # prev edge -> robot -> next edge, along the ground. Makes the two spheres read
        # as one measurement of the terrain profile the policy is actually acting on,
        # and shows at a glance which side of the robot each edge is on.
        link = Marker()
        link.header.frame_id = map_frame
        link.header.stamp = now
        link.ns = "mitriakov_link"
        link.id = 3
        link.type = Marker.LINE_STRIP
        link.action = Marker.ADD
        link.pose.orientation.w = 1.0
        link.scale.x = 0.015                   # LINE_STRIP: scale.x is the line width
        link.color.r, link.color.g, link.color.b = 0.6, 0.8, 1.0
        link.color.a = 0.9 if solid else 0.4
        for pt in (prev_pt, robot_pt, next_pt):
            p = Point()
            p.x, p.y, p.z = (float(v) for v in pt)
            link.points.append(p)
        arr.markers.append(link)

        txt = Marker()
        txt.header.frame_id = map_frame
        txt.header.stamp = now
        txt.ns = "mitriakov_state"
        txt.id = 2
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(rx), float(ry), float(rz + 0.5)
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.14
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"{'ASCEND' if is_ascending else 'DESCEND'}  risers={n_found}"
        arr.markers.append(txt)
        self.pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MitriakovPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
