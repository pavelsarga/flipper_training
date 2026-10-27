"""The velocity controller's input features from ROS-side data -- one implementation for both the
offline dataset (misc/velocity_il/extract_velocity_il.py, from bags) and the live Gazebo/robot
node, so training and deployment cannot drift apart. Pure numpy/scipy, no ROS.

Conventions match FtrEnv (velocity_control.py), which builds the same features in Isaac:

  heightmap  45 x 21 cells at 0.05 m, row 0 = front (+1.10 m), col 0 = robot's RIGHT (-0.50 m),
             rotated with the robot's yaw, heights relative to (base z - 0.1165). Isaac's own crop
             is ftr_env.calc_current_frame_height_maps + marv_rl_module.calc_scanned_height_maps.
  flippers   logical angle = ROS joint angle x [-1, -1, 1, 1]: positive = UP on all four,
             radians, order fl, fr, rl, rr.
  proprio    [v measured, v commanded at the previous step, flipper velocity x 4, flipper angle x 4]
             flipper velocity = measured angle change over the last control step / its duration
             (rad/s) -- from the measured angles, not the commands, so it means the same whichever
             controller (position or velocity mode, operator, Isaac policy) moves the flippers

The heightmap source is a grid_map layer in grid_map's index order once the start index is 0:
row i runs from +x to -x, column j from +y to -y, cell (0, 0) is the (+x, +y) corner.
"""
import numpy as np
from scipy.ndimage import map_coordinates

HM_ROWS, HM_COLS, HM_RES = 45, 21, 0.05
TRACK_WHEEL_RADIUS = 0.1165     # FtrEnvCfg.render_radius; = GT base_link z on flat ground in Gazebo
RATE = 20.0                     # control rate of the velocity controller (Hz)

CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
JOINTS = ("front_left_flipper_j", "front_right_flipper_j", "rear_left_flipper_j", "rear_right_flipper_j")
FTR_SIGN = np.array([-1.0, -1.0, 1.0, 1.0])
# Logical flipper limits of the Isaac configs (marv_flipper_{front,back}_{up,down}_deg 60/80/60/80 ->
# front [-60, +80], rear [-80, +60] deg): an Isaac target can never leave them, so neither may this one.
TARGET_LOW = np.deg2rad([-60.0, -60.0, -80.0, -80.0])
TARGET_HIGH = np.deg2rad([80.0, 80.0, 60.0, 60.0])
VEL_HOLD_S = 0.25
TARGET_LEAD_MAX = 0.35          # rad


def gridmap_layer(msg, layer="elevation_inpainted"):
    """(H, (cx, cy), res) for one grid_map_msgs/GridMap layer -- an rclpy message or one decoded
    from a bag by mcap_ros2 (same fields).

    H[i, j] follows grid_map's index convention once the start index is 0 (the filter chain
    calls convertToDefaultStartIndex): row i runs from +x to -x, column j from +y to -y, so
    cell (0, 0) is the (+x, +y) corner. The data is a column-major Eigen matrix, which is
    why the flat array is reshaped (cols, rows) and transposed."""
    if msg.outer_start_index or msg.inner_start_index:
        raise ValueError("GridMap with a non-zero start index (circular buffer not unrolled)")
    k = msg.layers.index(layer)
    arr = msg.data[k]
    dims = {d.label: d.size for d in arr.layout.dim}
    n_cols, n_rows = dims["column_index"], dims["row_index"]
    H = np.asarray(arr.data, dtype=np.float32).reshape(n_cols, n_rows).T
    p = msg.info.pose.position
    return H, (p.x, p.y), float(msg.info.resolution)


def window_offsets(rows=HM_ROWS, cols=HM_COLS, res=HM_RES, fwd_offset=0.0):
    """Robot-frame (x, y) of every window cell centre, FTR order: [rows, cols, 2]."""
    xr = (rows // 2 - np.arange(rows)) * res + fwd_offset      # row 0 = front
    yr = (np.arange(cols) - cols // 2) * res                    # col 0 = right (-y)
    return np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1)


def sample_world(H, center, res, xy):
    """Bilinear lookup of grid-map H at world points xy [..., 2]; NaN outside the map or in holes."""
    n_rows, n_cols = H.shape
    x0 = center[0] + (n_rows - 1) / 2 * res     # world x of row 0
    y0 = center[1] + (n_cols - 1) / 2 * res     # world y of column 0
    i = (x0 - xy[..., 0]) / res
    j = (y0 - xy[..., 1]) / res
    vals = map_coordinates(H, [i.ravel(), j.ravel()], order=1, mode="constant", cval=np.nan)
    return vals.reshape(xy.shape[:-1])


def robot_window(H, center, res, pose, fwd_offset=0.0, relative=True, margin=0):
    """FTR-convention 45 x 21 window of map H around pose (x, y, z, yaw), robot-centric.

    relative=True subtracts (z - TRACK_WHEEL_RADIUS), FTR's observation reference. NaN cells
    (unmapped) stay NaN; the caller decides how to fill them. margin=m widens the window by m
    cells on every side (same centre), for filters that need context past the window edge."""
    x, y, z, yaw = pose
    off = window_offsets(HM_ROWS + 2 * margin, HM_COLS + 2 * margin, fwd_offset=fwd_offset)
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.stack([x + c * off[..., 0] - s * off[..., 1], y + s * off[..., 0] + c * off[..., 1]], axis=-1)
    win = sample_world(H, center, res, xy)
    return win - (z - TRACK_WHEEL_RADIUS) if relative else win


class FlipperTargetTracker:
    """One absolute flipper target per flipper (logical convention) from whatever command mode the
    flipper controller uses -- Isaac's flipper-only policy commands absolute targets, Gazebo's
    controllers mostly velocities. Feed commands as they arrive, call step() once per control tick:

        /flippers_cmd_pos/*         target = value
        /flippers_cmd_pos_rel/*     target = measured + value
        /flippers_cmd_vel/*         velocity u held (until the next message or VEL_HOLD_S), target += u dt
        /flippers_cmd_vel_manual/*  the same, but the only channel that counts while /flipper_override
                                    is on (the relay gates the policy out exactly then)

    The target is kept within TARGET_LEAD_MAX of the measured angle (a saturated joint must not wind
    it up) and inside the Isaac logical limits. Command values are given in ROS joint convention.
    """

    KINDS = ("pos", "pos_rel", "vel", "vel_manual")

    def __init__(self, angle0, dt=1.0 / RATE):
        self.dt = dt
        self.target = np.asarray(angle0, dtype=float).copy()
        self.u = np.zeros(4)
        self.u_t = np.full(4, -np.inf)
        self.u_man = np.zeros(4)
        self.u_man_t = np.full(4, -np.inf)
        self.pending = []           # (t, kind, k, logical value), applied at the next step

    def command(self, t, kind, k, ros_value):
        self.pending.append((t, kind, k, FTR_SIGN[k] * ros_value))

    def step(self, t, angle, override):
        """Advance to tick time t with the measured logical angles [4]; returns (target [4], velocity [4])."""
        self.pending.sort(key=lambda e: e[0])
        for te, kind, k, val in self.pending:
            if kind == "vel_manual":
                self.u_man[k], self.u_man_t[k] = val, te
            elif override:
                continue
            elif kind == "pos":
                self.target[k], self.u[k] = val, 0.0
            elif kind == "pos_rel":
                self.target[k], self.u[k] = angle[k] + val, 0.0
            else:
                self.u[k], self.u_t[k] = val, te
        self.pending = []
        if override:
            vel = np.where(t - self.u_man_t <= VEL_HOLD_S, self.u_man, 0.0)
        else:
            vel = np.where(t - self.u_t <= VEL_HOLD_S, self.u, 0.0)
        tgt = np.clip(self.target + vel * self.dt, angle - TARGET_LEAD_MAX, angle + TARGET_LEAD_MAX)
        self.target = np.clip(tgt, TARGET_LOW, TARGET_HIGH)
        return self.target.copy(), vel


def flipper_velocity(angle, dt=1.0 / RATE):
    """[T, 4] measured flipper velocity of a [T, 4] angle series on the control grid; 0 at the
    first tick, as a fresh history has no previous angle."""
    angle = np.asarray(angle)
    return np.concatenate([np.zeros_like(angle[:1]), np.diff(angle, axis=0) / dt])


def proprio_frame(v_measured, v_cmd_prev, flip_vel, flip_angle):
    """One [10] proprio frame in the order velocity_control.FeatureHistory / VelocityNet expect."""
    return np.concatenate([[v_measured, v_cmd_prev], flip_vel, flip_angle]).astype(np.float32)
