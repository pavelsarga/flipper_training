"""RViz observability for the deployed learned policies (flipper_policy_node).

Renders, per control tick, exactly what the loaded policy was fed -- from the values
``PPOPolicyInferenceModule`` already computed (``last_obs_raw`` / ``last_obs``, stashed
when ``capture_debug_obs`` is on) plus each observation factory's read-only
``viz_geometry()`` introspection. This module is the ONLY place observation semantics
meet ROS message types: the observation factories themselves stay ROS-free (they are
used in training without ROS), and nothing here alters observation math.

Outputs (published by the node, gated behind its ``debug_viz`` parameter):

* ``build_obs_marker_array`` -> visualization_msgs/MarkerArray on ``/policy_obs_markers``:
  - ``ElevationBoxFeatures`` (Azayev): each author box as a wireframe box + translucent
    CUBE at its TRUE zero-roll-pitch base-link-frame extents (world frame, yaw-only robot
    pose carrier), colored by coverage (red=empty -> green=full), spheres at the
    median/min_bnd/max_bnd heights, and a floating text label with the 4 raw stats.
  - ``PanTerrainState`` (Pan AT-D3QN/ICM-D3QN): the 15 Eq.-1 bins as a line of bars along
    the heading at their sampled heights (bin footprint = bin_width x 2*y_window), a
    profile LINE_STRIP over the bar tops, and a text label with theta_f1/theta_f2/theta_R.
  - ``Heightmap`` (C-TRAC h_t^l and any heightmap-fed policy): the actual resampled
    ``percep_extent`` patch the encoder consumes, as height-colored POINTS, plus the patch
    outline at robot height.
  - ``LocalStateVector`` (all native policies): the goal-direction arrow in base_link (the
    frame the policy's goal vector lives in) + roll/pitch/|goal| text; the terrain-lookahead
    variant additionally gets a sphere at its height-ahead sample point.
* ``build_obs_debug_json`` -> std_msgs/String JSON on ``/policy_obs_debug``: per factory
  name the flat raw (physical-units) and post-VecNorm observation vectors (summarized by
  stats when large, e.g. heightmaps), the factory geometry, and the raw action.

Dispatch is by class NAME over the factory's MRO (no isinstance imports), so adding a new
observation class degrades gracefully to JSON-only instead of crashing the deploy node.
"""

from __future__ import annotations

import json
import math

import numpy as np
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

__all__ = ["build_obs_marker_array", "build_obs_debug_json"]

# flat vectors up to this many elements are dumped verbatim into the JSON; larger
# observations (heightmap patches) are summarized (their pixels are already visible on
# /policy_obs_markers, /policy_heightmap_debug and /policy_heightmap)
_JSON_FULL_VECTOR_MAX = 64


# ---------------------------------------------------------------------------- helpers

def _mro_names(obj) -> set[str]:
    return {c.__name__ for c in type(obj).__mro__}


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _height_color(t: float, alpha: float = 1.0) -> ColorRGBA:
    """Blue (low) -> green (mid) -> red (high) for t in [0, 1] (the same ramp the node's
    /policy_heightmap_debug cloud uses)."""
    t = min(max(float(t), 0.0), 1.0)
    return ColorRGBA(r=t, g=1.0 - abs(t - 0.5) * 2.0, b=1.0 - t, a=alpha)


def _new_marker(frame: str, stamp, ns: str, mid: int, mtype: int) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def _carrier_pose(m: Marker, robot_xyz, yaw: float) -> None:
    """Pose the marker at the robot's position with a YAW-ONLY orientation, so the
    marker's ``points`` can be given directly in the zero-roll-pitch robot-local frame
    (local x/y offsets, z = height relative to the robot base) that the terrain-sampling
    observations are defined in."""
    m.pose.position.x = float(robot_xyz[0])
    m.pose.position.y = float(robot_xyz[1])
    m.pose.position.z = float(robot_xyz[2])
    qx, qy, qz, qw = _yaw_quat(yaw)
    m.pose.orientation.x = qx
    m.pose.orientation.y = qy
    m.pose.orientation.z = qz
    m.pose.orientation.w = qw


def _box_wire_points(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) -> list[Point]:
    """12 edges of an axis-aligned box as LINE_LIST point pairs (local frame)."""
    c = [(x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo),
         (x_lo, y_lo, z_hi), (x_hi, y_lo, z_hi), (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # bottom
             (4, 5), (5, 6), (6, 7), (7, 4),  # top
             (0, 4), (1, 5), (2, 6), (3, 7)]  # verticals
    pts = []
    for a, b in edges:
        pts.append(Point(x=float(c[a][0]), y=float(c[a][1]), z=float(c[a][2])))
        pts.append(Point(x=float(c[b][0]), y=float(c[b][1]), z=float(c[b][2])))
    return pts


def _text_marker(frame, stamp, ns, mid, robot_xyz, yaw, local_xyz, text, height=0.08) -> Marker:
    """TEXT_VIEW_FACING at a robot-local (yaw-frame) offset."""
    m = _new_marker(frame, stamp, ns, mid, Marker.TEXT_VIEW_FACING)
    cy, sy = math.cos(yaw), math.sin(yaw)
    m.pose.position.x = float(robot_xyz[0] + cy * local_xyz[0] - sy * local_xyz[1])
    m.pose.position.y = float(robot_xyz[1] + sy * local_xyz[0] + cy * local_xyz[1])
    m.pose.position.z = float(robot_xyz[2] + local_xyz[2])
    m.scale.z = height
    m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
    m.text = text
    return m


# ------------------------------------------------------------------- per-factory markers

def _markers_elevation_boxes(factory, raw: np.ndarray | None, frame, stamp, robot_xyz, yaw) -> list[Marker]:
    """Azayev's author boxes: wireframe + translucent CUBE at true extents, colored by
    coverage, med/min/max level spheres, and a [median,min,max,coverage] text label."""
    geom = factory.viz_geometry()
    markers: list[Marker] = []
    vec = None if raw is None else np.asarray(raw, dtype=np.float32).reshape(-1)
    for i, (name, bg) in enumerate(geom["boxes"].items()):
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = bg["bounds"]
        stats = vec[4 * i:4 * i + 4] if vec is not None and vec.size >= 4 * (i + 1) else None
        cov = float(stats[3]) if stats is not None else 0.0
        col = ColorRGBA(r=1.0 - cov, g=cov, b=0.15, a=1.0) if stats is not None else ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0)

        wire = _new_marker(frame, stamp, f"azayev_box/{name}", 0, Marker.LINE_LIST)
        _carrier_pose(wire, robot_xyz, yaw)
        wire.scale.x = 0.01
        wire.color = ColorRGBA(r=col.r, g=col.g, b=col.b, a=0.9)
        wire.points = _box_wire_points(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        markers.append(wire)

        fill = _new_marker(frame, stamp, f"azayev_box/{name}", 1, Marker.CUBE)
        cx, cyy, cz = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        fill.pose.position.x = float(robot_xyz[0] + cy_ * cx - sy_ * cyy)
        fill.pose.position.y = float(robot_xyz[1] + sy_ * cx + cy_ * cyy)
        fill.pose.position.z = float(robot_xyz[2] + cz)
        qx, qy, qz, qw = _yaw_quat(yaw)
        fill.pose.orientation.x, fill.pose.orientation.y = qx, qy
        fill.pose.orientation.z, fill.pose.orientation.w = qz, qw
        fill.scale.x, fill.scale.y, fill.scale.z = x_hi - x_lo, y_hi - y_lo, z_hi - z_lo
        fill.color = ColorRGBA(r=col.r, g=col.g, b=col.b, a=0.15)
        markers.append(fill)

        if stats is not None:
            med, mn, mx, _ = (float(s) for s in stats)
            levels = _new_marker(frame, stamp, f"azayev_box/{name}", 2, Marker.SPHERE_LIST)
            _carrier_pose(levels, robot_xyz, yaw)
            levels.scale.x = levels.scale.y = levels.scale.z = 0.05
            for z_val, c in ((mn, ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.95)),
                             (med, ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)),
                             (mx, ColorRGBA(r=1.0, g=0.3, b=0.2, a=0.95))):
                levels.points.append(Point(x=float(cx), y=float(cyy), z=float(z_val)))
                levels.colors.append(c)
            markers.append(levels)
            label = (f"{name}\nmed {med:+.2f}  min {mn:+.2f}  max {mx:+.2f}\ncov {cov:.2f}")
        else:
            label = f"{name}\n(no obs yet)"
        markers.append(_text_marker(frame, stamp, f"azayev_box/{name}", 3, robot_xyz, yaw,
                                    (cx, cyy, z_hi + 0.18), label))
    return markers


def _markers_pan_terrain(factory, raw: np.ndarray | None, frame, stamp, robot_xyz, yaw) -> list[Marker]:
    """Pan's Eq.-1 profile: n_heights bars along the heading at the sampled bin heights,
    a line over the bar tops, and the E = (theta_f1, theta_f2, theta_R) text."""
    geom = factory.viz_geometry()
    n = geom["n_heights"]
    centers = geom["bin_centers"]
    bw, yw = geom["bin_width"], geom["y_window"]
    markers: list[Marker] = []
    if raw is None:
        return markers
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vec.size < n + 3:
        return markers
    h = vec[:n]
    tf1, tf2, tr = (float(v) for v in vec[n:n + 3])
    h_scale = max(float(np.abs(h).max()), 0.1)

    bars = _new_marker(frame, stamp, "pan_bins", 0, Marker.CUBE_LIST)
    _carrier_pose(bars, robot_xyz, yaw)
    bars.scale.x, bars.scale.y, bars.scale.z = bw * 0.9, 2 * yw, 0.02
    profile = _new_marker(frame, stamp, "pan_bins", 1, Marker.LINE_STRIP)
    _carrier_pose(profile, robot_xyz, yaw)
    profile.scale.x = 0.015
    profile.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
    for x_c, h_i in zip(centers, h):
        p = Point(x=float(x_c), y=0.0, z=float(h_i))
        bars.points.append(p)
        bars.colors.append(_height_color(0.5 + float(h_i) / (2 * h_scale), alpha=0.85))
        profile.points.append(Point(x=float(x_c), y=0.0, z=float(h_i)))
    markers += [bars, profile]

    deg = 180.0 / math.pi
    markers.append(_text_marker(
        frame, stamp, "pan_bins", 2, robot_xyz, yaw, (0.0, 0.0, 0.65),
        f"PanTerrainState  H[{n}] range [{h.min():+.2f}, {h.max():+.2f}] m\n"
        f"θf1 {tf1 * deg:+.1f}°  θf2 {tf2 * deg:+.1f}°  θR {tr * deg:+.1f}°"))
    return markers


def _markers_heightmap_patch(factory, raw: np.ndarray | None, frame, stamp, robot_xyz, yaw) -> list[Marker]:
    """The exact resampled percep_extent patch the heightmap encoder consumes, as
    height-colored points, plus the patch outline at robot height."""
    markers: list[Marker] = []
    name = type(factory).__name__
    ext = tuple(float(v) for v in factory.percep_extent)  # (x_start, y_start, x_end, y_end)
    rows, cols = (int(v) for v in factory.percep_shape)

    outline = _new_marker(frame, stamp, f"heightmap_patch/{name}", 0, Marker.LINE_STRIP)
    _carrier_pose(outline, robot_xyz, yaw)
    outline.scale.x = 0.012
    outline.color = ColorRGBA(r=0.9, g=0.9, b=0.2, a=0.9)
    for x, y in ((ext[0], ext[1]), (ext[2], ext[1]), (ext[2], ext[3]), (ext[0], ext[3]), (ext[0], ext[1])):
        outline.points.append(Point(x=x, y=y, z=0.0))
    markers.append(outline)

    if raw is not None:
        hm = np.asarray(raw, dtype=np.float32).reshape(rows, cols)
        z_min, z_max = float(hm.min()), float(hm.max())
        z_rng = (z_max - z_min) or 1.0
        pts = _new_marker(frame, stamp, f"heightmap_patch/{name}", 1, Marker.POINTS)
        _carrier_pose(pts, robot_xyz, yaw)
        pts.scale.x = pts.scale.y = 0.025
        xs = np.linspace(ext[0], ext[2], rows)
        ys = np.linspace(ext[1], ext[3], cols)
        for i in range(rows):
            for j in range(cols):
                z = float(hm[i, j])
                pts.points.append(Point(x=float(xs[i]), y=float(ys[j]), z=z))
                pts.colors.append(_height_color((z - z_min) / z_rng, alpha=0.95))
        markers.append(pts)
        markers.append(_text_marker(frame, stamp, f"heightmap_patch/{name}", 2, robot_xyz, yaw,
                                    ((ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2, z_max + 0.25),
                                    f"{name} {rows}x{cols}\n[{z_min:+.2f}, {z_max:+.2f}] m"))
    return markers


def _markers_local_state(factory, raw: np.ndarray | None, base_frame, world_frame, stamp,
                         robot_xyz, yaw) -> list[Marker]:
    """Goal-direction arrow (base_link, the frame the policy's goal vector lives in) +
    roll/pitch/|goal| text; lookahead variant gets its height-ahead sample sphere."""
    markers: list[Marker] = []
    if raw is None:
        return markers
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    names = _mro_names(factory)
    has_lookahead = "LocalStateVectorWithTerrainHeightAhead" in names
    goal_sl = slice(-4, -1) if has_lookahead else slice(-3, None)
    max_dist = float(getattr(factory, "max_dist", 1.0))
    goal_m = vec[goal_sl].astype(np.float64) * max_dist  # de-normalized -> meters, base_link frame
    roll_deg = float(vec[0]) * 180.0
    pitch_deg = float(vec[1]) * 180.0

    arrow = _new_marker(base_frame, stamp, "local_state", 0, Marker.ARROW)
    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.03, 0.08, 0.1
    arrow.color = ColorRGBA(r=0.1, g=0.9, b=1.0, a=0.9)
    dist = float(np.linalg.norm(goal_m))
    tip = goal_m / dist * min(dist, 1.5) if dist > 1e-6 else np.zeros(3)
    arrow.points = [Point(x=0.0, y=0.0, z=0.35),
                    Point(x=float(tip[0]), y=float(tip[1]), z=0.35 + float(tip[2]))]
    markers.append(arrow)

    label = f"LocalStateVector\nroll {roll_deg:+.1f}°  pitch {pitch_deg:+.1f}°  |goal| {dist:.2f} m"
    if has_lookahead and vec.size >= 1:
        h_ahead = float(vec[-1])
        label += f"\nh_ahead {h_ahead:+.2f} m"
        sph = _new_marker(world_frame, stamp, "local_state", 2, Marker.SPHERE)
        la = float(getattr(factory, "lookahead_dist", 0.2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        sph.pose.position.x = float(robot_xyz[0] + cy * la)
        sph.pose.position.y = float(robot_xyz[1] + sy * la)
        sph.pose.position.z = float(robot_xyz[2] + h_ahead)
        sph.scale.x = sph.scale.y = sph.scale.z = 0.07
        sph.color = ColorRGBA(r=1.0, g=0.5, b=0.1, a=0.95)
        markers.append(sph)
    txt = _new_marker(base_frame, stamp, "local_state", 1, Marker.TEXT_VIEW_FACING)
    txt.pose.position.z = 0.9
    txt.scale.z = 0.08
    txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
    txt.text = label
    markers.append(txt)
    return markers


# ------------------------------------------------------------------------- public API

def build_obs_marker_array(
    factories,
    last_obs_raw: dict[str, np.ndarray] | None,
    stamp,
    robot_xyz,
    robot_yaw: float,
    world_frame: str = "world",
    base_frame: str = "base_link",
) -> MarkerArray:
    """MarkerArray rendering each observation factory's just-computed input.

    Args:
        factories: live Observation instances (``PPOPolicyInferenceModule.observation_factories``).
        last_obs_raw: raw from_realistic_world outputs keyed by factory name
            (``PPOPolicyInferenceModule.last_obs_raw``); geometry-only markers when missing.
        stamp: builtin_interfaces/Time for all headers.
        robot_xyz / robot_yaw: robot world pose (yaw-only carrier for the zero-roll-pitch
            local frame the terrain observations are defined in).
    """
    arr = MarkerArray()
    wipe = Marker()
    wipe.header.frame_id = world_frame
    wipe.header.stamp = stamp
    wipe.action = Marker.DELETEALL
    arr.markers.append(wipe)

    raw_by_name = last_obs_raw or {}
    for o in factories:
        names = _mro_names(o)
        raw = raw_by_name.get(o.name)
        try:
            if "ElevationBoxFeatures" in names:
                arr.markers += _markers_elevation_boxes(o, raw, world_frame, stamp, robot_xyz, robot_yaw)
            elif "PanTerrainState" in names:
                arr.markers += _markers_pan_terrain(o, raw, world_frame, stamp, robot_xyz, robot_yaw)
            elif "PrivilegedHeightmap" in names:
                continue  # critic-only, zeros at deployment -- nothing real to show
            elif "Heightmap" in names:
                arr.markers += _markers_heightmap_patch(o, raw, world_frame, stamp, robot_xyz, robot_yaw)
            elif "LocalStateVector" in names:
                arr.markers += _markers_local_state(o, raw, base_frame, world_frame, stamp, robot_xyz, robot_yaw)
            # other factories (PreviousAction, LatentControlParameter, contacts, ...) have
            # no natural 3D geometry -- they appear in the /policy_obs_debug JSON only
        except Exception:  # noqa: S112 -- a single bad factory must not kill the whole array
            continue
    return arr


def _vector_json(v: np.ndarray | None):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1)
    out = {"shape": list(v.shape)}
    if flat.size <= _JSON_FULL_VECTOR_MAX:
        out["values"] = [round(float(x), 5) for x in flat]
    else:
        out["summary"] = {
            "min": round(float(flat.min()), 5),
            "max": round(float(flat.max()), 5),
            "mean": round(float(flat.mean()), 5),
        }
    return out


def build_obs_debug_json(
    factories,
    last_obs_raw: dict[str, np.ndarray] | None,
    last_obs: dict[str, np.ndarray] | None,
    action: np.ndarray | None,
    stamp_sec: float,
    policy_type: str,
) -> str:
    """JSON snapshot of the policy's inputs/output this tick, per factory name:
    ``raw`` = from_realistic_world output (physical units), ``normalized`` = the
    post-transform (VecNorm etc.) value the actor consumed, plus factory geometry."""
    payload = {
        "stamp": round(float(stamp_sec), 4),
        "policy_type": policy_type,
        "action": None if action is None else [round(float(a), 5) for a in np.asarray(action).reshape(-1)],
        "factories": {},
    }
    raw_by_name = last_obs_raw or {}
    post_by_name = last_obs or {}
    for o in factories:
        entry = {
            "class": type(o).__name__,
            "raw": _vector_json(raw_by_name.get(o.name)),
            "normalized": _vector_json(post_by_name.get(o.name)),
        }
        viz_geom = getattr(o, "viz_geometry", None)
        if callable(viz_geom):
            try:
                entry["geometry"] = viz_geom()
            except Exception:  # noqa: S110
                pass
        payload["factories"][o.name] = entry
    return json.dumps(payload)
