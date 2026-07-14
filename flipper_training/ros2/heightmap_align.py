"""Align the world-axis elevation map to the robot's local frame for policy input.

Deployment bug found 2026-07-14 (obs-viz audit): every observation factory's
``from_realistic_world`` (heightmap.py, elevation_boxes.py, pan_terrain.py) and
the training-path math define the heightmap input as ROBOT-LOCAL — rows along
the robot's yaw heading (row 0 = furthest ahead), columns along robot-left, and
heights RELATIVE to the robot base. ``flipper_policy_node`` however fed the
``elevation_mapping`` GridMap straight through: WORLD-axis-aligned rows and
ABSOLUTE world elevations. Consequences before this fix: at yaw != 0 the boxes/
bins/patches sampled rotated-wrong terrain; on any terrain the height statistics
carried the robot's absolute z as a constant offset.

Array convention (matches the node's post-transpose layout and the factories'
grid_sample mapping, extent = (x_max, y_max, x_min, y_min)): row i corresponds
to x = x_max - (i + 1/2)/rows * (x_max - x_min) ... i.e. row 0 = +x edge, col 0
= +y edge (ROS: x forward, y left). The same convention holds for the input
(world axes) and output (robot axes), so the transform is a pure in-place
rotation about the array center by the robot yaw, plus a scalar shift.

Sign convention locked by tests/test_heightmap_align.py against analytic
patterns (see there), not by derivation alone.
"""
from __future__ import annotations

import numpy as np

__all__ = ["align_heightmap_to_robot"]


def align_heightmap_to_robot(
    hm_world: np.ndarray,
    extent: list[float] | tuple[float, float, float, float],
    yaw: float,
    robot_z: float,
) -> np.ndarray:
    """World-aligned, absolute-height map -> robot-local yaw-aligned, robot-relative map.

    Args:
        hm_world: (rows, cols) heights, world-axis aligned, row 0 = world +x edge,
            col 0 = world +y edge, ABSOLUTE elevation. Assumed robot-centered.
        extent: (x_max, y_max, x_min, y_min) meters (the node's convention).
        yaw: robot yaw in the world frame (rad).
        robot_z: robot base height in the world frame (m).

    Returns:
        (rows, cols) heights, row 0 = furthest AHEAD of the robot, col 0 = robot
        LEFT edge, heights relative to the robot base. Same shape as input.
        Samples falling outside the source map are clamped to the border
        (consistent with the factories' grid_sample padding_mode="border").
    """
    hm_world = np.asarray(hm_world, dtype=np.float32)
    rows, cols = hm_world.shape
    x_max, y_max, x_min, y_min = float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])

    # local-frame metric coordinates of every output cell center
    xl = np.linspace(x_max, x_min, rows, dtype=np.float32).reshape(rows, 1)
    yl = np.linspace(y_max, y_min, cols, dtype=np.float32).reshape(1, cols)
    c, s = np.cos(yaw, dtype=np.float32), np.sin(yaw, dtype=np.float32)
    # local -> world (map is robot-centered, so no translation)
    xw = c * xl - s * yl
    yw = s * xl + c * yl

    # world metric -> fractional source indices (row 0 = x_max, col 0 = y_max)
    fi = (x_max - xw) / (x_max - x_min) * (rows - 1)
    fj = (y_max - yw) / (y_max - y_min) * (cols - 1)
    fi = np.clip(fi, 0.0, rows - 1.000001)
    fj = np.clip(fj, 0.0, cols - 1.000001)

    i0 = np.floor(fi).astype(np.int32)
    j0 = np.floor(fj).astype(np.int32)
    di = fi - i0
    dj = fj - j0
    i1 = np.minimum(i0 + 1, rows - 1)
    j1 = np.minimum(j0 + 1, cols - 1)

    out = ((1 - di) * (1 - dj) * hm_world[i0, j0]
           + (1 - di) * dj * hm_world[i0, j1]
           + di * (1 - dj) * hm_world[i1, j0]
           + di * dj * hm_world[i1, j1])
    return (out - np.float32(robot_z)).astype(np.float32)
