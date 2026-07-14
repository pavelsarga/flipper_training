"""Sign-convention lock for heightmap_align (analytic patterns, all four yaws)."""
import sys
import numpy as np
from flipper_training.ros2.heightmap_align import align_heightmap_to_robot

fails = []
N = 81
extent = [2.0, 2.0, -2.0, -2.0]
xw = np.linspace(2.0, -2.0, N).reshape(N, 1)   # row 0 = world +x
yw = np.linspace(2.0, -2.0, N).reshape(1, N)   # col 0 = world +y
ROBOT_Z = 0.1

def expect(name, got, want, tol=1e-4):
    if abs(got - want) > tol:
        fails.append(f"{name}: {got:.4f} != {want:.4f}")

# pattern A: height = 0.2 * x_world (absolute)
hmA = np.broadcast_to(0.2 * xw, (N, N)).astype(np.float32)
# yaw=0: local x == world x -> front cell (row 0, center col) = 0.2*2 - z
a0 = align_heightmap_to_robot(hmA, extent, 0.0, ROBOT_Z)
expect("yaw0 front", a0[0, N//2], 0.2*2.0 - ROBOT_Z)
expect("yaw0 rear", a0[-1, N//2], -0.2*2.0 - ROBOT_Z)
# yaw=+90deg (robot faces world +y): local forward = world +y -> pattern becomes 0.2*y_l? NO:
# world x at local (x_l, y_l) with yaw=90: x_w = -y_l. Height=0.2*x_w = -0.2*y_l.
# front cell (x_l=2, y_l=0) -> x_w=0 -> 0 - z; left edge (y_l=2, x_l=0) -> x_w=-2 -> -0.4 - z
a90 = align_heightmap_to_robot(hmA, extent, np.pi/2, ROBOT_Z)
expect("yaw90 front", a90[0, N//2], 0.0 - ROBOT_Z)
expect("yaw90 left", a90[N//2, 0], -0.4 - ROBOT_Z)
expect("yaw90 right", a90[N//2, -1], 0.4 - ROBOT_Z)
# yaw=180: forward = world -x -> front = -0.4 - z
a180 = align_heightmap_to_robot(hmA, extent, np.pi, ROBOT_Z)
expect("yaw180 front", a180[0, N//2], -0.4 - ROBOT_Z)
# pattern B: height = 0.3 * y_world; yaw=-90 (facing world -y): front (x_l=2) -> y_w = ... 
# yaw=-90: x_w = c*x - s*y with c=0, s=-1 -> x_w = y_l; y_w = -x_l. Height=0.3*y_w = -0.3*x_l.
hmB = np.broadcast_to(0.3 * yw, (N, N)).astype(np.float32)
b = align_heightmap_to_robot(hmB, extent, -np.pi/2, ROBOT_Z)
expect("yawm90 front", b[0, N//2], -0.3*2.0 - ROBOT_Z)
expect("yawm90 left", b[N//2, 0], 0.0 - ROBOT_Z)  # y_l=2 -> y_w=-x_l=0? x_l=0 -> y_w=0
# identity: yaw=0, z=0 must be exact passthrough
c0 = align_heightmap_to_robot(hmA, extent, 0.0, 0.0)
if not np.allclose(c0, hmA, atol=1e-5):
    fails.append("identity passthrough broken")

print("FAILURES:", fails if fails else "none — alignment sign conventions locked")
sys.exit(1 if fails else 0)
