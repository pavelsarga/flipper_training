"""Pins the two heightmap lateral conventions and the mirror between them.

Sim (FtrEnv.current_frame_height_maps): row 0 = front, col 0 = the robot's RIGHT (-y).
Deployment (grid_map):                  row 0 = front, col 0 = the robot's LEFT  (+y).

They are mirrored, so every deployment path must flip the lateral axis. Getting this wrong
is silent: the policy simply acts as if obstacles were on the other side. Run with any
python that has numpy + cv2:

    python src/flipper_training/marv_rl_training/training/test_heightmap_orientation.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)                                             # .../src/flipper_training
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "FTR-Benchmark"))

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)

# --- sim side: measure MapHelper.get_obs rather than trust a comment -----------------
try:
    import tempfile
    from ftr_envs.assets.terrain.terrain import MapHelper
    CELL = 0.05
    LOWER = np.array([-5.0, -5.0])
    m = np.zeros((200, 200), dtype=np.float32)
    comp = -(LOWER / CELL).astype(np.int32)
    def _idx(x, y):
        return int(np.floor(x / CELL + comp[0])), int(np.floor(y / CELL + comp[1]))
    r, c = _idx(0.0, +0.35); m[r - 3:r + 3, c - 2:c + 2] = 5.0   # +y  = LEFT
    r, c = _idx(0.0, -0.35); m[r - 3:r + 3, c - 2:c + 2] = 1.0   # -y  = RIGHT
    r, c = _idx(+0.8, 0.0);  m[r - 2:r + 2, c - 2:c + 2] = 9.0   # +x  = FRONT
    f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False); np.save(f, m); f.close()
    loc = np.asarray(MapHelper(LOWER, np.array([5.0, 5.0]), CELL, f.name)
                     .get_obs([0.0, 0.0, 0.0], 0.0, (2.25, 1.05)))
    os.unlink(f.name)
    sim = loc[::-1, :]                                            # ftr_env does .flip(0)
    def _at(a, v):
        p = np.argwhere(np.isclose(a, v, atol=0.4)); return (int(p[:, 0].mean()), int(p[:, 1].mean()))
    check("sim: row 0 is the FRONT", _at(sim, 9.0)[0] < sim.shape[0] // 2, f"+x at row {_at(sim, 9.0)[0]}")
    check("sim: col 0 is the robot's RIGHT (-y)", _at(sim, 1.0)[1] < _at(sim, 5.0)[1],
          f"-y at col {_at(sim, 1.0)[1]}, +y at col {_at(sim, 5.0)[1]}")
except Exception as e:
    check("sim convention probe ran", False, f"{type(e).__name__}: {e}")

# --- grid_map message decode --------------------------------------------------------
def emulate_gridmap(M):
    """grid_map_ros packs the Eigen matrix column-major; dim[0]=nCols, dim[1]=nRows."""
    nRows, nCols = M.shape
    return M.flatten(order="F"), nCols, nRows

for shape in [(4, 4), (4, 3), (3, 5)]:
    M = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    data, cols, rows = emulate_gridmap(M)
    check(f"grid_map decode is exact for a {shape[0]}x{shape[1]} map",
          np.array_equal(data.reshape((rows, cols), order="F"), M))

# --- the mirror itself --------------------------------------------------------------
from marv_rl_training.training.ftr_heightmap_window import ftr_heightmap_window

EXTENT = [1.125, 0.525, -1.125, -0.525]          # exactly the FTR window, so no rescaling
rows, cols = 45, 21
dep = np.zeros((rows, cols), dtype=np.float32)   # grid_map order: col 0 = +y = LEFT
dep[:, :4] = 7.0                                 # a wall on the robot's LEFT
out = ftr_heightmap_window(dep, EXTENT)
check("window flips a LEFT-side wall onto the sim's right-hand columns",
      out[:, -4:].mean() > out[:, :4].mean(),
      f"left cols {out[:, :4].mean():.1f} vs right cols {out[:, -4:].mean():.1f}")
check("window preserves the 45x21 shape", out.shape == (rows, cols), str(out.shape))

no_extent = ftr_heightmap_window(dep, None)
check("without extent NO flip happens (so extent must always be passed)",
      no_extent[:, :4].mean() > no_extent[:, -4:].mean())

# --- the PPO deployment path must apply that flip -----------------------------------
src = open(os.path.join(ROOT, "marv_rl_training", "training",
                        "ftr_policy_inference_module.py")).read()
check("ftr_policy_inference_module uses the window helper (not a bare cv2.resize)",
      "ftr_heightmap_window(" in src)
check("ftr_policy_inference_module passes the extent through",
      "_build_obs(heightmap, heightmap_extent" in src)

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
