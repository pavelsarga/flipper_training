"""The one place that turns a deployment elevation map into FTR's observation window.

Every FTR-trained policy (the marv_rl/FTR PPO actor and the D3QN/CREPS baselines) was
trained on a *fixed-size window*, not on "whatever map you have, resized". In training
(FTR-Benchmark ftr_env.py:748) the window comes from
``Map.get_obs(pos, yaw, height_map_length)`` with ``height_map_length = (2.25, 1.05)``
at ``cell_size = 0.05``, i.e. exactly 45 x 21 cells covering +/-1.10 m ahead/behind x
+/-0.50 m left/right of the robot, then ``.flip(0)`` so row 0 is the front.

Deployment maps are much bigger -- a typical elevation_mapping crop is 8 m x 8 m at
0.05 m = 160 x 160 -- so resizing one straight to 45 x 21 squeezes 8 m of terrain into
the 2.25 m the policy believes it is looking at: obstacles arrive ~3.6x too small and
too close along-track (~7.6x laterally), which resembles nothing seen in training.
Crop to the window first, and only then resample.

Ported verbatim from the deploy-only checkout's own copy
(flipper_training/experiments/ppo/ftr_heightmap_window.py in
marv_flipper_control_research) so every *_policy_inference_module.py here does the
crop itself and doesn't depend on its caller to have done it -- ftr_policy_inference_module.py
in this same directory still does a bare cv2.resize and has this same squashing bug;
that one hasn't been ported to use this yet.
"""
import numpy as np

HM_ROWS = 45
HM_COLS = 21
HM_RES = 0.05  # m/cell


def ftr_heightmap_window(heightmap: np.ndarray, extent=None) -> np.ndarray:
    """Crop + resample an egocentric heightmap onto FTR's own 45 x 21 grid.

    Input convention (grid_map / flipper_training realistic-world):
        row 0 = x_max (front), row -1 = x_min (rear)
        col 0 = y_max (left),  col -1 = y_min (right)
    FTR convention (Map.get_obs + the .flip(0) in calc_current_frame_height_maps):
        row 0 = front, col index increases toward the robot's LEFT
    -- so the lateral axis is mirrored between the two and has to be flipped.
    (Immaterial for the atd3qn/icmd3qn/hfc profile observations, which average over
    all 21 columns, but the marv_rl/FTR actor sees the full grid.)

    ``extent`` is (x_max, y_max, x_min, y_min) in metres, robot frame. When it is None
    the map is assumed to already be an FTR-extent window and is only resampled.
    """
    import cv2

    hm = np.asarray(heightmap, dtype=np.float32)
    if extent is not None and hm.ndim == 2 and hm.shape[0] > 1 and hm.shape[1] > 1:
        x_max, y_max, x_min, y_min = (float(v) for v in extent)
        rows, cols = hm.shape
        res_x = (x_max - x_min) / (rows - 1)
        res_y = (y_max - y_min) / (cols - 1)
        # Half-windows exactly as FTR computes them: (size/2)//cell_size cells.
        half_x = (HM_ROWS // 2) * HM_RES  # 22 * 0.05 = 1.10 m
        half_y = (HM_COLS // 2) * HM_RES  # 10 * 0.05 = 0.50 m
        # Anchor on the robot (row/col index of x=0 / y=0) and take a FIXED cell count
        # either side, so the crop is exactly the right number of cells whichever way
        # the sub-cell rounding falls. Deriving both edges independently can round to
        # 44 or 20 and leave the resize below to stretch it back, quietly shifting the
        # scale by half a cell.
        n_rows = int(round(2.0 * half_x / res_x)) + 1
        n_cols = int(round(2.0 * half_y / res_y)) + 1
        r_lo = int(round(x_max / res_x - half_x / res_x))
        c_lo = int(round(y_max / res_y - half_y / res_y))
        r_hi = r_lo + n_rows - 1
        c_hi = c_lo + n_cols - 1
        # Clamp to the available map and edge-pad whatever falls outside it, so a
        # window larger than the mapped area still yields the right scale.
        pad = (max(0, -r_lo), max(0, r_hi + 1 - rows), max(0, -c_lo), max(0, c_hi + 1 - cols))
        crop = hm[max(0, r_lo): min(rows, r_hi + 1), max(0, c_lo): min(cols, c_hi + 1)]
        if crop.size == 0:
            crop = hm
        elif any(pad):
            crop = np.pad(crop, ((pad[0], pad[1]), (pad[2], pad[3])), mode="edge")
        hm = crop[:, ::-1]  # our col 0 = left -> FTR col 0 = right
    if hm.shape != (HM_ROWS, HM_COLS):
        hm = cv2.resize(np.ascontiguousarray(hm), (HM_COLS, HM_ROWS), interpolation=cv2.INTER_LINEAR)
    return hm
