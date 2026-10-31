"""Flight recorder for the PhysX ``CUDA error: an illegal memory access`` crash.

Once that error is raised the CUDA context is dead and no GPU tensor can be read back, so the
simulator state that led to it is lost with the process. This keeps a CPU ring buffer of the
last ``FTR_CRASH_RECORDER`` control steps (root state, joint state, joint targets, actions,
resets) and writes it to ``FTR_CRASH_RECORDER_DIR`` when ``ftr_env.step`` raises. The copies
are taken *before* each step, so the last record is the state PhysX was handed on the step
that faulted.

Off unless ``FTR_CRASH_RECORDER`` is set: each record forces a GPU->CPU sync.
"""
from __future__ import annotations

import collections
import os
import sys
import time

import torch


def from_env(ftr_env) -> "CrashRecorder | None":
    n = int(os.environ.get("FTR_CRASH_RECORDER", "0") or 0)
    if n <= 0:
        return None
    out_dir = os.environ.get("FTR_CRASH_RECORDER_DIR", "/ws/logs/crash_recorder")
    print(f"[crash_recorder] recording the last {n} control steps -> {out_dir}", flush=True)
    return CrashRecorder(ftr_env, n, out_dir)


class CrashRecorder:
    def __init__(self, ftr_env, capacity: int, out_dir: str):
        self.env = ftr_env.unwrapped
        self.buf: collections.deque = collections.deque(maxlen=capacity)
        self.out_dir = out_dir
        self.step = 0
        self._prev_done: torch.Tensor | None = None

    def record(self, action: torch.Tensor) -> None:
        data = self.env._robot.data
        rec = {
            "step": self.step,
            "time": time.time(),
            "action": action.detach().cpu(),
            "root_state_w": data.root_state_w.detach().cpu(),
            "joint_pos": data.joint_pos.detach().cpu(),
            "joint_vel": data.joint_vel.detach().cpu(),
            "joint_pos_target": data.joint_pos_target.detach().cpu(),
            "joint_vel_target": data.joint_vel_target.detach().cpu(),
            "episode_length": self.env.episode_length_buf.detach().cpu(),
            # envs reset at the end of the previous step, i.e. spawned into this one
            "reset_before": self._prev_done,
        }
        for k in ("start_positions", "target_positions", "flipper_target_pos"):
            v = getattr(self.env, k, None)
            if isinstance(v, torch.Tensor):
                rec[k] = v.detach().cpu()
        self.buf.append(rec)
        self.step += 1

    def mark_done(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        self._prev_done = (terminated | truncated).detach().reshape(-1).cpu()

    def dump(self, exc: BaseException) -> None:
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}"
            path = os.path.join(self.out_dir, f"crash_{tag}.pt")
            torch.save({"error": repr(exc), "records": list(self.buf),
                        "num_envs": self.env.num_envs,
                        "terrain": getattr(self.env.cfg, "terrain_name", None)}, path)
            print(f"[crash_recorder] {len(self.buf)} steps before the crash saved to {path}", flush=True)
        except Exception as e:  # never mask the original error
            print(f"[crash_recorder] dump failed: {e!r}", file=sys.stderr, flush=True)
