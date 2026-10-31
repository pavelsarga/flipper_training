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


def install_physx_sync(ftr_env) -> None:
    """``FTR_PHYSX_SYNC=1|views|set|get|step``: device-wide synchronize around every get_*/set_*
    (``1``/``views``), only set_* (``set``) or only get_* (``get``) of the robot's PhysX
    ArticulationView, or once after every ``sim.step()`` (``step``) — tests for a stream race.

    omni.physics.tensors converts each index tensor with ``tensor.to(torch.int32)``, a temporary
    that returns to torch's caching allocator as soon as the setter returns; if PhysX consumes it
    asynchronously on its own stream, the block can be reused and overwritten first, handing
    PhysX garbage indices. ``torch.cuda.synchronize()`` waits for every stream, PhysX's included.
    """
    mode = os.environ.get("FTR_PHYSX_SYNC", "0")
    if mode in ("0", ""):
        return

    def wrap(fn, before=True, after=True):
        def synced(*args, **kwargs):
            if before:
                torch.cuda.synchronize()
            out = fn(*args, **kwargs)
            if after:
                torch.cuda.synchronize()
            return out
        return synced

    env = ftr_env.unwrapped
    if mode == "step":  # one sync after every physics step: its GPU work is done before any tensor-API call
        env.sim.step = wrap(env.sim.step, before=False)
        print("[crash_recorder] FTR_PHYSX_SYNC=step: torch.cuda.synchronize() after every sim.step()", flush=True)
        return
    prefixes = {"1": ("get_", "set_", "apply_"), "views": ("get_", "set_", "apply_"), "set": ("set_", "apply_"), "get": ("get_",)}[mode]
    view = env._robot.root_physx_view
    names = [n for n in dir(type(view)) if n.startswith(prefixes)]
    for n in names:
        setattr(view, n, wrap(getattr(view, n)))
    print(f"[crash_recorder] FTR_PHYSX_SYNC={mode}: {len(names)} ArticulationView methods wrapped in torch.cuda.synchronize()", flush=True)


def install_physx_keepalive() -> None:
    """``FTR_PHYSX_KEEPALIVE=<n>``: hold the last n tensors omni.physics.tensors converts with
    ``as_contiguous_*`` — the cheap counterpart of FTR_PHYSX_SYNC. If PhysX reads a converted
    index/data temporary after the setter returns, keeping it referenced stops torch's caching
    allocator from handing that block to the next allocation before PhysX is done with it.
    """
    n = int(os.environ.get("FTR_PHYSX_KEEPALIVE", "0") or 0)
    if n <= 0:
        return
    from omni.physics.tensors.impl.frontend_torch import FrontendTorch

    if getattr(FrontendTorch, "_keepalive", None) is not None:
        return
    FrontendTorch._keepalive = collections.deque(maxlen=n)
    for name in ("as_contiguous_float32", "as_contiguous_uint32", "as_contiguous_uint8"):
        orig = getattr(FrontendTorch, name)

        def kept(self, tensor, _orig=orig):
            out = _orig(self, tensor)
            FrontendTorch._keepalive.append(out)
            return out

        setattr(FrontendTorch, name, kept)
    print(f"[crash_recorder] FTR_PHYSX_KEEPALIVE: last {n} converted tensors kept alive", flush=True)


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
