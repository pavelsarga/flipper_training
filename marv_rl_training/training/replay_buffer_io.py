"""Periodic, crash-safe persistence of a slice of the D3QN replay buffer.

Why
---
`_save_training_checkpoint` persists the optimizer, both schedulers and `grad_steps`, but
NOT the replay buffer, so every respawn restarts collection from an empty buffer. Measured
across the completed runs, the share of iterations trained on a below-capacity buffer was:

    AT-D3QN  (2M capacity)   26-44%
    ICM-D3QN (8M capacity)   76-95%   <- effectively never full

For AT the cost is not detectable in the success rate (restart windows come out +0.029 vs a
-0.006 control over 481 non-boundary windows of the same width), but ICM's nominal 8M buffer
has never actually existed, which quietly confounds any AT-vs-ICM comparison.

What is saved
-------------
A uniform random subset WITHOUT replacement of `fraction` x the buffer's *capacity* (not its
current occupancy), so the file has a fixed upper bound regardless of when the crash lands.
While the buffer is still filling and holds less than that, the whole thing is saved.

One file, overwritten in place each time — this is a warm-start convenience, not a
checkpoint history, so it deliberately does not accumulate per-step copies.

Crash safety
------------
The write goes to a temporary file in the same directory and is then `os.replace`d onto the
target, which is atomic on POSIX, so a job dying mid-write leaves either the previous
complete file or the new complete one — never a torn one. A leftover `.tmp` is harmless and
is overwritten by the next save.

Loading is best-effort by contract: ANY failure (missing, truncated, unpickleable, wrong
schema, buffer rejects it) is logged and swallowed, and the caller proceeds with an empty
buffer. Losing a warm start costs a few million frames of refill; failing to start costs the
whole job.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

FILENAME = "replay_buffer.pt"
_FORMAT = 1


def save_replay_subset(buffer, capacity: int, weights_path, fraction: float = 1.0 / 3.0, logger=None) -> int:
    """Atomically overwrite <weights_path>/replay_buffer.pt with a random subset. Returns
    the number of transitions written (0 if nothing was written)."""
    try:
        n_have = len(buffer)
        if n_have == 0 or fraction <= 0:
            return 0
        n_keep = min(n_have, max(1, int(capacity * fraction)))
        idx = torch.randperm(n_have)[:n_keep]
        data = buffer[idx]
        # `index` is the storage slot the sample came from; it is meaningless once the data
        # is re-extended into a fresh buffer, and keeping it would overwrite the new slots'
        # own indices.
        if "index" in data.keys():
            data = data.exclude("index")

        target = Path(weights_path) / FILENAME
        tmp = target.with_suffix(".pt.tmp")
        torch.save({"format": _FORMAT, "n": n_keep, "data": data}, tmp)
        os.replace(tmp, target)   # atomic: readers see old or new, never partial
        if logger is not None:
            logger.info(f"Saved replay subset: {n_keep}/{n_have} transitions -> {target}")
        return n_keep
    except Exception as e:
        # Never let buffer persistence take down a training run.
        if logger is not None:
            logger.warning(f"Failed to save replay subset ({type(e).__name__}: {e}); continuing.")
        try:
            tmp = Path(weights_path) / (FILENAME + ".tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return 0


def load_replay_subset(buffer, candidate_dirs, logger=None) -> int:
    """Warm-start `buffer` from the newest readable replay_buffer.pt in `candidate_dirs`
    (searched in order). Returns the number of transitions loaded; 0 means the buffer is
    untouched and training proceeds with a cold buffer."""
    for d in candidate_dirs:
        path = Path(d) / FILENAME
        if not path.exists():
            continue
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(blob, dict) or blob.get("format") != _FORMAT or "data" not in blob:
                raise ValueError(f"unexpected schema in {path}")
            data = blob["data"]
            n = int(data.batch_size[0])
            if n == 0:
                return 0
            buffer.extend(data)
            if logger is not None:
                logger.info(f"Warm-started replay buffer with {n} transitions from {path}")
            return n
        except Exception as e:
            # Truncated/corrupt/schema-drifted file, or a buffer that rejects the layout.
            # Warn and fall through: a cold buffer is correct, just slower.
            if logger is not None:
                logger.warning(
                    f"Could not load replay subset from {path} ({type(e).__name__}: {e}); "
                    "starting with an empty buffer."
                )
            return 0
    return 0
