"""Pure helpers for the BC chunk dataset — no Isaac imports, so they can be unit tested.

Kept out of collect_chunk_dataset.py deliberately: that module cannot be imported without a
running Isaac Sim (AppLauncher at import time), which would make its two most error-prone
pieces — the sliding-window boundary logic and the velocity-to-position label conversion —
testable only by burning a cluster job.
"""

from __future__ import annotations

import torch

__all__ = ["flipper_angles_to_position_action", "build_chunks"]


def flipper_angles_to_position_action(theta: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Achieved flipper angle -> the position-mode action that commands it.

    This is why switching the Phase 1/2 action space to position control costs nothing at
    the BC stage. The demonstrator was trained in VELOCITY mode, so its emitted flipper
    values are rates and are not valid position-mode labels. But what we record is the angle
    the robot actually reached, and position mode maps [-1, 1] linearly onto [low, high] --
    so inverting that map turns a velocity-mode demonstration into an exact position-mode
    label. `low`/`high` come from FtrEnv.flipper_angle_bounds(), which already accounts for
    MARV's asymmetric per-pair limits.
    """
    span = (high - low).clamp_min(1e-6)
    return (2.0 * (theta - low) / span - 1.0).clamp(-1.0, 1.0)


def build_chunks(obs_hist, actions, thetas, dones, T_p, low, high):
    """Slide a T_p window over one rollout, dropping any window crossing an episode reset.

    Args:
        obs_hist: [T, N, T_o*obs_dim]   observation window at each control step
        actions:  [T, N, A]             demonstrator action (velocity mode)
        thetas:   [T, N, n_flippers]    flipper angle reached AFTER that step
        dones:    [T, N]                episode ended on that step

    Returns (obs_history [M, T_o*obs_dim], action_chunk [M, T_p, A]).

    The label for step t uses the angle measured AFTER step t, because that is the angle the
    position-mode action at step t is asking for. Using the pre-step angle would teach the
    policy to command where it already is, i.e. to stand still.
    """
    T, N = actions.shape[0], actions.shape[1]
    n_flip = thetas.shape[-1]
    out_obs, out_act = [], []
    for t in range(T - T_p + 1):
        # Drop the window if any of its steps except the last ends an episode: a reset
        # inside the window means the later actions belong to a different episode.
        crossing = dones[t : t + T_p - 1].any(dim=0)          # [N]
        keep = ~crossing
        if not bool(keep.any()):
            continue
        chunk = actions[t : t + T_p].permute(1, 0, 2).clone()  # [N, T_p, A]
        pos = flipper_angles_to_position_action(thetas[t : t + T_p], low, high)  # [T_p, N, n_flip]
        chunk[..., -n_flip:] = pos.permute(1, 0, 2)
        out_obs.append(obs_hist[t][keep])
        out_act.append(chunk[keep])
    if not out_obs:
        return None, None
    return torch.cat(out_obs, 0), torch.cat(out_act, 0)


