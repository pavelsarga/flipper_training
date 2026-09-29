"""Convert an HFC-IL recording into per-step marv_rl observations + state-mode action targets.

The recordings (flipper_eval_marv hfcil_recorder, see paper_repos/HFC-IL-data-collection.md)
store a 24-D summary per 10 Hz tick -- 15 terrain bands, 4 raw flipper joint angles, roll,
roll rate, pitch, forward velocity, reset flag -- plus the operator's state label. The
diffusion policy observes the full marv_rl vector (945 heightmap + 15 state + prev action),
which is rebuilt here as closely as the summary allows:

  heightmap   band i (0 = front) spread over rows 3i..3i+2 and all 21 columns; both the bands
              and the policy's map are relative to z - wheel radius
  roll, pitch /pi
  lin vel     (fwd_vel, 0, 0) / hmap diagonal
  ang vel     (roll_rate, 0, 0) / pi
  flippers    raw ROS angle x [-1, -1, 1, 1] -> logical, then (a + limit) / (2 limit)
  goal        a constant goal_dist_m straight ahead / hmap diagonal
  prev action [fwd_vel, 0, +1 on the current state, -1 elsewhere]

Pairing follows train_hfcil (obs[t] with state[t] -> state[t+1]): the observation at t carries
state[t] as its previous action, and the action target at t is [fwd_vel[t+1], 0, one-hot
state[t+1]] with the one-hot scaled to +-target_scale so it sits inside the tanh range. The
last step of every episode has no target and is marked invalid.

The synthetic heightmaps are banded and constant across the robot's width, so a policy
pretrained on them sees a shifted distribution once it runs on the real maps.

Needs numpy only. Output: one .npz with obs (T, 960+A), action (T, A), valid (T,), state (T,),
episode (T,), plus the settings used.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

HM_ROWS, HM_COLS, N_BANDS = 45, 21, 15
HMAP_DIAG = math.hypot(HM_ROWS * 0.05, HM_COLS * 0.05)
RAW_TO_LOGICAL = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shards shard_*.npz with obs (T,24), state (T,), episode (T,); episode ids made unique."""
    obs_l, st_l, ep_l, offset = [], [], [], 0
    for f in sorted(glob.glob(str(path / "shard_*.npz"))):
        d = np.load(f)
        ep = d["episode"].astype(np.int64)
        obs_l.append(d["obs"].astype(np.float32))
        st_l.append(d["state"].astype(np.int64))
        ep_l.append(ep + offset)
        offset += int(ep.max()) + 1
    if not obs_l:
        raise FileNotFoundError(f"no shard_*.npz in {path}")
    return np.concatenate(obs_l), np.concatenate(st_l), np.concatenate(ep_l)


def build_observations(rec: np.ndarray, state: np.ndarray, num_states: int, flipper_limit_deg: float,
                       goal_dist_m: float) -> np.ndarray:
    T = rec.shape[0]
    bands = rec[:, 0:N_BANDS]
    heightmap = np.repeat(bands, HM_ROWS // N_BANDS, axis=1)[:, :, None].repeat(HM_COLS, axis=2).reshape(T, -1)
    limit = math.radians(flipper_limit_deg)
    flippers = (rec[:, 15:19] * RAW_TO_LOGICAL + limit) / (2 * limit)
    roll, roll_rate, pitch, fwd_vel = rec[:, 19], rec[:, 20], rec[:, 21], rec[:, 22]
    zeros = np.zeros(T, dtype=np.float32)
    orient = np.stack([roll, pitch], axis=1) / math.pi
    lin_vel = np.stack([fwd_vel, zeros, zeros], axis=1) / HMAP_DIAG
    ang_vel = np.stack([roll_rate, zeros, zeros], axis=1) / math.pi
    goal = np.tile(np.array([goal_dist_m, 0.0, 0.0], dtype=np.float32) / HMAP_DIAG, (T, 1))
    prev_scores = -np.ones((T, num_states), dtype=np.float32)
    prev_scores[np.arange(T), state] = 1.0
    prev_action = np.concatenate([fwd_vel[:, None], zeros[:, None], prev_scores], axis=1)
    return np.concatenate([heightmap, orient, lin_vel, ang_vel, flippers, goal, prev_action], axis=1).astype(np.float32)


def build_targets(rec: np.ndarray, state: np.ndarray, episode: np.ndarray, num_states: int,
                  target_scale: float) -> tuple[np.ndarray, np.ndarray]:
    T = rec.shape[0]
    valid = np.zeros(T, dtype=bool)
    valid[:-1] = episode[1:] == episode[:-1]
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    scores = -target_scale * np.ones((T, num_states), dtype=np.float32)
    scores[np.arange(T), state[nxt]] = target_scale
    v = np.clip(rec[nxt, 22], -target_scale, target_scale)
    action = np.concatenate([v[:, None], np.zeros((T, 1), dtype=np.float32), scores], axis=1).astype(np.float32)
    return action, valid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recording", required=True, help="directory with shard_*.npz")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--num_states", type=int, default=7, help="states 0..K-1 (flipper_states_deg order); others are dropped")
    p.add_argument("--flipper_limit_deg", type=float, default=90.0, help="the config's flipper_pos_max_deg")
    p.add_argument("--goal_dist_m", type=float, default=2.0)
    p.add_argument("--target_scale", type=float, default=0.9)
    args = p.parse_args()

    rec, state, episode = load_recording(Path(args.recording))
    keep = state < args.num_states
    if not keep.all():
        print(f"dropping {int((~keep).sum())} steps with state >= {args.num_states}")
        rec, state, episode = rec[keep], state[keep], episode[keep]
    obs = build_observations(rec, state, args.num_states, args.flipper_limit_deg, args.goal_dist_m)
    action, valid = build_targets(rec, state, episode, args.num_states, args.target_scale)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs, action=action, valid=valid, state=state, episode=episode,
                        settings=json.dumps(vars(args)))
    trans = valid & (np.r_[state[1:], state[-1]] != state)
    print(f"{out}: {len(obs)} steps, {len(np.unique(episode))} episodes, obs {obs.shape[1]}-D, "
          f"action {action.shape[1]}-D, {int(valid.sum())} with targets ({int(trans.sum())} transitions)")
    print("state counts:", np.bincount(state, minlength=args.num_states).tolist())


if __name__ == "__main__":
    main()
