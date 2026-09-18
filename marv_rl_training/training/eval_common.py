"""Rollout helpers shared by the eval entry points.

``eval_ftr.py``, ``eval_d3qn.py``, ``eval_sac.py``, ``eval_creps.py`` and
``eval_ftr_rand.py`` each had their own copy of the same result printer and, for the two
PPO-family ones, of the three rollout loops (plain, heightmap-plotting, action-printing) and
the weight-remapping helper. They live here now; the entry points keep only the parts that
differ, which is how the policy is built.

Everything takes the env pair explicitly — ``env`` is the transformed TorchRL env the policy
steps, ``ftr_torchrl_env`` the adapter underneath it that owns the termination/reward stats,
and ``ftr_gym_env`` the Isaac env whose raw state the plots read.
"""

from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.training.eval_data import _compute_obs_stats
from marv_rl_training.utils.logutils import get_terminal_logger

__all__ = [
    "ActionOverrideWrapper",
    "exit_flushed",
    "install_hard_exit_excepthook",
    "print_results",
    "print_lin_vels",
    "remap_native_to_ftr_weights",
    "run_single_rollout",
    "run_single_rollout_print_actions",
    "run_single_rollout_with_heightmap",
    "save_heightmap",
]

_log = get_terminal_logger("eval")

ACTION_LABELS_6 = ["lin_vel", "ang_vel", "fl_flip", "fr_flip", "rl_flip", "rr_flip"]
ACTION_LABELS_8 = ["track_fl", "track_fr", "track_rl", "track_rr",
                   "flip_fl", "flip_fr", "flip_rl", "flip_rr"]


def exit_flushed(code: int = 0) -> None:
    """Flush stdout/stderr, then leave via ``os._exit``.

    ``os._exit`` is required — Isaac Sim's shutdown re-initialises GPU foundation and
    regularly deadlocks there, holding the SLURM slot for hours — but it also skips flushing.
    When stdout is a redirected file it is block-buffered, so without the explicit flush the
    entire results summary is discarded and the run looks like it produced nothing while
    still exiting 0.
    """
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def install_hard_exit_excepthook() -> None:
    """Make an uncaught exception exit 1 instead of 0.

    Isaac Sim's Kit app runs its own teardown on normal interpreter exit and the process ends
    with status 0 whatever Python raised — a crashed eval under SLURM looked like a success.
    The trainers avoid it by wrapping ``train()`` (entrypoint.run_trainer); the eval scripts
    have no single call to wrap, so this hook does the same for anything that escapes their
    module body: print the traceback, flush, ``os._exit(1)``.
    """
    import sys
    import traceback

    def _hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        exit_flushed(1)

    sys.excepthook = _hook


def _termination_and_reward_stats(ftr_torchrl_env) -> dict[str, float]:
    """Drain the adapter's per-episode termination counters and per-component reward means."""
    term_info = ftr_torchrl_env.pop_termination_info()
    stats = {
        ("eval/explosion_rate" if k == "explosions/rate" else "eval/" + k.split("/", 1)[-1]): v
        for k, v in term_info.items()
    }
    stats.update(ftr_torchrl_env.pop_reward_info())
    return stats


def print_results(results: dict[str, float], header: str) -> None:
    """Print a metrics dict sorted by key, which groups it by ``rew/``, ``eval/``, ... prefix."""
    print(f"\n{'=' * 60}")
    print(header)
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:<45} {v:.6f}")


def remap_native_to_ftr_weights(state_dict: dict) -> dict:
    """Remap native flipper_training encoder key names to MarvRLFlatObservation naming.

    Native models have two separate observation encoders keyed by class name
    (LocalStateVector, Heightmap). FTR uses a single MarvRLFlatObservation with a
    FtrFlipperStyleEncoder whose sub-modules match but sit under different names.
    Actor/critic MLP heads have identical paths and transfer without remapping.
    """
    return {
        k.replace("encoders.LocalStateVector.mlp.mlp.",
                  "encoders.MarvRLFlatObservation.state_encoder.mlp.")
         .replace("encoders.Heightmap.encoder.",
                  "encoders.MarvRLFlatObservation.cnn.encoder."): v
        for k, v in state_dict.items()
    }


class ActionOverrideWrapper(torch.nn.Module):
    """Wraps a TensorDict policy operator to override components of the action it produces.

    Used to evaluate a policy outside the regime it was trained in: pinning the linear
    velocity for a policy trained without forward-command control, or flipping the rear
    flipper sign for one trained under the opposite convention.
    """

    def __init__(self, actor, const_linear_vel: float | None = None,
                 invert_rear_flippers: bool = False):
        super().__init__()
        self.actor = actor
        self.const_linear_vel = const_linear_vel
        self.invert_rear_flippers = invert_rear_flippers

    def forward(self, td):
        td = self.actor(td)
        if self.const_linear_vel is not None:
            td["action"][..., 0] = self.const_linear_vel
        if self.invert_rear_flippers:
            td["action"][..., 4:] = -td["action"][..., 4:]
        return td


def save_heightmap(ftr_gym_env, step: int, out_dir: Path) -> None:
    """Save the robot-0 heightmap for one step, as both step_NNNN.png and heightmap.png."""
    import matplotlib.pyplot as plt

    unwrapped = ftr_gym_env.unwrapped
    hmap = unwrapped.current_frame_height_maps[0].cpu().numpy()  # (45, 21)
    pos = unwrapped.positions[0].cpu()
    lin_vel = unwrapped.robot_lin_velocities[0].cpu().norm().item()
    ang_vel = unwrapped.robot_ang_velocities[0].cpu().norm().item()
    dist = (unwrapped.target_positions[0, :2] - unwrapped.positions[0, :2]).cpu().norm().item()

    fig, ax = plt.subplots(figsize=(5, 9))
    # origin="upper": row 0 at top = front (+x); row N at bottom = rear (-x).
    im = ax.imshow(hmap, origin="upper", cmap="terrain", aspect="auto")
    plt.colorbar(im, ax=ax, label="height (m)")
    cy, cx = hmap.shape[0] // 2, hmap.shape[1] // 2
    ax.plot(cx, cy, "r^", markersize=10, label="robot")   # robot sits at the map centre
    ax.set_title(
        f"step={step:04d}  pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})\n"
        f"lin_vel={lin_vel:.2f}m/s  ang_vel={ang_vel:.2f}rad/s  dist_goal={dist:.2f}m"
    )
    ax.set_xlabel("← −y (left)   +y (right) →")
    ax.set_ylabel("rear (bottom) ↑ front (top)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "heightmap.png", dpi=80)
    fig.savefig(out_dir / f"step_{step:04d}.png", dpi=80)
    plt.close(fig)


def print_lin_vels(ftr_gym_env, label: str = "Linear velocities", logger=None) -> None:
    """Print a per-robot linear velocity table with vx, vy, vz, speed and aggregate stats."""
    logger = logger or _log
    vels = ftr_gym_env.unwrapped.robot_lin_velocities.cpu()   # [N, 3]
    speeds = vels.norm(dim=-1)                                # [N]

    logger.info(f"{label}:")
    logger.info(f"  {'Robot':>5}  {'vx (m/s)':>9}  {'vy (m/s)':>9}  {'vz (m/s)':>9}  {'speed':>7}")
    for i in range(vels.shape[0]):
        logger.info(f"  {i:>5}  {vels[i, 0]:>9.4f}  {vels[i, 1]:>9.4f}  "
                    f"{vels[i, 2]:>9.4f}  {speeds[i]:>7.4f}")
    logger.info(f"  Summary — mean={speeds.mean():.4f}  max={speeds.max():.4f}  "
                f"min={speeds.min():.4f}  std={speeds.std():.4f} m/s")


def run_single_rollout(env, ftr_torchrl_env, actor, max_steps: int, *,
                       collect_obs_stats: bool = True) -> dict[str, float]:
    """Run one deterministic rollout and return a flat dict of metrics.

    ``collect_obs_stats=False`` skips the per-observation-slice summary, which assumes the
    marv_rl observation layout (see ``_OBS_SLICES`` in eval_data.py).
    """
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        rollout = env.rollout(max_steps, actor, auto_reset=True, break_when_all_done=True)
    results: dict[str, float] = {
        "eval/mean_step_reward": rollout["next", "reward"].mean().item(),
        "eval/max_step_reward":  rollout["next", "reward"].max().item(),
        "eval/min_step_reward":  rollout["next", "reward"].min().item(),
        "eval/pct_terminated":   rollout["next", "terminated"].float().mean().item(),
        "eval/pct_truncated":    rollout["next", "truncated"].float().mean().item(),
        "eval/rollout_steps":    float(rollout.shape[1]),
    }
    if collect_obs_stats:
        results.update(_compute_obs_stats(rollout[OBS_KEY]))
    del rollout
    results.update(_termination_and_reward_stats(ftr_torchrl_env))
    return results


def run_single_rollout_with_heightmap(env, ftr_torchrl_env, ftr_gym_env, actor, max_steps: int,
                                      out_dir: Path, plot_interval: int, *,
                                      collect_obs_stats: bool = True,
                                      logger=None) -> dict[str, float]:
    """Manual step loop that saves a heightmap image every ``plot_interval`` steps.

    A manual loop rather than ``env.rollout`` because the plots read the Isaac env's raw
    state, which is only valid between steps. Stitches the frames into a GIF at the end when
    imageio is available.
    """
    logger = logger or _log
    logger.info(f"Saving heightmap plots to {out_dir} every {plot_interval} step(s)")
    out_dir.mkdir(parents=True, exist_ok=True)

    td = env.reset()
    total_reward = 0.0
    n_steps = 0
    obs_list: list[torch.Tensor] = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        for step in range(max_steps):
            if plot_interval > 0 and step % plot_interval == 0:
                save_heightmap(ftr_gym_env, step, out_dir)
            if collect_obs_stats:
                obs_list.append(td[OBS_KEY].detach())
            td = actor(td)
            td = env.step(td)
            total_reward += td["next", "reward"].mean().item()
            n_steps += 1
            if td["next", "done"].all():
                break
            td = td["next"]

    save_heightmap(ftr_gym_env, n_steps, out_dir)

    try:
        import imageio.v2 as imageio
        gif_path = out_dir / "heightmap.gif"
        imageio.mimsave(str(gif_path),
                        [imageio.imread(str(f)) for f in sorted(out_dir.glob("step_*.png"))],
                        fps=10)
        logger.info(f"Saved GIF: {gif_path}")
    except Exception as e:  # noqa: BLE001 — imageio is optional; the PNGs are the real output
        logger.info(f"Could not create GIF ({e}). Individual PNGs are in {out_dir}")

    results: dict[str, float] = {
        "eval/mean_step_reward": total_reward / max(n_steps, 1),
        "eval/rollout_steps": float(n_steps),
    }
    if obs_list:
        results.update(_compute_obs_stats(torch.stack(obs_list)))
    results.update(_termination_and_reward_stats(ftr_torchrl_env))
    return results


def run_single_rollout_print_actions(env, ftr_torchrl_env, actor, max_steps: int,
                                     logger=None) -> dict[str, float]:
    """Manual step loop that prints the policy's action vector each step, for env 0."""
    logger = logger or _log
    td = env.reset()
    labels: list[str] | None = None
    total_reward = 0.0
    n_steps = 0

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        for step in range(max_steps):
            td = actor(td)

            if labels is None:
                action_dim = td["action"].shape[-1]
                labels = {6: ACTION_LABELS_6, 8: ACTION_LABELS_8}.get(
                    action_dim, [f"a[{i}]" for i in range(action_dim)])
                logger.info("step   " + "  ".join(f"{l:>9}" for l in labels))
                logger.info("-" * (7 + 11 * action_dim))

            logger.info(f"{step:>5}  " + "  ".join(f"{v:>9.4f}" for v in td["action"][0].cpu().tolist()))

            td = env.step(td)
            total_reward += td["next", "reward"].mean().item()
            n_steps += 1
            if td["next", "done"].all():
                break
            td = td["next"]

    results: dict[str, float] = {
        "eval/mean_step_reward": total_reward / max(n_steps, 1),
        "eval/rollout_steps": float(n_steps),
    }
    results.update(_termination_and_reward_stats(ftr_torchrl_env))
    return results
