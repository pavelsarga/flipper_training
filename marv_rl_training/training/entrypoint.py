"""The ``if __name__ == "__main__"`` body shared by the training entry points.

Loading the config (with the respawn and ``--play`` special cases), applying the CLI
overrides, and running the trainer behind the exit guard are identical in ``train_ftr.py``,
``train_d3qn.py``, ``train_icmd3qn.py``, ``train_sac.py`` and ``train_creps.py``.

Import this from BLOCK 2 — it pulls in RunLogger, so Isaac Sim must already be running.
"""

import sys
import traceback

from omegaconf import OmegaConf

from marv_rl_training.utils.logutils import RunLogger

__all__ = ["load_raw_config", "resolve_train_config", "run_trainer"]


def load_raw_config(config_path: str, cli_overrides: list[str]):
    """Load a config YAML and merge the OmegaConf dotlist overrides on top."""
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


def resolve_train_config(args, unknown_args: list[str]):
    """Work out which config this invocation should train (or play) with.

    Three cases, in order:

    ``--play RUN_DIR``  read the config saved next to that run and point the weight paths at
                        its final checkpoint, with the logging backends off.

    a SLURM respawn     prefer the config the previous attempt saved over the one named on
                        the command line. The checkpoint being resumed was trained with those
                        reward weights and that LR schedule; re-reading configs/ would apply
                        whatever has been edited there since attempt_0 started.

    otherwise           ``args.config``.

    ``--num_envs`` / ``--terrain`` / ``--task`` are applied last, on top of whichever won.
    """
    play_dir = getattr(args, "play", None)
    if play_dir is not None:
        from pathlib import Path

        play_path = Path(play_dir)
        saved_cfg_path = play_path / "config.yaml"
        if not saved_cfg_path.exists():
            raise FileNotFoundError(f"No config.yaml found in {play_path}")
        raw_cfg = load_raw_config(str(saved_cfg_path), unknown_args)
        weights_dir = play_path / "weights"
        raw_cfg.policy_weights_path = str(weights_dir / "policy_final.pth")
        raw_cfg.vecnorm_weights_path = str(weights_dir / "vecnorm_final.pth")
        raw_cfg.use_wandb = False
        raw_cfg.use_tensorboard = False
    else:
        prev_cfg_path = RunLogger.latest_attempt_config()
        raw_cfg = None
        if prev_cfg_path is not None:
            print(f"[INFO] Respawn detected — loading config from previous attempt: {prev_cfg_path}",
                  flush=True)
            raw_cfg = load_raw_config(str(prev_cfg_path), unknown_args)
            if not raw_cfg:
                print(f"[WARNING] Previous attempt config at {prev_cfg_path} is empty — "
                      f"falling back to {args.config}", flush=True)
                raw_cfg = None
        if raw_cfg is None:
            raw_cfg = load_raw_config(args.config, unknown_args)

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.terrain is not None:
        raw_cfg.terrain = args.terrain
    if args.task is not None:
        raw_cfg.task = args.task
    return raw_cfg


def run_trainer(trainer_cls, raw_cfg, ftr_gym_env, **trainer_kwargs):
    """Build and run a trainer, exiting hard on anything that escapes ``train()``.

    Isaac Sim's atexit handlers deadlock on normal interpreter shutdown, so an exception
    propagating out of ``train()`` leaves the job holding its node until walltime instead of
    failing it — a crashed run was once observed sitting on a GPU for 15 minutes doing
    nothing. ``train()`` already force-exits on CUDA/W&B errors; this covers every other
    cause. Exit 1, not 75: 75 means "transient, respawn me" to the sbatch loop.
    """
    import os

    trainer = trainer_cls(raw_cfg, ftr_gym_env, **trainer_kwargs)
    try:
        return trainer.train()
    except BaseException as exc:  # noqa: BLE001 — must catch everything, see the docstring
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1)
