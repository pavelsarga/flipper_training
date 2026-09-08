"""Shared command-line front end for the Isaac Sim entry points.

Every ``train_*.py`` / ``eval_*.py`` script opens with the same block: build an
``ArgumentParser``, let ``AppLauncher`` add its own flags, parse, drop the leftovers that
are not OmegaConf overrides, and start Isaac Sim. It has to happen before anything imports
``omni.*``, which is why each script carried its own copy. This module holds it once.

Nothing here imports ``omni`` at module level — ``AppLauncher`` is imported inside
:func:`launch_isaac_app`, so importing this module can never start the simulator or pull in
an omni module ahead of the launcher.

Typical use, at the very top of an entry point::

    from marv_rl_training.training.cli import launch_isaac_app, train_arg_parser

    if __name__ == "__main__":
        parser = train_arg_parser("Train PPO policy inside FTR-Benchmark (Isaac Sim)")
        args, unknown_args, simulation_app = launch_isaac_app(parser)

The ``if __name__`` guard matters for the trainers: the eval scripts import their config
dataclass from them, and an unguarded launcher would start a second Isaac Sim app on import.
"""

import argparse

__all__ = ["launch_isaac_app", "train_arg_parser", "eval_arg_parser", "add_eval_output_args"]


def _keep_only_dotlist_overrides(unknown_args: list[str]) -> list[str]:
    """Drop leftover ``--flag [value]`` pairs, keeping only ``key=value`` overrides.

    AppLauncher reads some of its flags (``--gpu``, for instance) straight out of
    ``sys.argv`` without removing them from ``parse_known_args``' leftovers, so they reach
    ``OmegaConf.from_dotlist`` and crash it. Anything that starts with ``--`` and has no
    ``=`` is therefore dropped along with the token after it, which is that flag's value.
    """
    kept, skip_next = [], False
    for arg in unknown_args:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--") and "=" not in arg:
            skip_next = True  # also drop the following positional value
            continue
        kept.append(arg)
    return kept


def launch_isaac_app(parser: argparse.ArgumentParser):
    """Add the AppLauncher flags, parse, start Isaac Sim.

    Returns ``(args, unknown_args, simulation_app)``, where ``unknown_args`` is the
    OmegaConf dotlist the caller merges on top of its config.
    """
    from omni.isaac.lab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, _keep_only_dotlist_overrides(unknown_args), AppLauncher(args).app


def train_arg_parser(description: str, *, play: bool = False) -> argparse.ArgumentParser:
    """Parser for a training entry point: ``--config`` plus the usual env overrides.

    ``play=True`` adds ``--play RUN_DIR``, which loads a finished run's weights and
    visualises the policy instead of training.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to the training config yaml")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
    parser.add_argument("--terrain", type=str, default=None, help="Override terrain in config")
    parser.add_argument("--task", type=str, default=None,
                        help="Override task in config (e.g. Ftr-Crossing-Direct-v0)")
    if play:
        parser.add_argument("--play", type=str, default=None, metavar="RUN_DIR",
                            help="Visualise a trained policy instead of training. Pass the run "
                                 "directory; loads policy_final.pth + vecnorm_final.pth from "
                                 "<RUN_DIR>/weights/.")
    return parser


def eval_arg_parser(
    description: str,
    *,
    policy_default: str = "policy_final.pth",
    policy_help: str | None = None,
    vecnorm: bool = True,
) -> argparse.ArgumentParser:
    """Parser for an eval entry point, with the flags every eval script accepts.

    ``vecnorm=False`` is for CREPS, which keeps its whole policy in one ``creps_state_*.pth``
    and writes no VecNorm checkpoint.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rundir", type=str, required=True, metavar="RUN_DIR",
                        help="Path to the run directory (must contain config.yaml and weights/).")
    parser.add_argument("--policy", type=str, default=policy_default,
                        help=policy_help or
                             f"Policy checkpoint filename inside <run>/weights/. (default: {policy_default})")
    if vecnorm:
        parser.add_argument("--vecnorm", type=str, default="vecnorm_final.pth",
                            help="VecNorm checkpoint filename inside <run>/weights/. "
                                 "(default: vecnorm_final.pth)")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots from config.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of independent eval rollouts to run and average. (default: 1)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override the eval rollout horizon (default: 2x the env's "
                             "max_episode_length, so each env gets ~2 episodes per repeat).")
    parser.add_argument("--map", type=str, default=None, metavar="TERRAIN",
                        help="Override the terrain from the saved config (e.g. ground, cur_mixed, "
                             "cur_stairs_up, exp_stair33_up).")
    add_eval_output_args(parser)
    return parser


def add_eval_output_args(parser: argparse.ArgumentParser) -> None:
    """The ``--output_dir`` family: where the per-env-type CSVs go and how rows are labelled."""
    parser.add_argument("--output_dir", type=str, default=None, metavar="DIR",
                        help="Directory to save CSV results (eval_summary.csv, eval_per_env.csv, "
                             "eval_episodes.csv, eval_per_spot.csv). Enables per-robot tracking via "
                             "a manual step loop. If omitted, prints metrics only (fast path).")
    parser.add_argument("--num_env_types", type=int, default=None,
                        help="Number of distinct env types cycling across robots. Default: looked up "
                             "from the terrain's registered layout (env_type_registry.py).")
    parser.add_argument("--env_names_yaml", type=str, default=None, metavar="YAML",
                        help="Path to YAML file mapping env-type index -> name (list or dict), "
                             "overriding the terrain's registered default names.")
    parser.add_argument("--eval_id", type=str, default=None,
                        help="Identifier for this eval run (default: auto UTC timestamp).")
