import os

os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "4"

import math
from importlib import import_module
from pathlib import Path
from typing import Type

import torch
from lovely_tensors import monkey_patch
from omegaconf import OmegaConf

try:
    import torch._inductor.config
    torch._inductor.config.fallback_random = True
except (AttributeError, ImportError):
    pass
try:
    import torch._dynamo.config
    torch._dynamo.config.cache_size_limit = 128
except (AttributeError, ImportError):
    pass


PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT = PACKAGE_ROOT.parent
print(ROOT)


def resolve_class(typename: str) -> Type:
    module, class_name = typename.rsplit(".", 1)
    return getattr(import_module(module), class_name)


monkey_patch()


# MARV_RL's marv_rl_training package (imported transitively whenever a rl_baselines
# config's policy_config/env resolves a marv_rl_training.* or rl_modules.* class --
# see experiments/ppo/rl_baseline_inference_modules.py) registers these SAME resolver
# names with identical lambdas in its own __init__.py. Since flipper_training is
# always imported first in this deployment's PYTHONPATH ordering,
# OmegaConf.register_new_resolver is patched here, before flipper_training's own
# registrations, so marv_rl_training's later (redundant) registration attempt is a
# no-op instead of a ValueError.
_orig_register_new_resolver = OmegaConf.register_new_resolver


def _tolerant_register_new_resolver(name, resolver, **kwargs):
    try:
        return _orig_register_new_resolver(name, resolver, **kwargs)
    except ValueError:
        pass


OmegaConf.register_new_resolver = _tolerant_register_new_resolver


OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("mul", lambda *args: math.prod(args))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("intdiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("cls", resolve_class)
OmegaConf.register_new_resolver("dtype", lambda s: getattr(torch, s))  # get a torch dtype
OmegaConf.register_new_resolver("tensor", lambda s: torch.tensor(s))
