"""Generic config-attribute schedulers, shared across training scripts (PPO, D3QN, ...).

Each scheduler mutates a single float attribute on an arbitrary config/object
(e.g. env_cfg.step_penalty, policy.epsilon) on every `.step()` call, mirroring
torch.optim.lr_scheduler semantics but decoupled from any specific optimizer.
"""


class _BaseCfgScheduler:
    """Base class for config attribute schedulers."""

    def __init__(self, cfg, attr: str, init_value: float, start_factor: float = 1.0, end_factor: float = 0.0, total_iters: int = 1):
        self._cfg = cfg
        self._attr = attr
        self._init = init_value
        self._start_f = start_factor
        self._end_f = end_factor
        self._total = total_iters
        self._step = 0
        setattr(self._cfg, self._attr, self._init * self._start_f)

    def _factor(self) -> float:
        """Multiplier applied to the initial value at the current step."""
        raise NotImplementedError

    def _apply(self):
        setattr(self._cfg, self._attr, self._init * self._factor())

    def step(self):
        self._step = min(self._step + 1, self._total)
        self._apply()

    @property
    def current_value(self) -> float:
        return getattr(self._cfg, self._attr)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {"_step": self._step}

    def load_state_dict(self, state_dict: dict):
        """Restore scheduler state from checkpoint.

        The restored value is applied immediately — otherwise the attribute would sit at its initial
        value for one more iteration after a resume, until the next `step()`.
        """
        self._step = state_dict.get("_step", 0)
        self._apply()


class _LinearCfgScheduler(_BaseCfgScheduler):
    """Linearly anneals a float attribute on a config object, mirroring torch LinearLR semantics."""

    def _factor(self) -> float:
        return self._start_f + (self._end_f - self._start_f) * self._step / self._total


class _ExponentialCfgScheduler(_BaseCfgScheduler):
    """Exponentially anneals a float attribute.

    Factor decays as: start_factor * (end_factor / start_factor)^(step / total_iters)
    Example: start_factor=1.0, end_factor=0.1, total_iters=1000
    - step 0: factor = 1.0
    - step 500: factor ≈ 0.316
    - step 1000: factor = 0.1
    """

    def _factor(self) -> float:
        # Exponential decay: avoid log(0) by clamping
        if self._end_f <= 0 or self._start_f <= 0:
            raise ValueError("Exponential scheduler requires start_factor and end_factor > 0")
        # factor = start_f * (end_f / start_f)^(step / total)
        exponent = self._step / self._total
        return self._start_f * ((self._end_f / self._start_f) ** exponent)


def _make_cfg_scheduler(cfg, attr: str, init_value: float, sched_dict: dict, total_iters: int) -> _BaseCfgScheduler | None:
    """Factory to create appropriate scheduler based on config dict.

    Args:
        cfg: Config object to mutate
        attr: Attribute name to annealing
        init_value: Initial value before scheduling
        sched_dict: Dict with keys: type (linear/exponential), start_factor, end_factor, total_iters (optional)
        total_iters: Default total iterations if not in sched_dict

    Returns:
        Scheduler instance or None if sched_dict is None
    """
    if sched_dict is None:
        return None

    sched_type = sched_dict.get("type", "linear").lower()
    start_f = sched_dict.get("start_factor", 1.0)
    end_f = sched_dict.get("end_factor", 0.0)
    total_i = sched_dict.get("total_iters", total_iters)

    if sched_type == "linear":
        return _LinearCfgScheduler(cfg, attr, init_value, start_factor=start_f, end_factor=end_f, total_iters=total_i)
    elif sched_type == "exponential":
        return _ExponentialCfgScheduler(cfg, attr, init_value, start_factor=start_f, end_factor=end_f, total_iters=total_i)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}. Choose from: linear, exponential")
