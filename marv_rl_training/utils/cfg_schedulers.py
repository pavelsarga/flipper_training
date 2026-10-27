"""Generic config-attribute schedulers, shared across training scripts (PPO, D3QN, ...).

Each scheduler mutates a single float attribute on an arbitrary config/object
(e.g. env_cfg.step_penalty, policy.epsilon) on every `.step()` call, mirroring
torch.optim.lr_scheduler semantics but decoupled from any specific optimizer.
"""

import math


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


class SRGatedCoefScheduler:
    """Hysteresis controller that ramps a config float attribute toward a target based on
    eval success rate, rather than a fixed iteration count.

    Motivation: a pure iteration-based schedule (see the classes above) has to guess in
    advance how long training takes to reach a given performance level, which varies wildly
    across coefficient values -- exactly the quantity this is meant to tune. Instead this
    reacts to the actual eval/success_rate reported at each mid-training eval:

      SR >= sr_threshold  -> advance one ramp step toward target_value
      SR <  sr_floor       -> retreat one ramp step back toward init_value
      otherwise            -> hold

    A single step size is used both ways, so a noisy bad eval costs at most one notch, not a
    full reset -- the point is a self-correcting controller, not a one-shot trigger. Call
    ``on_eval(success_rate)`` once per mid-training eval, not once per training iteration.
    """

    def __init__(
        self,
        cfg,
        attr: str,
        init_value: float,
        target_value: float,
        sr_threshold: float,
        sr_floor: float,
        ramp_evals: int = 10,
    ):
        if sr_floor >= sr_threshold:
            raise ValueError(f"sr_floor ({sr_floor}) must be < sr_threshold ({sr_threshold}) or the controller oscillates every eval")
        self._cfg = cfg
        self._attr = attr
        self._init = init_value
        self._target = target_value
        self._sr_threshold = sr_threshold
        self._sr_floor = sr_floor
        self._ramp_evals = max(1, ramp_evals)
        self._progress = 0  # 0 = init_value, ramp_evals = target_value
        self._apply()

    def _apply(self):
        frac = self._progress / self._ramp_evals
        setattr(self._cfg, self._attr, self._init + (self._target - self._init) * frac)

    def on_eval(self, success_rate: float, frames: int | None = None) -> None:
        if success_rate >= self._sr_threshold:
            self._progress = min(self._progress + 1, self._ramp_evals)
        elif success_rate < self._sr_floor:
            self._progress = max(self._progress - 1, 0)
        self._apply()

    @property
    def current_value(self) -> float:
        return getattr(self._cfg, self._attr)

    @property
    def progress_frac(self) -> float:
        return self._progress / self._ramp_evals

    def state_dict(self) -> dict:
        return {"_progress": self._progress}

    def load_state_dict(self, state_dict: dict):
        self._progress = state_dict.get("_progress", 0)
        self._apply()


class SlowStartCoefScheduler:
    """TCP-congestion-control-style ramp of a config float attribute, driven by eval SR.

    Why not SRGatedCoefScheduler: its gates are absolute SR levels, so a policy that can reach
    0.85 keeps being pushed while it slides from 0.83 to 0.70 (every eval still passes 0.7),
    and the retreat only fires below sr_floor -- the smoothness x3 runs (2026-10-23) rode
    the coef to its cap and lost 0.15 SR without one retreat. This one reacts to the SR
    TREND instead and finds the coefficient the policy can carry, the way TCP finds the
    congestion window a link can carry:

      wait        coef = init until frames >= min_frames and SR >= sr_min
      slow start  progress 1, 2, 4, ... (doubling per eval) up to ssthresh
      linear      progress += 1 per eval, up to the cap (target_value)
      loss        least-squares slope of SR over the last slope_window evals < -drop_slope:
                    in slow start: ssthresh = progress // 2, progress = ssthresh, go linear
                    in linear:     progress -= linear_drop_steps
                  then hold for cooldown_evals and refill the slope window with post-loss
                  evals only, so one drop is not counted again by the windows that still
                  contain it
      timeout     SR < sr_min after entry: ssthresh = progress // 2, progress = 0, wait

    coef = init + step * progress, clipped to target_value. Call ``on_eval(sr, frames)``
    once per mid-training eval with the cumulative frame count.
    """

    PHASES = ("wait", "slow_start", "linear")

    def __init__(
        self,
        cfg,
        attr: str,
        init_value: float,
        target_value: float,
        step: float,
        ssthresh_init: int = 4,
        min_frames: int = 15_000_000,
        sr_min: float = 0.45,
        slope_window: int = 3,
        drop_slope: float = 0.015,
        linear_drop_steps: int = 2,
        cooldown_evals: int = 2,
    ):
        if step <= 0 or target_value <= init_value:
            raise ValueError(f"need step > 0 and target_value > init_value (got step={step}, {init_value} -> {target_value})")
        if slope_window < 2:
            raise ValueError("slope_window must be >= 2 to fit a line")
        self._cfg = cfg
        self._attr = attr
        self._init = init_value
        self._target = target_value
        self._step = step
        # ceil, so the last step is clipped to the cap; rounded first because 0.15 / 0.015 is 10.000000000000002
        self._max_progress = math.ceil(round((target_value - init_value) / step, 9))
        self._min_frames = min_frames
        self._sr_min = sr_min
        self._window = slope_window
        self._drop_slope = drop_slope
        self._linear_drop = linear_drop_steps
        self._cooldown_evals = cooldown_evals
        self._progress = 0
        self._ssthresh = min(ssthresh_init, self._max_progress)
        self._phase = "wait"
        self._cooldown = 0
        self._history: list[float] = []
        self._last_slope = float("nan")
        self._losses = 0
        self._apply()

    def _apply(self):
        setattr(self._cfg, self._attr, min(self._init + self._step * self._progress, self._target))

    def _slope(self) -> float:
        n = len(self._history)
        xm = (n - 1) / 2
        num = sum((i - xm) * y for i, y in enumerate(self._history))
        return num / sum((i - xm) ** 2 for i in range(n))

    def _grow(self):
        if self._phase == "slow_start":
            self._progress = min(max(1, 2 * self._progress), self._ssthresh)
            if self._progress >= self._ssthresh:
                self._phase = "linear"
        else:
            self._progress = min(self._progress + 1, self._max_progress)

    def on_eval(self, success_rate: float, frames: int | None = None) -> None:
        self._history = (self._history + [success_rate])[-self._window:]
        full = len(self._history) == self._window
        self._last_slope = self._slope() if full else float("nan")
        if self._phase == "wait":
            if (frames is None or frames >= self._min_frames) and success_rate >= self._sr_min:
                self._phase = "slow_start" if self._progress < self._ssthresh else "linear"
                self._grow()
        elif success_rate < self._sr_min:
            self._ssthresh = max(self._progress // 2, 1)
            self._progress = 0
            self._phase = "wait"
            self._cooldown = 0
            self._history = []
            self._losses += 1
        elif self._cooldown > 0:
            self._cooldown -= 1
        elif full and self._last_slope < -self._drop_slope:
            if self._phase == "slow_start":
                self._ssthresh = max(self._progress // 2, 1)
                self._progress = self._ssthresh
                self._phase = "linear"
            else:
                self._progress = max(self._progress - self._linear_drop, 0)
            self._cooldown = self._cooldown_evals
            self._history = []
            self._losses += 1
        else:
            self._grow()
        self._apply()

    @property
    def current_value(self) -> float:
        return getattr(self._cfg, self._attr)

    @property
    def progress_frac(self) -> float:
        return self._progress / self._max_progress

    def diagnostics(self, prefix: str) -> dict[str, float]:
        """Controller state for the training log, keys ``<prefix>_<name>``."""
        return {
            f"{prefix}_progress": float(self._progress),
            f"{prefix}_ssthresh": float(self._ssthresh),
            f"{prefix}_phase": float(self.PHASES.index(self._phase)),
            f"{prefix}_sr_slope": self._last_slope,
            f"{prefix}_losses": float(self._losses),
        }

    def state_dict(self) -> dict:
        return {
            "_progress": self._progress, "_ssthresh": self._ssthresh, "_phase": self._phase,
            "_cooldown": self._cooldown, "_history": list(self._history), "_losses": self._losses,
        }

    def load_state_dict(self, state_dict: dict):
        self._progress = state_dict.get("_progress", 0)
        self._ssthresh = state_dict.get("_ssthresh", self._ssthresh)
        self._phase = state_dict.get("_phase", "wait")
        self._cooldown = state_dict.get("_cooldown", 0)
        self._history = list(state_dict.get("_history", []))
        self._losses = state_dict.get("_losses", 0)
        self._apply()
