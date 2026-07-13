"""Mitriakov et al. staircase-negotiation reward, transcribed from the AUTHORS' OWN RELEASED
CODE (``gwaxG/robot_ws``, cloned read-only at ``/home/cnuc/upstream_refs/mitriakov2022_robot_ws``),
not just Mitriakov, Papadakis, Kerdreux & Garlatti (2021) "Reinforcement Learning Based,
Staircase Negotiation Learning" (IEEE RA Magazine 28(4))'s Eq. 3-6 prose. The paper's Eq. 3-6
describe the reward at a level the actual ROS graph does not implement in one place -- the
gym-training `TrainingEnv.step()` (gym-training/gym_training/envs/training_env.py:322-329) is a
thin RPC client: it publishes the action, sleeps ``ACTION_TIME=0.25`` s (line 27), then calls the
``/rollout/step_return`` *service* and returns whatever float comes back. All of the actual reward
bookkeeping lives server-side in the ``monitor`` ROS node. This module transcribes THAT code.

File:line provenance (mitriakov2022_robot_ws)
----------------------------------------------
monitor/scripts/monitor_app/monitor_app.py
  - ``callback_odometry`` (227-249): progress bookkeeping. ``dist = ||odom - goal|| - 0.3``
    (the "0.3" shift is ``monitor/scripts/monitor_app/utils.py::get_distance`` line 100-101,
    a fixed standoff, not exposed anywhere else). While ``dist`` beats the BEST (lowest) distance
    reached so far this episode AND cumulative progress is still ``< 1.0``: reward +=
    ``(closest_distance_before - dist) / maximum_distance``, progress += the same increment
    (lines 237-239). ``maximum_distance`` is the odom-goal distance measured once, at rollout
    start (``callback_start_rollout``, line 89-91).

    SEMANTICS -- CORRECTED after independent verification (2026-07-13): upstream is
    ONE-STEP-LOOKBACK, POSITIVE-ONLY progress shaping. ``callback_odometry`` pays
    ``max(0, closest_distance - dist) / maximum_distance`` and then, in its
    ``else`` branch, UNCONDITIONALLY overwrites ``closest_distance = dist`` on
    every non-terminal tick -- ``closest_distance`` is therefore just the
    PREVIOUS tick's distance, not a historical minimum. Consequences faithfully
    reproduced here: moving away from the goal is never penalized (positive-only),
    and backtrack-then-retrace is RE-rewarded for re-covered ground (the
    mechanism is exploitable; that is the authors' code, so we keep it).
    An earlier revision of this class implemented a strict running minimum
    (rewarding only new best distances) and described upstream as monotonic
    "best-distance-ever" shaping -- both wrong; fixed. It remains a materially
    different formula from this repo's own ``PotentialGoal*`` family (telescoping
    ``gamma*phi(s')-phi(s)`` shaping, which rewards/penalizes every delta
    symmetrically) -- the reason this needed its own class.
  - ``callback_odometry`` (250-273): tip-over detection -- ``|roll| > pi/2`` or ``|pitch| > pi/2``
    (the accident-message strings conflate roll/pitch with "front/rear" vs "left/right" tipping,
    but the numeric thresholds are unambiguous). Sets ``done=True``.
  - ``callback_step_return`` (158-180): once per control tick, returns the accumulated progress
    reward, reshaped by ``Guidance.reshape_reward`` (see below), MINUS a flat ``1.0`` if
    "tipping over" is in ``accidents`` (line 164-165).
monitor/scripts/monitor_app/guidance.py
  - ``reshape_reward`` (130-139) + ``update`` (68-98): IF a stability penalty is enabled for the
    task (``penalty_deviation`` for ascent, ``penalty_angular`` for descent -- see
    ``backend/scripts/learning_scripts/stables3_template.json``: the ascent template sets
    ``penalty_deviation: true``, the launched descent example
    (``stables3_launch.json``) sets ``penalty_angular: true``), a per-tick safety signal (see
    ``safety.py`` below) is subtracted from the reward -- but ONLY after an ADAPTIVE
    normalization constant K has been calibrated: for the first ``start_size=30`` *episodes* of
    a run, NO penalty is subtracted at all (used purely to observe the raw penalty scale); K is
    then fixed as ``1 / mean(episode penalty sums over those 30 episodes)`` and used for the
    remainder of the run (``update``, line 77-82). This calibration scheme is inherently
    SEQUENTIAL / single-environment (``backend/scripts/learning_scripts/stables3_launch.py`` (Learner; an earlier revision mis-cited policies.py) wraps
    exactly ONE ``TrainingEnv`` in a ``DummyVecEnv``) -- it has no defined analogue for a 256-way
    *vectorized* batch of parallel episodes (per-robot calibration vs. one calibration pooled
    across the whole batch is a modeling choice their code never had to make, since it never ran
    more than one episode at a time). NOT transcribed here -- see "REMAINING GAP" below. A FIXED
    ``ascent_stability_coef`` / ``descent_stability_coef`` is used in its place.
monitor/scripts/safety/safety.py
  - ``descent_task_penalty`` (67-74): the descent-task safety SIGNAL = ``0.5 * (mean IMU-linear-
    acceleration-derived "shake" over the last 0.1 s tick + a COG-deviation-from-support-line
    proxy)``.
  - ``broadcast_cog_projections`` (147-176): the ascent-task safety SIGNAL = the robot's center-
    of-gravity ground-projection distance from the currently-traversed stair tread's own
    inclined-frame origin -- a literal 3-D geometry computation requiring a hand-built TF frame
    PER CURRENTLY-OCCUPIED STAIR TREAD (their sim broadcasts ``/p_cent``, ``/p_cent_inc`` etc.
    every 0.25 s via ``simulation/scripts/env_generation/env_gen_services.py::broadcast_stair``,
    lines 102-159). Neither signal has a literal analogue here: both depend on runtime machinery
    (an external marker/TF rig identifying which physical tread the robot currently overhangs)
    that this framework's terrain representation -- a scalar heightmap plus a ``step_indices``
    mask, see ``rl_objectives/stair_crossing.py`` -- was not built to expose, and inventing that
    machinery from scratch is out of this task's scope ("do not invent physics"). This module
    instead reuses |roll| (ascent) / |pitch rate| (descent) -- the SAME phase-agnostic proxies
    this repo's own ``PotentialGoalWithPenaltiesConfigurable.roll_coef`` /
    ``.pitch_rate_coef`` already compute elsewhere in this codebase -- as the closest available,
    terrain-and-robot-agnostic stand-ins. This is an ADMITTED STAND-IN, not a transcription of
    Eq. 5's ``D = sqrt(d^2 + h^2)`` COG/support-polygon formula.

    The one formula-level improvement this class DOES make relative to the previous
    (adaptation-only) config: the paper's penalty is PHASE-CONDITIONAL -- ascent gets ONLY the
    COG-deviation term (``penalty_deviation``), descent gets ONLY the shake term
    (``penalty_angular``); a single rollout never applies both. The old config's
    ``PotentialGoalWithPenaltiesConfigurable`` applied roll AND pitch(-rate) penalties
    UNCONDITIONALLY on every step regardless of ascent/descent. This class gates the (still
    proxy-based) penalty on ascent-vs-descent using the terrain's own ``step_indices`` (see
    ``_is_ascent``), which the objective (``StairCrossing``) already threads through
    ``terrain_cfg.grid_extras`` for exactly this purpose (see
    ``PotentialGoalWithStepAscentBonus`` in ``rewards.py`` for the established pattern this
    reuses).

REMAINING GAP (not transcribed, and why)
-----------------------------------------
1. The adaptive per-run K normalization (guidance.py:77-91) -- see above; architecture mismatch
   between their sequential single-env training loop and this framework's vectorized batch, not
   a missing config knob.
2. The exact COG/support-polygon (ascent) and IMU-accelerometer-shake (descent) SIGNALS
   (safety.py) -- replaced by |roll| / |pitch rate| proxies, as above.
3. Two of the monitor's OTHER termination paths -- ``dist < 0`` (reached, with the 0.3 m shift)
   and ``dist > 1.2 * maximum_distance`` ("wandered too far away", monitor_app.py:240-247) -- have
   no analogue in ``StairCrossing`` (whose own success/fail conditions are a plain distance
   threshold and a roll/pitch limit, ``rl_objectives/stair_crossing.py::check_reached_goal`` /
   ``check_terminated_wrong``). A ``Reward`` cannot itself create a termination in this framework
   (see ``pan_reward.py``'s module docstring for the same architectural note) -- this is a
   property of the shared ``StairCrossing`` objective, out of scope for a reward-only module.
4. Their tip-over test (|roll|>pi/2 or |pitch|>pi/2, monitor_app.py:259-271) is instead obtained
   by setting ``StairCrossing``'s ``max_feasible_pitch``/``max_feasible_roll`` to exactly
   ``pi/2`` in ``objective_opts`` (see ``mitriakov.yaml``) -- so this class's ``failed_reward``
   (applied via the shared ``fail`` flag every ``Reward`` receives) reproduces their flat ``-1``
   exactly, without duplicating the roll/pitch check here.
"""

import math
from dataclasses import dataclass

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import quaternion_to_roll, quaternion_to_pitch

from . import Reward

__all__ = ["MitriakovStaircaseReward"]


@dataclass
class MitriakovStaircaseReward(Reward):
    """Mitriakov et al. (2021/2022) staircase reward -- see module docstring for full
    file:line provenance against the authors' own code.

    Args:
        goal_reached_reward: extra terminal bonus on success. Their code has NO such bonus
            (progress simply asymptotes to ~1.0 as the robot nears the goal) -- default 0.0 for
            fidelity; exposed only because every ``Reward`` in this repo's convention carries one.
        failed_reward: flat penalty on failure. Their code's flat tip-over penalty is exactly
            ``-1.0`` (monitor_app.py:164-165) -- see module docstring point 4 for how the tip-over
            CONDITION itself is reproduced via ``StairCrossing.max_feasible_pitch/roll``, not here.
        progress_coef: multiplies the (dimensionless, already-normalized-by-maximum_distance)
            progress increment. Their code's implicit coefficient is exactly 1.0
            (monitor_app.py:239: ``reward += diff / maximum_distance``, no extra scaling).
        goal_distance_shift: the fixed standoff subtracted from the raw robot-goal distance before
            any progress bookkeeping (``utils.py::get_distance`` line 100-101: ``... - shift``,
            ``shift = 0.3``).
        ascent_stability_coef: fixed stand-in for their auto-calibrated ascent-only COG-deviation
            penalty coefficient (K_D) -- see module docstring's REMAINING GAP point 1. Applied to
            ``|roll| / pi``.
        descent_stability_coef: fixed stand-in for their auto-calibrated descent-only shake
            penalty coefficient (K_W). Applied to ``|pitch_rate| / pi``.
    """

    goal_reached_reward: float = 0.0
    failed_reward: float = -1.0
    progress_coef: float = 1.0
    goal_distance_shift: float = 0.3
    ascent_stability_coef: float = 0.8
    descent_stability_coef: float = 0.4

    def __post_init__(self):
        n = self.env.n_robots
        device = self.env.device
        # Sentinels: max_dist < 0 means "not yet initialized this episode" (set on reset,
        # lazily filled in on the first __call__ of the new episode -- see below).
        self._max_dist = torch.full((n,), -1.0, device=device)
        self._closest_dist = torch.full((n,), float("inf"), device=device)
        self._progress = torch.zeros(n, device=device)

    def reset(self, reset_mask: torch.Tensor, training: bool):
        self._max_dist[reset_mask] = -1.0
        self._closest_dist[reset_mask] = float("inf")
        self._progress[reset_mask] = 0.0

    def state_dict(self) -> dict:
        return {
            "max_dist": self._max_dist,
            "closest_dist": self._closest_dist,
            "progress": self._progress,
        }

    def load_state_dict(self, state_dict: dict):
        self._max_dist = state_dict["max_dist"]
        self._closest_dist = state_dict["closest_dist"]
        self._progress = state_dict["progress"]

    def _is_ascent(self, start_state: PhysicsState, goal_state: PhysicsState) -> torch.Tensor:
        """Ascent vs. descent, from the terrain's own step_indices mask (same source
        ``PotentialGoalWithStepAscentBonus`` in ``rewards.py`` uses) -- goal on a higher step
        than start means ascent. Recomputed every call (start/goal are constant within an
        episode) rather than cached at reset, to avoid stale-cache-across-resets bugs.
        """
        terrain_cfg = self.env.terrain_cfg
        if terrain_cfg.grid_extras is None or "step_indices" not in terrain_cfg.grid_extras:
            raise ValueError("MitriakovStaircaseReward requires 'step_indices' in terrain_cfg.grid_extras (use a stairs heightmap generator).")
        step_indices = terrain_cfg.grid_extras["step_indices"]  # (B, H, W)
        b_range = torch.arange(self.env.n_robots, device=self.env.device)
        start_ij = terrain_cfg.xy2ij(start_state.x[..., :2])
        goal_ij = terrain_cfg.xy2ij(goal_state.x[..., :2])
        start_step = step_indices[b_range, *start_ij.unbind(1)]
        goal_step = step_indices[b_range, *goal_ij.unbind(1)]
        return goal_step > start_step

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        # --- progress term (monitor_app.py:227-249, utils.py:92-101) ---
        curr_dist = torch.linalg.norm(goal_state.x - curr_state.x, dim=-1) - self.goal_distance_shift
        prev_dist = torch.linalg.norm(goal_state.x - prev_state.x, dim=-1) - self.goal_distance_shift

        # NOTE: every buffer update below uses in-place MASKED assignment (`buf[mask] = ...`)
        # rather than reassigning the attribute (`self._buf = torch.where(...)`). The collector's
        # rollout loop runs under `torch.inference_mode()`; reassigning would rebind `self._buf`
        # to a freshly-allocated "inference tensor", and a later in-place write from OUTSIDE that
        # mode (e.g. `reset()`, called from `_reset()` outside the collector's inference context)
        # then raises `RuntimeError: Inplace update to inference tensor outside InferenceMode is
        # not allowed` -- caught by actually running a multi-iteration (not just 1-step) smoke
        # test, which triggers a real env reset mid-rollout; a 1-step smoke test does not. Kept
        # as the established in-place pattern `pan_reward.py` already uses successfully.
        uninitialized = self._max_dist < 0.0
        if uninitialized.any():
            # First tick of a (newly reset) episode: "maximum_distance" is measured once, at
            # rollout start (callback_start_rollout:89-91), from the pre-step (start) state.
            init_dist = prev_dist.clamp_min(1e-6)
            self._max_dist[uninitialized] = init_dist[uninitialized]
            self._closest_dist[uninitialized] = prev_dist[uninitialized]

        # VERIFIED-EXACT semantics (audit correction, 2026-07-13): upstream
        # monitor_app.py:227-249 rewards max(0, prev_closest - dist)/max_dist and
        # then overwrites closest_distance = dist UNCONDITIONALLY on every
        # non-terminal tick (the `else: closest_distance = dist` branch) -- it is
        # ONE-STEP-LOOKBACK positive-only progress shaping, NOT a running
        # historical minimum. A previous revision of this file implemented a
        # strict running min (only updating on improvement), which is a stricter,
        # different formula: it makes re-advancing over backtracked ground
        # unrewarded, whereas the authors' code re-rewards it. We reproduce the
        # authors' literal mechanism here, exploitability and all.
        diff = (self._closest_dist - curr_dist).clamp_min(0.0)
        still_climbing = self._progress < 1.0
        progress_increment = torch.where(still_climbing, diff / self._max_dist, torch.zeros_like(diff))
        self._closest_dist[:] = curr_dist  # unconditional, per upstream's else-branch
        self._progress += progress_increment
        self._progress.clamp_(0.0, 1.0)

        reward = (self.progress_coef * progress_increment).unsqueeze(-1)

        # --- phase-conditional stability penalty (safety.py, admitted proxy -- see module docstring) ---
        is_ascent = self._is_ascent(start_state, goal_state)
        roll_term = quaternion_to_roll(curr_state.q).abs() / math.pi
        pitch_rate_term = curr_state.omega[..., 1].abs() / math.pi
        stability_penalty = torch.where(is_ascent, self.ascent_stability_coef * roll_term, self.descent_stability_coef * pitch_rate_term)
        reward = reward - stability_penalty.unsqueeze(-1)

        # --- terminal bonuses/penalties (monitor_app.py:164-165 tip-over -1, via `fail`) ---
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)
