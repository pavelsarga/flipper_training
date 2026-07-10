"""AT-D3QN reward (Pan et al. 2023), Sec. III-D EXACTLY: R_flipper (Eq. 4) + R_pitch (Eq. 6)
+ R_end (Eq. 7), combined via Eq. 10's weighted sum.

Pan, H. et al. "Deep Reinforcement Learning for Flipper Control of Tracked Robots,"
arXiv:2306.10352 (AT-D3QN), Sec. III-D. Equation numbers below refer to this paper; the
Remote Sensing 15(18):4616 journal version (ICM-D3QN) restates the SAME three terms
word-for-word in its Sec. 3.4 (Eq. 4-9) and ADDS a fourth term, R_contact (its Eq. 8),
weighted by kappa_3 in its Eq. 10. This module implements ONLY the three terms the task
scopes it to (R_flipper + R_pitch + R_end, matching AT-D3QN Sec. III-D exactly) -- R_contact
is out of scope and not implemented here. Table 2 (from the journal paper, the only one of
the two papers that publishes a hyperparameter table) is nonetheless the natural source for
this module's default coefficients, since both papers share the same authors and the same
R_flipper/R_pitch/R_end equations verbatim: lambda1=0.1, lambda2=0.33, kappa1=kappa2=kappa3=0.005
(we default kappa_flipper=kappa_pitch=0.005, dropping kappa3/R_contact).

Eq. 10 (R_contact term dropped, see above): ``R^e_t = R_end + kappa_flipper*R_flipper + kappa_pitch*R_pitch``.

What is transcribed EXACTLY vs. what had to be operationalized
----------------------------------------------------------------
Eq. 4, 5, 6, 7 are closed-form and are transcribed literally (see each method's docstring for
the one deliberate reading of Eq. 4's "+-pi/36" notation, which is genuinely ambiguous as
written -- see ``_r_flipper``). Two things the paper describes only qualitatively/by figure,
not by formula, had to be given a concrete, documented implementation:

* **theta*_f1, the "candidate angle"** (Fig. 4): the paper describes it as "the angle with
  the largest value from the hinge to an expanded terrain point in front of the flipper" but
  never gives T_f/T_bf/T_br/T_b/T_r (its labeled point-cloud sub-regions) a closed-form
  definition -- they are diagram-only regions. ``_candidate_angle`` implements the single
  stated RULE (max angle from the hinge to a "thickness-expanded" forward terrain point)
  over our own operational definition of the forward look-ahead window (see its docstring).
* **"got stuck"** (Eq. 7/9's 4th branch): no formula or threshold is given anywhere in either
  paper (checked via ``pdftotext`` over both PDFs -- no numeric value for a stuck condition
  exists in the text). ``_stuck`` implements a documented, configurable no-forward-progress
  latch (``stuck_window`` steps of less than ``stuck_min_progress`` meters of movement).

Also: **R_flipper is FRONT-flipper-only in the paper.** Sec. III-D's own words: "we designed a
motion-based reward function SPECIFICALLY FOR THE FRONT FLIPPER... We denote the reward of
front flippers as R_flipper" (Eq. 4). There is no rear-flipper analogue given anywhere in
Sec. III-D. We do not invent an unstated symmetric term; ``apply_flipper_reward_to_rear``
defaults to ``False`` (paper-faithful) and is offered purely as an opt-in extension.

Architectural note: R_end (Eq. 7/9)
-------------------------------------
In the paper, all four of R_end's conditions coincide EXACTLY with episode termination (it is
literally "Reward of Terminate", paid once, at the step the episode ends). In this framework,
only the env's ``Objective`` can end an episode (``Env._step``: ``terminated = failed |
reached_goal``, plus step-limit ``truncated``) -- a ``Reward`` cannot. So:

* "reached" -> the ``success`` flag this reward already receives from the Objective, which
  DOES coincide with real termination.
* "t >= t_max" -> ``env.step_count >= env.step_limits`` (or the ``t_max`` override), which
  also coincides with real truncation (same condition ``Env._step`' computes ``truncated`` from).
* "|theta_R| >= pi/3" -> computed directly here from live state (paper-exact), independent of
  whatever the Objective's own ``fail`` condition is. This does NOT necessarily coincide with
  real termination -- if your Objective's ``max_feasible_pitch`` (or equivalent) is not also
  set to ``pi/3`` (or tighter), the episode will keep running past this step, and this branch
  will re-fire ``-R`` every qualifying step instead of exactly once. Configure the Objective to
  match if you need literal once-only semantics.
* "stuck" -> our own no-progress LATCH (fires ``-R`` once, then stays silent until the next
  reset), specifically to approximate the paper's "pay once, at the end" semantics as closely
  as this architecture allows, even though (like the pitch branch) it cannot itself end the
  episode.

The passed-in ``fail`` argument (the Objective's own terminal-failure flag) is deliberately
NOT consumed here -- its exact meaning is Objective-specific (e.g. excess roll) and not one of
Eq. 7/9's four named conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import quaternion_to_pitch
from flipper_training.observations.pan_terrain import (
    resolve_front_rear_indices,
    resolve_front_rear_hinges,
    sample_terrain_points_relative,
)

from . import Reward

__all__ = ["PanReward"]


@dataclass
class PanReward(Reward):
    """AT-D3QN's R_flipper + R_pitch + R_end (Sec. III-D, Eq. 4/6/7), Eq. 10 weighted sum
    (R_contact/kappa3 dropped -- see module docstring). Defaults are Table 2 (journal paper)
    where given; fields documented "NOT given a numeric value by the paper" are our own,
    clearly-flagged choices.

    Args:
        lambda1: Eq. 4's threshold coefficient (Table 2: 0.1).
        flipper_tolerance: the "+-pi/36" (+-5 deg) term in Eq. 4, read as a symmetric
            DEAD-ZONE around the candidate angle theta*_f1 (matches the surrounding prose:
            "conducive to reducing the little and meaningless action exploration") --
            Delta_theta_f1 = max(0, |theta_f1 - theta*_f1| - flipper_tolerance).
        candidate_lookahead: meters ahead of (behind, for the rear) the flipper hinge searched
            for the candidate angle theta*_f1/theta*_f2. NOT given a numeric value by the
            paper; default 0.4 m ~= MARV's flipper reach (0.3815 m, ENVIRONMENT geometry).
        candidate_samples: number of terrain points sampled over that look-ahead window.
        robot_half_thickness: the paper's "B", the half-thickness the candidate terrain point
            is "expanded" by (Fig. 4). NOT given a numeric value by the paper; default 0.1 m
            = MARV's flipper track wheel outer radius (robots/marv.yaml driving_parts wheel
            radius [0.1, 0.065]). We apply this expansion as a vertical (+Z) offset, not a
            true terrain-normal offset -- an approximation, see module docstring.
        apply_flipper_reward_to_rear: opt-in, NON-paper extension -- see module docstring.
        lambda2: Eq. 6's threshold coefficient (Table 2: 0.33).
        pitch_smoothness_window: k, the window length in Eq. 5's Delta_theta_R^k. NOT given a
            numeric value by the paper. Implemented as a CAUSAL trailing window (the k most
            recent per-step |Delta theta_R| terms), since the paper's forward-looking
            ``i=t..t+k-1`` cannot be evaluated online at step t.
        pitch_hard_limit: the pi/4 in Eq. 6's first branch.
        settlement_reward: "R" in Eq. 7/9. NOT given a numeric value by the paper.
        pitch_overturn_threshold: the pi/3 in Eq. 7/9's second branch.
        t_max: episode step budget for Eq. 7/9's "t >= t_max" branch. ``None`` (default) uses
            the env's own ``step_limits`` (set by the Objective) -- the honest choice, since
            that is what "t_max... the maximum number of steps the robot performs in a single
            terrain traversal round" (the paper's own words) already IS in this framework.
        stuck_window / stuck_min_progress: our own operationalization of "got stuck" -- see
            module docstring. No formula is given by the paper.
        kappa_flipper / kappa_pitch: Eq. 10's kappa_1 / kappa_2 (Table 2: both 0.005).
        front_indices / rear_indices: see ``pan_terrain.resolve_front_rear_indices``.
    """

    lambda1: float = 0.1
    flipper_tolerance: float = math.pi / 36
    candidate_lookahead: float = 0.4
    candidate_samples: int = 9
    robot_half_thickness: float = 0.1
    apply_flipper_reward_to_rear: bool = False

    lambda2: float = 0.33
    pitch_smoothness_window: int = 5
    pitch_hard_limit: float = math.pi / 4

    settlement_reward: float = 100.0
    pitch_overturn_threshold: float = math.pi / 3
    t_max: int | None = None
    stuck_window: int = 50
    stuck_min_progress: float = 0.02

    kappa_flipper: float = 0.005
    kappa_pitch: float = 0.005

    front_indices: tuple[int, ...] | None = None
    rear_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.pitch_smoothness_window < 2:
            raise ValueError(f"pitch_smoothness_window must be >= 2 (Eq. 5 divides by k-1), got {self.pitch_smoothness_window}.")
        robot_cfg = self.env.robot_cfg
        self.front_idx, self.rear_idx = resolve_front_rear_indices(robot_cfg, self.front_indices, self.rear_indices)
        self.front_idx = self.front_idx.to(self.env.device)
        self.rear_idx = self.rear_idx.to(self.env.device)
        self.front_x, self.front_z, self.rear_x, self.rear_z = resolve_front_rear_hinges(robot_cfg, self.front_idx.cpu(), self.rear_idx.cpu())

        n = self.env.n_robots
        device = self.env.device
        self._pitch_delta_buf = torch.zeros(n, self.pitch_smoothness_window, device=device)
        self._pitch_buf_n_valid = torch.zeros(n, dtype=torch.long, device=device)
        self._stuck_ref_pos = torch.zeros(n, 3, device=device)
        self._stuck_ref_init = torch.zeros(n, dtype=torch.bool, device=device)
        self._stuck_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._stuck_latched = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, reset_mask: torch.Tensor, training: bool):
        self._pitch_delta_buf[reset_mask] = 0.0
        self._pitch_buf_n_valid[reset_mask] = 0
        self._stuck_ref_init[reset_mask] = False
        self._stuck_steps[reset_mask] = 0
        self._stuck_latched[reset_mask] = False

    def state_dict(self) -> dict:
        return {
            "pitch_delta_buf": self._pitch_delta_buf,
            "pitch_buf_n_valid": self._pitch_buf_n_valid,
            "stuck_ref_pos": self._stuck_ref_pos,
            "stuck_ref_init": self._stuck_ref_init,
            "stuck_steps": self._stuck_steps,
            "stuck_latched": self._stuck_latched,
        }

    def load_state_dict(self, state_dict: dict):
        self._pitch_delta_buf = state_dict["pitch_delta_buf"]
        self._pitch_buf_n_valid = state_dict["pitch_buf_n_valid"]
        self._stuck_ref_pos = state_dict["stuck_ref_pos"]
        self._stuck_ref_init = state_dict["stuck_ref_init"]
        self._stuck_steps = state_dict["stuck_steps"]
        self._stuck_latched = state_dict["stuck_latched"]

    def _paper_angles(self, state: PhysicsState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """theta_f1, theta_f2, theta_R in the paper's sign convention -- see
        ``observations/pan_terrain.py``'s module docstring for the derivation."""
        theta_f1 = -state.thetas[:, self.front_idx].mean(dim=-1)
        theta_f2 = state.thetas[:, self.rear_idx].mean(dim=-1)
        theta_r = -quaternion_to_pitch(state.q)
        return theta_f1, theta_f2, theta_r

    def _candidate_angle(self, curr_state: PhysicsState, front: bool) -> torch.Tensor:
        """theta*_f1 (or theta*_f2 for the rear, if ``apply_flipper_reward_to_rear``): the
        largest angle (paper's own "positive = up" convention, same as theta_f1/theta_f2)
        from the flipper's hinge to a terrain point "expanded" by the robot's half-thickness,
        searched over ``candidate_samples`` points spanning ``candidate_lookahead`` meters
        ahead of the hinge (behind it, for the rear) -- see module docstring for what is and
        isn't literally specified by the paper here (Fig. 4).
        """
        device = curr_state.x.device
        hinge_x = self.front_x if front else self.rear_x
        hinge_z = self.front_z if front else self.rear_z
        outward_sign = 1.0 if front else -1.0
        offsets = torch.linspace(0.0, self.candidate_lookahead, self.candidate_samples + 1, device=device)[1:]  # exclude the hinge itself
        local_x = hinge_x + outward_sign * offsets
        local_y = torch.zeros_like(local_x)  # sagittal (centerline) slice, see module docstring
        z_rel = sample_terrain_points_relative(self.env, curr_state, local_x, local_y)  # (B, candidate_samples), relative to chassis
        rise = (z_rel - hinge_z) + self.robot_half_thickness  # "expanded" terrain point, Fig. 4
        run = offsets.view(1, -1).clamp_min(1e-6)  # outward distance from the hinge, always > 0 by construction
        angle = torch.atan2(rise, run)
        return angle.max(dim=-1).values  # (B,)

    def _r_flipper(self, curr_state: PhysicsState, theta_f: torch.Tensor, front: bool) -> torch.Tensor:
        """Eq. 4, transcribed exactly given the dead-zone reading of "+-pi/36" (module
        docstring's Args section spells out the reading)."""
        theta_star = self._candidate_angle(curr_state, front=front)
        delta = (theta_f - theta_star).abs().sub(self.flipper_tolerance).clamp_min(0.0)
        return torch.where(delta > 1.0 / self.lambda1, torch.full_like(delta, -1.0), -self.lambda1 * delta)

    def _r_pitch(self, theta_r_prev: torch.Tensor, theta_r_curr: torch.Tensor) -> torch.Tensor:
        """Eq. 5 + Eq. 6, transcribed exactly (Eq. 5's window made causal, see class docstring)."""
        delta_abs = theta_r_curr.abs() - theta_r_prev.abs()  # Eq. 5 line 1 (difference of absolute values, NOT abs of difference)
        step_delta = (theta_r_curr - theta_r_prev).abs()
        self._pitch_delta_buf = torch.roll(self._pitch_delta_buf, shifts=-1, dims=1)
        self._pitch_delta_buf[:, -1] = step_delta
        self._pitch_buf_n_valid = (self._pitch_buf_n_valid + 1).clamp_max(self.pitch_smoothness_window)
        window_avg = self._pitch_delta_buf.sum(dim=1) / (self.pitch_smoothness_window - 1)  # Eq. 5 line 2, literal 1/(k-1)
        enough_history = self._pitch_buf_n_valid >= 2

        cond1 = (theta_r_curr.abs() > self.pitch_hard_limit) & (delta_abs > 0)
        cond2 = enough_history & (window_avg > 1.0 / self.lambda2)
        linear = torch.where(enough_history, -self.lambda2 * window_avg, torch.zeros_like(window_avg))
        return torch.where(cond1, torch.full_like(window_avg, -1.0), torch.where(cond2, torch.full_like(window_avg, -1.0), linear))

    def _r_end(self, curr_state: PhysicsState, theta_r_curr: torch.Tensor, success: torch.Tensor) -> torch.Tensor:
        """Eq. 7/9, transcribed exactly for the 3 conditions the paper gives a formula for
        (reached / |theta_R|>=pi/3 / t>=t_max); "stuck" is our own operationalization -- see
        module/class docstrings, including the architectural caveats about what "coincides
        with termination" means in this framework.
        """
        overturned = theta_r_curr.abs() >= self.pitch_overturn_threshold
        t_max = self.env.step_limits if self.t_max is None else torch.full_like(self.env.step_limits, self.t_max)
        timeout = self.env.step_count >= t_max

        uninit = ~self._stuck_ref_init
        if uninit.any():
            self._stuck_ref_pos[uninit] = curr_state.x[uninit].detach()
            self._stuck_ref_init[uninit] = True
        moved = (curr_state.x.detach() - self._stuck_ref_pos).norm(dim=-1)
        progressed = moved >= self.stuck_min_progress
        self._stuck_ref_pos[progressed] = curr_state.x[progressed].detach()
        self._stuck_steps = torch.where(progressed, torch.zeros_like(self._stuck_steps), self._stuck_steps + 1)
        stuck_now = (self._stuck_steps >= self.stuck_window) & ~self._stuck_latched
        self._stuck_latched = self._stuck_latched | stuck_now

        r = torch.zeros_like(theta_r_curr)
        r = torch.where(timeout, torch.full_like(r, -self.settlement_reward), r)
        r = torch.where(stuck_now, torch.full_like(r, -self.settlement_reward), r)
        r = torch.where(overturned, torch.full_like(r, -self.settlement_reward), r)
        r = torch.where(success, torch.full_like(r, self.settlement_reward), r)  # reached takes precedence
        return r

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
        theta_f1_prev, _theta_f2_prev, theta_r_prev = self._paper_angles(prev_state)
        theta_f1, theta_f2, theta_r = self._paper_angles(curr_state)

        r_flipper = self._r_flipper(curr_state, theta_f1, front=True)
        if self.apply_flipper_reward_to_rear:
            r_flipper = r_flipper + self._r_flipper(curr_state, theta_f2, front=False)

        r_pitch = self._r_pitch(theta_r_prev, theta_r)
        r_end = self._r_end(curr_state, theta_r, success)

        reward = r_end + self.kappa_flipper * r_flipper + self.kappa_pitch * r_pitch
        return reward.unsqueeze(-1).to(self.env.out_dtype)
