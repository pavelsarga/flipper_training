"""Constrained REPS (C-REPS) trainer for the Pecka et al. (2016, IROS) linear
positional flipper policy (``flipper_training.policies.pecka_policy``).

Adds the episodic black-box policy-search loop the paper actually uses -- there
is no gradient descent anywhere in this file. Implements Algorithm 1 / eqs
1-11 of "Autonomous Flipper Control with Safety Constraints", specialized to
the paper's own experimental setting: context fixed to zero (Sec IV-A "e)
Context": "we did not make use of the context -- it was always set to
zeros"), which is why the dual below only has ``(eta, gamma)``, not
``(eta, gamma, theta)`` -- see "Context-free specialization" below.

What each iteration does (Algorithm 1, one outer loop = one "for k=1..K" line):
  1. Sample K = ``num_robots`` parameter vectors ``omega_i ~ q(mu, Sigma)``,
     ONE PER ROBOT ROW (the env is already batched over robots, so this maps
     Algorithm 1's inner ``for i=1..N`` loop onto the existing vectorization
     instead of looping in Python -- "map one omega per robot row").
  2. Roll out ONE fixed-length episode with all K robots in parallel, robot
     row i driven throughout by ``omega_i`` (via
     ``pecka_policy._PeckaLinearCore.compute_action``, the exact same function
     the deployed actor calls -- see that module's docstring). Collect the
     PER-EPISODE return ``R_i`` (summed over the whole window) and the
     PER-EPISODE binary safety ``S_i`` (Sec III's cautious-simulator proxy,
     scoped by this fix to a tilt threshold -- see ``max_tilt`` below). This
     is the credit-assignment fix: the previous implementation weighted
     per-STEP cumulative returns; C-REPS eq. (1)/(11) is defined per EPISODE.
  3. Solve the constrained dual for ``(eta, gamma)`` (``solve_creps_dual``
     below) and read off the eq. (11) sample weights.
  4. Refit ``(mu, Sigma)`` by weighted maximum likelihood (Algorithm 1's
     "Update upper-level policy").

Context-free specialization of the dual (eqs 4/6/7 -> the code below)
------------------------------------------------------------------------
The full paper dual is ``g(eta,gamma,theta) = eta*log(sum_i q_i * exp((R_i -
theta.phi(s_i) - gamma.1 + gamma.C_i)/eta)) + eta*epsilon + theta.phi_hat +
gamma.delta`` (eq. 6/7), with ``theta`` the multiplier for the context
feature-matching constraint ``sum p(s,w) phi(s) = phi_hat`` (eq. 3, line 4).
When every sample shares the SAME (zero) context, ``phi(s_i)`` is a constant
``c`` for all ``i``, so ``sum_i p_i phi(s_i) = c`` for ANY normalized ``p``
-- the equality constraint is satisfied identically regardless of ``theta``,
so ``theta`` drops out of the optimization entirely (its stationary value is
irrelevant/undetermined, consistently with it not appearing in eq. (11) once
``phi_hat``'s contribution is folded into a state-independent normalizing
constant). With ``C = S`` (the paper's own safety-only use of the general
constraint vector, eq. 1 vs eq. 2) this leaves exactly:

    g(eta, gamma) = eta*epsilon + gamma*delta
                    + eta * log( (1/N) sum_i exp( (R_i - gamma*(1 - S_i)) / eta ) )

which is what ``solve_creps_dual`` minimizes (see that function's docstring
for why "minimize", even though the paper's eq. (5) literally reads "max").

Safety criterion (Sec III) -- FULL, both clauses wired up
------------------------------------------------------------------
The paper's cautious simulator flags a rollout unsafe if the robot "tips over,
hits hard on the ground or obstacle ..., or hits objects with delicate parts".
This trainer now implements the first two clauses as an OR (a rollout is safe
iff NEITHER fires at any point during the fixed-length window); the third has
no MARV analogue (see below).

* **Tip-over**: ``|roll|`` or ``|pitch|`` exceeds ``max_tilt`` (radians). This
  is the ``max_tilt`` config key the previous implementation read but never
  used ("safety-masked" claims in an old docstring were false -- there was no
  masking anywhere); it was the sole source of ``S_i`` before this fix.
* **Hard impact** ("hits hard on the ground or obstacle ..., measured as
  [the paper's own parenthetical] deceleration"): the engine has no collision
  callback / impulse log to hook (``engine.py`` is a soft-contact penalty
  method, not an impulse-based solver), so this reads the CONTACT-INDUCED
  part of the robot's CoG acceleration straight out of
  ``PhysicsStateDer`` -- ``(f_spring + f_friction).sum(over contact points) /
  robot_mass`` (equivalently ``xdd - gravity_vector``, since
  ``engine.py: xdd = gravity_vector + act_force/mass``). This is exactly the
  "specific force" a real onboard accelerometer reads: ~0 in free flight, ~1g
  (9.81 m/s^2) at rest (contact force alone supports the robot's weight),
  spiking well above that under a hard landing or collision -- verified
  empirically against this engine (a ~1.5m drop peaks at ~90g; ordinary
  resting/settling stays under ~5g). "Hard impact" = this magnitude exceeding
  ``max_impact_accel`` (m/s^2, default ~4g) at any point in the window.
  ``impact_warmup_steps`` env-level steps right after reset are EXCLUDED from
  this check: the soft-contact model rings down from the spawn pose's initial
  interpenetration with a brief high-magnitude transient (several g,
  confirmed empirically) that is a simulator settling artifact, not a
  policy-caused impact -- checking from t=0 would misclassify literally every
  episode's opening step as unsafe. Requires ``PhysicsStateDer`` in the
  tensordict, so this trainer forces ``prepare_env(..., force_return_derivative=True)``
  regardless of ``mode="train"``. Caveat: with ``engine_iters_per_env_step >
  1`` (this repo's usual convention), only the FIRST of the per-step physics
  substeps' derivative is exposed (``env.py: Env._step_engine``, unmodified
  here) -- a spike confined to a later substep within the same env-step could
  be under-detected; set ``engine_iters_per_env_step: 1`` for tighter (but
  slower) fidelity.
* **Delicate parts** (paper: "hits objects with delicate parts of its body,
  e.g. sensors"): N/A -- MARV/this engine has no distinguished "delicate"
  collision geometry separate from the rest of the body (see CLAUDE.md's
  sensor list; none of it is modelled as its own collision primitive here),
  so there is nothing to check. Documented gap, not a silent omission.

Run:   python -m flipper_training.experiments.creps.train --local <cfg.yaml>
Config (all optional, defaults documented on each ``_cfg`` call below):
  policy_config: ${cls:flipper_training.policies.pecka_policy.PeckaLinearPolicyConfig}
  policy_opts: {}   # see PeckaLinearPolicyConfig's docstring
  epsilon, delta, max_tilt, max_impact_accel, impact_warmup_steps, n_iters,
  steps_per_iter, init_std, ridge, eval_episodes, eta_init, gamma_init,
  roll_idx, roll_scale.
"""

from __future__ import annotations

import dataclasses
import json
import math
import traceback

import torch
from scipy.optimize import minimize

from flipper_training.environment.env import Env
from flipper_training.experiments.ppo.common import (
    prepare_env,
    make_transformed_env,
    parse_and_load_config,
)
from flipper_training.experiments.ppo.config import PPOExperimentConfig
from flipper_training.utils.logutils import RunLogger, get_terminal_logger

log = get_terminal_logger("creps_train")


def _cfg(c, k, d):
    return c[k] if k in c else d


def solve_creps_dual(
    R: torch.Tensor,
    S: torch.Tensor,
    epsilon: float,
    delta: float,
    eta_init: float = 1.0,
    gamma_init: float = 1.0,
    eta_min: float = 1e-6,
) -> tuple[float, float, torch.Tensor]:
    """Context-free Constrained REPS dual (Pecka et al. 2016, eqs 5-11 with the
    context term ``theta.phi(s)`` dropped -- see module docstring).

    Minimizes::

        g(eta, gamma) = eta*epsilon + gamma*delta
                        + eta * log( (1/N) sum_i exp( (R_i - gamma*(1-S_i)) / eta ) )

    over ``eta >= eta_min``, ``gamma >= 0``.

    The paper's eq. (5) literally states "max_(eta,gamma,theta) g(...)". We
    MINIMIZE here instead, for two independent reasons: (a) standard
    Lagrangian duality -- ``g`` is obtained by substituting the Lagrangian's
    stationary ``p`` back in, i.e. ``g(eta,gamma) = sup_p L(p,eta,gamma)``,
    which for a concave-maximization primal with "<=" constraints upper-bounds
    the primal optimum for every dual-feasible ``(eta,gamma)``; the TIGHTEST
    such bound -- the one actually solved for by differentiating (eqs 8-10,
    which the paper does state correctly) -- is the INFIMUM of ``g`` over the
    dual-feasible region, not its supremum. (b) The paper itself calls ``g``
    "a convex function" immediately below eq. (7): a convex function is
    generically UNBOUNDED ABOVE on the unbounded feasible set
    ``eta>0, gamma>=0``, so "maximize" would not even be well-posed, whereas
    "minimize a convex function with a lower-bound box constraint" is exactly
    the well-posed problem an interior-point/L-BFGS-B solver converges on --
    which is what the paper reports using, and what every public REPS/C-REPS
    reference implementation does. We treat the paper's "max" as carried-over
    notation from the primal (eq. 3), not as a second, independent claim.

    Numerics: the ``(1/N) sum_i exp(...)`` is evaluated via
    ``torch.logsumexp``, and the returned weights via ``torch.softmax`` on the
    same un-normalized exponent -- both subtract the running max internally
    before exponentiating, so no term ever overflows regardless of the scale
    of ``R``/``eta``.

    Returns:
        ``(eta*, gamma*, weights)`` -- ``weights`` are the normalized eq. (11)
        sample probabilities ``p^[i]`` evaluated at the optimum, ready to use
        directly for a weighted-ML refit of the upper-level Gaussian. If the
        finite-sample dual is unbounded (solver saturates its box bound), a
        logged fallback returns hard-safe weights: 0 on unsafe samples,
        KL-bound-only REPS weights on the safe subsample (see inline comment).
    """
    R = R.detach().to(torch.float64).cpu()
    S = S.detach().to(torch.float64).cpu()
    if R.shape != S.shape or R.dim() != 1:
        raise ValueError(f"R and S must be 1-D and same-shaped, got {tuple(R.shape)} and {tuple(S.shape)}")
    n = R.shape[0]
    if n == 0:
        raise ValueError("solve_creps_dual requires at least one sample")
    unsafety = 1.0 - S  # [N], paper's (1 - S_sw) from eq. (1)

    def g_and_grad(x):
        eta_t = torch.tensor(float(x[0]), dtype=torch.float64, requires_grad=True)
        gamma_t = torch.tensor(float(x[1]), dtype=torch.float64, requires_grad=True)
        z = (R - gamma_t * unsafety) / eta_t
        log_mean_exp = torch.logsumexp(z, dim=0) - math.log(n)
        g = eta_t * epsilon + gamma_t * delta + eta_t * log_mean_exp
        grad_eta, grad_gamma = torch.autograd.grad(g, [eta_t, gamma_t])
        return float(g.detach()), torch.stack([grad_eta, grad_gamma]).numpy()

    import numpy as np

    # The FINITE-SAMPLE dual can be genuinely unbounded below along an
    # (eta, gamma) ray whenever the safety budget delta cannot be met at the
    # requested KL bound by reweighting the current samples -- NOT only in
    # degenerate toy settings: it happens with the paper's own epsilon=0.1 on
    # its Fig. 2 example, and with this trainer's defaults whenever the current
    # policy's safe fraction is low (i.e. exactly early training). When that
    # happens L-BFGS-B runs to this box bound and the eq.-11 weights degrade to
    # (near-)uniform over the safe samples -- reward-blind (audit round-3
    # finding; an earlier revision of this comment wrongly claimed the bound
    # "never binds" for well-posed epsilon/delta). The saturation is therefore
    # DETECTED below and handled by an explicit fallback rather than silently
    # accepted.
    eta_max = gamma_max = 1e8
    x0 = np.array([max(eta_init, eta_min), max(gamma_init, 0.0)], dtype=np.float64)
    res = minimize(g_and_grad, x0, jac=True, method="L-BFGS-B", bounds=[(eta_min, eta_max), (0.0, gamma_max)])
    eta_opt, gamma_opt = float(res.x[0]), float(res.x[1])
    # Saturation fallback: if the solver ran to the box bound, the constrained
    # dual has no finite minimizer for this batch (safety not satisfiable at
    # delta by reweighting). The principled limit of the diverging ray is
    # HARD-enforced safety: zero weight on unsafe samples, standard
    # unconstrained REPS (KL bound only) on the SAFE subsample -- which keeps
    # the reward signal instead of degrading to reward-blind uniform weights.
    saturated = eta_opt >= 0.99 * eta_max or gamma_opt >= 0.99 * gamma_max
    if saturated:
        safe = S >= 0.5
        n_safe = int(safe.sum())
        if n_safe == 0:
            log.warning("[creps] dual saturated with NO safe samples in the batch: uniform weights this iteration (pure exploration; consider relaxing max_tilt or delta).")
            return eta_opt, gamma_opt, torch.full((n,), 1.0 / n, dtype=torch.float32)
        log.warning(
            f"[creps] dual saturated (eta={eta_opt:.3g}, gamma={gamma_opt:.3g}): safety budget delta={delta} "
            f"not satisfiable by reweighting this batch (safe {n_safe}/{n}); falling back to hard-safe REPS "
            "on the safe subsample (weight 0 on unsafe samples, KL-bound-only dual)."
        )
        R_safe = R[safe]

        def g_safe(x):
            eta_t = torch.tensor(float(x[0]), dtype=torch.float64, requires_grad=True)
            g1 = eta_t * epsilon + eta_t * (torch.logsumexp(R_safe / eta_t, dim=0) - math.log(n_safe))
            (grad1,) = torch.autograd.grad(g1, [eta_t])
            return float(g1.detach()), np.array([float(grad1)])

        res_safe = minimize(g_safe, np.array([max(eta_init, eta_min)]), jac=True, method="L-BFGS-B", bounds=[(eta_min, eta_max)])
        eta_opt = float(res_safe.x[0])
        with torch.no_grad():
            weights = torch.zeros(n, dtype=torch.float64)
            weights[safe] = torch.softmax(R_safe / eta_opt, dim=0)
        return eta_opt, gamma_opt, weights.to(torch.float32)
    with torch.no_grad():
        z = (R - gamma_opt * unsafety) / eta_opt
        weights = torch.softmax(z, dim=0)  # eq. (11), normalized; softmax subtracts max(z) internally
    return eta_opt, gamma_opt, weights.to(torch.float32)


_PPO_CFG_FIELDS = {f.name for f in dataclasses.fields(PPOExperimentConfig)}


def main():
    config = parse_and_load_config()
    # PPOExperimentConfig is a strict dataclass: filter out the creps-only
    # top-level keys (epsilon/delta/max_tilt/n_iters/...) before constructing
    # it, otherwise any config that used the documented creps keys would fail
    # to load at all (same pattern as experiments/dqn/train.py).
    cfg = PPOExperimentConfig(**{k: v for k, v in config.items() if k in _PPO_CFG_FIELDS})
    # force_return_derivative=True: the Sec III hard-impact criterion needs PhysicsStateDer
    # (f_spring/f_friction) every step, which mode="train" alone does not request -- see
    # prepare_env's docstring and this module's "Hard impact" docstring section.
    env, device, rng = prepare_env(cfg, mode="train", force_return_derivative=True)
    assert env.return_derivative, "prepare_env(force_return_derivative=True) did not take effect -- hard-impact criterion needs PhysicsStateDer"
    robot_total_mass = float(env.robot_cfg.total_mass)  # captured on the BASE env, before make_transformed_env rewraps `env`

    wrapper, _optim_groups, policy_transforms = cfg.policy_config(**cfg.policy_opts).create(env, device=device)
    wrapper.eval()
    module = wrapper.module  # _PeckaLinearCore (pecka_policy.py); duck-typed here, no cross-file import needed
    obs_key, stash_key = wrapper.obs_key, wrapper.raw_obs_stash_key
    n_pairs, n_features = module.W.shape
    omega_dim = n_pairs * n_features
    # IMPORTANT: forward the policy's own transforms (the raw-obs stash) into
    # the training env -- the old implementation passed `[]` here, silently
    # dropping them, which would have fed VecNorm-whitened numbers into phi().
    env, _vecnorm = make_transformed_env(env, cfg, policy_transforms)

    n_robots = cfg.num_robots  # == K, the C-REPS sample count per iteration (one omega per robot row)
    epsilon = float(_cfg(config, "epsilon", 1.0))
    # NOTE on the default value: eq. (1) formally bounds EXPECTED UNSAFETY (`sum
    # p(1-S) <= delta`), but the paper's own prose repeatedly compares delta
    # directly against "mean safety" curves as if it were a safety FLOOR (Sec
    # II-C toy example: "average safety ... 0.6064 ... above the required
    # safety bound" of delta=0.6; Sec IV-D calls delta "the expected safety
    # lower bound", Fig. 4's "desired threshold of 0.8") -- the paper is not
    # fully self-consistent between formula and prose on this point. We follow
    # eq. (1)'s formula exactly (as specified for this fix), so `delta` here is
    # an unsafety CEILING; the default is chosen to be a genuinely-binding
    # constraint under THAT reading (<=20% of episodes unsafe) rather than a
    # literal copy of the paper's "0.8", which would be an unsafety ceiling of
    # 80% -- i.e. almost no constraint at all -- if copied without correcting
    # for the ambiguity. Override via `delta:` in the training config either way.
    delta = float(_cfg(config, "delta", 0.2))
    max_tilt = float(_cfg(config, "max_tilt", 0.6))  # rad; cautious-sim tip-over proxy (Sec III), see module docstring
    # m/s^2; cautious-sim hard-impact proxy (Sec III), see module docstring "Hard impact". ~4g: comfortably
    # above the ~1g resting/quasi-static baseline (contact force alone supporting the robot's weight) and
    # the several-g range seen during ordinary settling, while still well below genuine hard-landing spikes
    # (empirically two orders of magnitude above g for a real drop) -- see module docstring for the numbers.
    max_impact_accel = float(_cfg(config, "max_impact_accel", 40.0))
    # env-level steps right after reset excluded from the hard-impact check (soft-contact spawn settling
    # transient, not a policy-caused impact -- see module docstring "Hard impact").
    impact_warmup_steps = int(_cfg(config, "impact_warmup_steps", 2))
    n_iters = int(_cfg(config, "n_iters", 30))  # paper: "10 to 30 iterations ... until the policy converges"
    steps_per_iter = int(_cfg(config, "steps_per_iter", 50))  # fixed-length episode (paper: fixed 30s window)
    if impact_warmup_steps >= steps_per_iter:
        log.warning(
            f"[creps] impact_warmup_steps={impact_warmup_steps} >= steps_per_iter={steps_per_iter}: every step of "
            "every rollout falls inside the warmup window, so the hard-impact criterion can NEVER fire (S degrades "
            "to tip-over-only). Lower impact_warmup_steps or raise steps_per_iter if that's not intended."
        )
    init_std = float(_cfg(config, "init_std", 0.3))  # paper's toy example (Sec II-C) std
    ridge = float(_cfg(config, "ridge", 1e-4))  # covariance regularizer
    eval_episodes = int(_cfg(config, "eval_episodes", 3))  # FIX 3 safety-gate rounds ("a few episodes")
    eta_init = float(_cfg(config, "eta_init", 1.0))
    gamma_init = float(_cfg(config, "gamma_init", 1.0))
    roll_idx = int(_cfg(config, "roll_idx", 0))  # LocalStateVector layout: roll@0
    roll_scale = float(_cfg(config, "roll_scale", math.pi))
    pitch_idx, pitch_scale = module.pitch_idx, module.pitch_scale  # reuse the policy's own pitch reading -- guaranteed consistent with phi(s)

    mu = module.W.detach().clone().flatten().to(torch.float64)  # seeded from PeckaLinearPolicyConfig.seed_omega if set, else zeros
    Sigma = (init_std**2) * torch.eye(omega_dim, dtype=torch.float64)

    logger = RunLogger(train_config=config, use_wandb=cfg.use_wandb, use_tensorboard=cfg.use_tensorboard, category="creps")

    def raw_obs(td) -> torch.Tensor:
        raw = td.get(stash_key, None)
        return raw if raw is not None else td.get(obs_key)

    def tilt(obs: torch.Tensor) -> torch.Tensor:
        roll = obs[..., roll_idx] * roll_scale
        pitch = obs[..., pitch_idx] * pitch_scale
        return torch.maximum(roll.abs(), pitch.abs())

    def impact_accel(der_td) -> torch.Tensor:
        """Contact-only (gravity-excluded) CoG acceleration magnitude, m/s^2 --
        Sec III "hits hard ... measured as deceleration"; see module docstring
        "Hard impact" for the derivation and the empirical baseline/threshold
        reasoning. ``der_td`` is the ``Env.PREV_STATE_DER_KEY`` sub-tensordict.
        """
        contact_force = der_td.get("f_spring").sum(dim=1) + der_td.get("f_friction").sum(dim=1)  # (B, 3) N
        return contact_force.norm(dim=-1) / robot_total_mass  # (B,) m/s^2

    def rollout_with_omega(W_batch: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One fixed-length episode, robot row i driven throughout by ``W_batch[i]``.

        Returns ``(R, S)``: per-row episodic return (summed over the WHOLE
        window -- the credit-assignment fix, weights are per-EPISODE not
        per-step) and binary safety (Sec III: safe iff the robot NEITHER tips
        over -- ``|roll|``/``|pitch|`` > ``max_tilt`` -- NOR hits hard --
        contact-induced CoG acceleration > ``max_impact_accel``, skipping the
        first ``impact_warmup_steps`` steps' impact reading, see module
        docstring) at any point during the window, including the reset state
        for tilt (spawn orientation is meaningful) but NOT for impact (spawn
        contact-settling transient is a simulator artifact, not a policy-caused
        impact). No mid-rollout reset-on-done: all K rows share one continuous
        fixed-length window, matching the paper's fixed 30s episode and
        keeping the "one omega -> one clean episode -> one (R,S)" mapping exact.
        """
        td = env.reset()
        obs = raw_obs(td)
        ret = torch.zeros(n_robots, device=device)
        max_tilt_seen = tilt(obs)
        max_impact_seen = torch.zeros(n_robots, device=device)  # reset-time reading excluded -- see docstring above
        for step_i in range(steps):
            with torch.no_grad():
                action = module.compute_action(obs, W_batch)
            step_td = td.clone()
            step_td.set("action", action)
            next_td = env.step(step_td)
            ret = ret + next_td["next", "reward"].reshape(n_robots)
            td = next_td["next"]
            obs = raw_obs(td)
            max_tilt_seen = torch.maximum(max_tilt_seen, tilt(obs))
            if step_i >= impact_warmup_steps:
                max_impact_seen = torch.maximum(max_impact_seen, impact_accel(td.get(Env.PREV_STATE_DER_KEY)))
        tipped_over = max_tilt_seen > max_tilt
        hit_hard = max_impact_seen > max_impact_accel
        S = (~(tipped_over | hit_hard)).to(torch.float32)
        return ret, S

    try:
        for it in range(n_iters):
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma)
            omega_k = dist.sample((n_robots,))  # [K, omega_dim], K == n_robots (Algorithm 1's N samples)
            W_batch = omega_k.to(torch.float32).view(n_robots, n_pairs, n_features).to(device)

            R, S = rollout_with_omega(W_batch, steps_per_iter)
            eta, gamma, weights = solve_creps_dual(R, S, epsilon, delta, eta_init, gamma_init)
            eta_init, gamma_init = eta, gamma  # warm-start next iteration's dual solve

            w64 = weights.to(torch.float64)
            mu_new = (w64.unsqueeze(-1) * omega_k).sum(dim=0)
            centered = omega_k - mu_new
            denom = (1.0 - (w64**2).sum()).clamp_min(1e-6)  # weighted-ML "effective sample size" correction
            Sigma_new = torch.einsum("k,ki,kj->ij", w64, centered, centered) / denom
            Sigma_new = Sigma_new + ridge * torch.eye(omega_dim, dtype=torch.float64)
            mu, Sigma = mu_new, Sigma_new

            expected_unsafety = float((weights * (1.0 - S).cpu()).sum())
            mean_return = float(R.mean())
            log.info(
                f"[creps] iter {it + 1}/{n_iters} R_mean={mean_return:.3f} eta={eta:.4f} gamma={gamma:.4f} "
                f"mean_safety={float(S.mean()):.3f} expected_unsafety={expected_unsafety:.3f} (delta={delta})"
            )
            logger.log_data(
                {
                    "creps/mean_return": mean_return,
                    "creps/eta": eta,
                    "creps/gamma": gamma,
                    "creps/mean_safety": float(S.mean()),
                    "creps/expected_unsafety": expected_unsafety,
                },
                it,
            )

        # Deploy the converged mean policy, then FIX 3: certify it with a dedicated safety-gate eval
        # (Sec III: re-check the optimal policy's safety; here via `eval_episodes` extra rollouts).
        with torch.no_grad():
            module.W.copy_(mu.to(torch.float32).view(n_pairs, n_features))
        mean_W_batch = module.W.detach().unsqueeze(0).expand(n_robots, -1, -1).contiguous()
        unsafety_samples = []
        for _ in range(eval_episodes):
            _, S_eval = rollout_with_omega(mean_W_batch, steps_per_iter)
            unsafety_samples.append((1.0 - S_eval).cpu())
        expected_unsafety_final = float(torch.cat(unsafety_samples).mean()) if unsafety_samples else float("nan")
        certified_safe = expected_unsafety_final <= delta
        log.info(
            f"[creps] safety gate (Sec III): expected_unsafety={expected_unsafety_final:.3f} "
            f"{'<=' if certified_safe else '>'} delta={delta} -> certified_safe={certified_safe}"
        )
        logger.log_data({"creps/final_expected_unsafety": expected_unsafety_final, "creps/certified_safe": float(certified_safe)}, n_iters)

        logger.save_weights(wrapper.state_dict(), name="policy_final")
        info = {
            "algorithm": "constrained_reps_context_free",
            "n_iters": n_iters,
            "steps_per_iter": steps_per_iter,
            "epsilon": epsilon,
            "delta": delta,
            "max_tilt": max_tilt,
            "max_impact_accel": max_impact_accel,
            "impact_warmup_steps": impact_warmup_steps,
            "final_expected_unsafety": expected_unsafety_final,
            "certified_safe": certified_safe,
            "omega_mean": mu.tolist(),
            "omega_cov": Sigma.tolist(),
        }
        with open(logger.logpath / "safety_gate.json", "w") as f:
            json.dump(info, f, indent=2)
        log.info(f"C-REPS done; saved policy_final.pth (deploy via run_flipper_policy_sim.sh). certified_safe={certified_safe} (see safety_gate.json).")
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        traceback.print_exception(e)
        raise
    finally:
        # Always flush/close the CSV/W&B/TensorBoard writers, even on Ctrl-C or a
        # mid-rollout exception (the old implementation never closed the logger).
        logger.close()


if __name__ == "__main__":
    main()
