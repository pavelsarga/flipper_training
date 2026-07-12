"""C-TRAC trainer: two-stage asymmetric-SAC + contact-VAE training (Pan et al. 2025 IROS, Fig. 2).

Adds the off-policy SAC-with-replay-buffer loop MARV_RL's native experiments lacked for
this baseline (``policies/ctrac_policy.py``'s ``CTRACConfig``). Two stages, matching the
paper's Fig. 2:

  Stage 1 — "C-VAE pretraining" (``pretrain_iters`` collector iterations): roll out the
    policy AS CONSTRUCTED (random Gaussian head, random C-VAE) with the stochastic actor,
    and run ``cvae_updates_per_pretrain_iter`` C-VAE-hybrid-loss-only gradient steps per
    iteration — only the ``contact_vae`` optimiser group (shared encoder + C-VAE
    trunk/heads) is stepped; the actor head, critics, and SAC entropy coefficient are
    untouched. Divergence from the paper: the paper pretrains on "successful locomotion
    trajectories performed by terrain-specific ... flipper control policies" (expert data
    this repo has no equivalent of); this uses rollouts of the random-init policy instead,
    per the audit fix list (FIX 5) — documented here and in ``ctrac_NOTE.md``.

  Stage 2 — "C-VAE + SAC joint training" (``n_iters`` collector iterations): each
    iteration is 1 SAC update (actor / qvalue / alpha via ``torchrl.objectives.SACLoss``;
    its actor-loss gradient contribution to the shared encoder + C-VAE is also applied,
    through the ``contact_vae`` optimiser group) followed by ``vae_updates_per_sac_update``
    (default 5) further C-VAE-only updates — the paper's stated 1:5 alternation ratio.

Both stages share ONE ``SyncDataCollector`` (continuously reading the live, in-place
mutated actor operator — no explicit policy-weight sync needed, same as
``experiments/ppo/train.py``) and ONE off-policy ``TensorDictReplayBuffer`` (uniform
random sampling with replacement, standard for SAC).

C-VAE updates (``_vae_update``, both stages) are multi-frame- and clean-target-aware: they
re-derive the history-stacked encoding ``y_enc_hist`` for the sampled (shuffled) batch via
``wrapper.encoder_operator`` + ``wrapper.history_operator`` (``_y_enc_hist``, see its docstring
for exactly how a rollout-carried ring buffer interacts with off-policy replay), and use
``batch["next", "clean"]`` as the reconstruction target when the env supplies one
(``emit_clean_observations``), else plain ``batch["next"]``. See ``ctrac_policy.py``'s module
docstring ("Multi-frame history" / "Clean denoising target" sections) for the full design; both
default OFF (``history_len=1``, ``emit_clean_observations=False``), so this trainer's behaviour
is unchanged unless a config turns them on (``configs/baselines/ctrac_full.yaml`` does).

Run:   python -m flipper_training.experiments.ctrac.train --local <ctrac_config.yaml>

Config plumbing: reuses ``flipper_training.experiments.ppo.config.PPOExperimentConfig``
for the env/robot/terrain/observation/logging fields — the same pragmatic choice already
made by ``experiments/dqn`` and ``experiments/creps`` (a bespoke dataclass would just
duplicate the same required fields with zero benefit). This means the YAML must still
supply the PPO-only fields this trainer never reads (``gae_opts``, ``ppo_opts``,
``epochs_per_batch``, ``frames_per_sub_batch``, ``scheduler``/``scheduler_opts``) with
placeholder values — see ``configs/baselines/ctrac_NOTE.md`` at the MARV_RL repo root
(one level above this flipper_training tree, i.e. MARV_RL/configs/baselines/) for a concrete checklist.
Fields this trainer DOES read: ``time_steps_per_batch`` (collector frames-per-iteration,
divided by ``num_robots``), ``num_robots``, ``max_grad_norm``/``clip_grad_norm_p``,
``optimizer``/``optimizer_opts``, ``data_collector_opts`` (set
``exploration_type: RANDOM`` — SAC needs stochastic collection), ``eval_and_save_every``,
``max_eval_steps``, ``eval_repeats_after_training``, plus everything ``prepare_env`` /
``make_transformed_env`` need (robot/terrain/observations/reward/objective/vecnorm/...).
``policy_config`` must be ``${cls:flipper_training.policies.ctrac_policy.CTRACConfig}``
with matching ``policy_opts``, and ``observations:`` MUST include a
``flipper_training.observations.contacts.GroundTruthContacts`` entry — this trainer
refuses to start without it (it is both the asymmetric critic's privileged input and the
sole source of C-VAE training targets; there is no other ground truth in this repo).

Additional (optional) top-level config keys, read directly off the raw OmegaConf dict via
a plain ``dict.get``-style helper (not part of ``PPOExperimentConfig``), with the
following defaults. Where the paper (Sec. V-A.1) states a hyperparameter, the default
matches it:
    gamma: 0.99                        # paper: discount factor 0.99
    target_tau: 0.01                   # paper: target network update rate 0.01
    alpha_init: 1.5                    # paper: entropy coefficient 1.5
    alpha_lr: 3.0e-4                   # paper: Adam lr 3e-4 (applied to log_alpha too)
    fixed_alpha: false
    target_entropy: auto
    sac_batch_size: 256                 # minibatch size sampled from the replay buffer
    replay_buffer_size: null            # -> 200 * time_steps_per_batch * num_robots
    pretrain_iters: 20                  # stage-1 length, in collector iterations
    n_iters: 200                        # stage-2 length, in collector iterations
    cvae_updates_per_pretrain_iter: 10
    vae_updates_per_sac_update: 5       # paper Fig. 2: "5 iterations" per 1 SAC update

Q-network sourcing (a torchrl-0.8.1-specific wrinkle — see ``ctrac_policy.py``'s module
docstring, "Asymmetric privileged twin-Q critic"): ``CTRACConfig`` hands ``SACLoss`` a
SINGLE Q-network template + ``num_qvalue_nets``, since a plain list of separately-built Q
modules crashes ``SACLoss._set_in_keys`` in the installed torchrl version. torchrl expands
the template internally into ``num_qvalue_nets`` independent, freshly-reinitialised
parameter sets that are DISCONNECTED from the template module's own parameter identity
(verified empirically: the template's own ``.grad`` stays ``None`` and its weights never
change). The trainable copies live at ``sac_loss.qvalue_network_params`` — this trainer's
``qvalue_optim`` is built from THERE, not from ``wrapper``'s own "qvalue" optimiser group.
One practical consequence: ``wrapper.state_dict()``'s Q-net weights in the saved checkpoint
are the pre-training template init, NOT the SAC-learned critic — harmless for deployment
(the actor never reads them) but means the checkpoint can't warm-start a critic; only the
actor + C-VAE (whose torchrl conversion has no such disconnection — verified in the same
way) are faithfully saved/restored.

Saves ``policy_final.pth`` (``wrapper.state_dict()``: actor + Q-template-init + C-VAE) AND
``vecnorm_final.pth`` (plus ``policy_step_<i>``/``vecnorm_step_<i>`` at every
``eval_and_save_every``) via ``RunLogger`` under ``runs/ctrac/<run_name>/weights/`` — the
same two-file pattern ``experiments/ppo/train.py`` uses (round-4 fix: this trainer used to
save only the policy, never the VecNorm running statistics, so a checkpoint alone could not
actually restore the observation normalisation ``run_flipper_policy_sim.sh`` /
``PPOPolicyInferenceModule`` expect — the doc claim "deploy exactly like every other
baseline" was not literally true until this was added). Deploy through
``run_flipper_policy_sim.sh`` / ``flipper_policy_node`` exactly like every other baseline
here — they only ever call ``get_policy_operator()(td)["action"]``, so the critics/C-VAE
training-only state in the checkpoint is simply ignored at deploy time.
"""

from __future__ import annotations

import dataclasses
import traceback
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, RandomSampler, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SACLoss, SoftUpdate

from flipper_training.environment.env import Env
from flipper_training.experiments.ppo.common import (
    EVAL_LOG_OPT,
    log_from_eval_rollout,
    make_formatted_str_lines,
    make_transformed_env,
    parse_and_load_config,
    prepare_env,
)
from flipper_training.experiments.ppo.config import PPOExperimentConfig
from flipper_training.policies.ctrac_policy import DEFAULT_CONTACT_OBS_NAME, HISTORY_KEY, flatten_observations
from flipper_training.utils.logutils import RunLogger, get_terminal_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict

__all__ = ["CTRACTrainer"]

# PPOExperimentConfig is a strict dataclass (no **kwargs catch-all): filter out the
# ctrac-only top-level keys (gamma/target_tau/alpha_init/.../n_iters/... — the "Additional
# (optional) top-level config keys" list in this module's docstring) before constructing it,
# otherwise ANY config that actually used those documented ctrac keys would fail to even
# load (`PPOExperimentConfig(**config)` raises "unexpected keyword argument" on the first
# one) — same pattern as `experiments/dqn/train.py` and `experiments/creps/train.py`.
_PPO_CFG_FIELDS = {f.name for f in dataclasses.fields(PPOExperimentConfig)}


def _cfg(c, k, d):
    """Read an optional key straight off the raw OmegaConf dict (not a PPOExperimentConfig field)."""
    return c[k] if k in c else d


class CTRACTrainer:
    """Two-stage asymmetric-SAC + C-VAE trainer for ``CTRACConfig`` (see module docstring)."""

    def __init__(self, config: "DictConfig"):
        self.raw_config = config
        self.config = PPOExperimentConfig(**{k: v for k, v in config.items() if k in _PPO_CFG_FIELDS})
        self.term_logger = get_terminal_logger("ctrac_train")
        self.run_logger = RunLogger(
            train_config=config,
            category="ctrac",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="iteration",
        )

        # ---- hyperparameters not covered by PPOExperimentConfig (paper Sec. V-A.1 where stated)
        self.gamma = float(_cfg(config, "gamma", 0.99))
        self.target_tau = float(_cfg(config, "target_tau", 0.01))
        self.alpha_init = float(_cfg(config, "alpha_init", 1.5))
        self.alpha_lr = float(_cfg(config, "alpha_lr", 3.0e-4))
        self.fixed_alpha = bool(_cfg(config, "fixed_alpha", False))
        self.target_entropy = _cfg(config, "target_entropy", "auto")
        self.sac_batch_size = int(_cfg(config, "sac_batch_size", 256))
        self.pretrain_iters = int(_cfg(config, "pretrain_iters", 20))
        self.n_iters = int(_cfg(config, "n_iters", 200))
        self.cvae_updates_per_pretrain_iter = int(_cfg(config, "cvae_updates_per_pretrain_iter", 10))
        self.vae_updates_per_sac_update = int(_cfg(config, "vae_updates_per_sac_update", 5))
        self.total_iters = self.pretrain_iters + self.n_iters

        # ---- env + policy
        self.env, self.device, self.rng = prepare_env(self.config, mode="train")
        self.policy_config = self.config.policy_config(**self.config.policy_opts)
        self.wrapper, self.optim_groups, self.policy_transforms = self.policy_config.create(
            env=self.env,
            weights_path=self.config.policy_weights_path,
            device=self.device,
        )
        if getattr(self.wrapper, "contact_target_key", None) is None:
            raise ValueError(
                "C-TRAC requires a ground-truth contact observation in the env to supply the "
                "asymmetric critic's privileged input AND the C-VAE training targets — none was "
                f"found (looked for '{DEFAULT_CONTACT_OBS_NAME}', see policy_opts.contact_target_key "
                "and policy_opts.privileged_observations). Add "
                "`flipper_training.observations.contacts.GroundTruthContacts` to `observations:` "
                "in the config."
            )
        self.actor_operator = self.wrapper.get_policy_operator()
        self.qvalue_op = self.wrapper.get_qvalue_operator()  # SINGLE template, see module docstring
        self.env, self.vecnorm = make_transformed_env(self.env, self.config, self.policy_transforms)
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False)

        # ---- SAC loss (Q-only "second version" — no separate V-network, matches
        # offpolicy_ac_policy.py's guidance) + target-network updater. qvalue_network takes
        # the SINGLE template + num_qvalue_nets (torchrl expands it internally) — passing
        # several separately-built modules in a list crashes SACLoss._set_in_keys on this
        # torchrl version (see ctrac_policy.py's and this file's module docstrings).
        self.sac_loss = SACLoss(
            actor_network=self.actor_operator,
            qvalue_network=self.qvalue_op,
            num_qvalue_nets=self.wrapper.num_qvalue_nets,
            action_spec=self.actor_operator.spec,
            alpha_init=self.alpha_init,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
        )
        self.sac_loss = self.sac_loss.to(self.config.training_dtype)
        self.sac_loss.make_value_estimator(gamma=self.gamma)
        self.target_updater = SoftUpdate(self.sac_loss, tau=self.target_tau)

        # ---- disjoint optimizers per group, so stage 1 can step ONLY contact_vae, and the
        # SAC step's gradient contribution to the shared encoder/C-VAE (the actor loss
        # backprops through it too) is applied at the "contact_vae" group's own settings —
        # see ctrac_policy.py's module docstring ("Optimizer groups are disjoint").
        # NOTE: qvalue_optim is deliberately NOT built from wrapper's own "qvalue" optim
        # group — SACLoss's num_qvalue_nets expansion (above) trains independent parameter
        # copies disconnected from the template's identity; the copies that actually
        # receive gradient live at sac_loss.qvalue_network_params (verified empirically;
        # see module docstring). Optimizing the template's own params would be a silent
        # no-op (their .grad would always be None).
        by_name = {g["name"]: dict(g) for g in self.optim_groups}
        opt_defaults = self.config.optimizer_opts or {}
        self.actor_optim = self.config.optimizer([by_name["actor"]], **opt_defaults)
        # Reuse qvalue_optimizer_opts (e.g. lr) from policy_opts, but point them at the
        # ACTUAL trainable qvalue params (sac_loss.qvalue_network_params), not the
        # (disconnected, never-updated) Q-template's own params from by_name["qvalue"].
        qvalue_group = {k: v for k, v in by_name["qvalue"].items() if k != "params"}
        qvalue_group["params"] = list(self.sac_loss.qvalue_network_params.parameters())
        self.qvalue_optim = self.config.optimizer([qvalue_group], **opt_defaults)
        self.vae_optim = self.config.optimizer([by_name["contact_vae"]], **opt_defaults)
        self.alpha_optim = None
        if self.sac_loss.log_alpha.requires_grad:
            self.alpha_optim = torch.optim.Adam([self.sac_loss.log_alpha], lr=self.alpha_lr)

        # ---- off-policy replay buffer (uniform random sampling w/ replacement, as for SAC)
        frames_per_batch = self.config.time_steps_per_batch * self.config.num_robots
        replay_buffer_size = int(_cfg(config, "replay_buffer_size", 0) or 200 * frames_per_batch)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=replay_buffer_size, ndim=1, device=self.device),
            sampler=RandomSampler(),
            batch_size=self.sac_batch_size,
        )

        # ---- collector: ONE collector spans both stages; it invokes the SAME actor_operator
        # object every iteration, so stage-2 parameter updates (in-place) are immediately
        # visible to the next collection with no explicit weight sync (matches ppo/train.py).
        self.collector = SyncDataCollector(
            self.env,
            self.actor_operator,
            frames_per_batch=frames_per_batch,
            total_frames=frames_per_batch * self.total_iters,
            **self.config.data_collector_opts,
            device=self.device,
        )

        self.term_logger.info(
            f"Initialized CTRACTrainer: {self.pretrain_iters} pretrain iters "
            f"({self.cvae_updates_per_pretrain_iter} C-VAE updates/iter) + {self.n_iters} joint "
            f"iters (1 SAC : {self.vae_updates_per_sac_update} C-VAE), frames/iter={frames_per_batch}, "
            f"replay_buffer_size={replay_buffer_size}, sac_batch_size={self.sac_batch_size}."
        )
        print(OmegaConf.to_yaml(config, sort_keys=True))

    # ------------------------------------------------------------------ top level
    def train(self):
        try:
            self._train()
            post_training_eval_log = self._post_training_evaluation()
        except KeyboardInterrupt:
            self.term_logger.info("Training interrupted by user.")
            post_training_eval_log = None
        except Exception as e:
            self.term_logger.error(f"Training failed with error: {e}")
            traceback.print_exception(e)
            raise e
        finally:
            if self.run_logger is not None:
                self.run_logger.close()
        return post_training_eval_log

    def _train(self):
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        pbar = tqdm(total=self.total_iters, desc="C-TRAC", unit="iter", leave=False)
        for i, tensordict_data in enumerate(self.collector):
            if i >= self.total_iters:
                break
            tensordict_data.pop(Env.STATE_KEY, None)
            tensordict_data.pop(("next", Env.STATE_KEY), None)
            self.wrapper.train()
            self.env.train()
            self.replay_buffer.extend(tensordict_data.reshape(-1))

            stage = "stage1_pretrain" if i < self.pretrain_iters else "stage2_joint"
            log = self._stage1_step() if i < self.pretrain_iters else self._stage2_step()
            log["ctrac/stage"] = 0.0 if stage == "stage1_pretrain" else 1.0
            pbar.update(1)

            if i % self.config.eval_and_save_every == 0:
                log.update(self._get_eval_rollout_results())
                self.run_logger.save_weights(self.wrapper.state_dict(), f"policy_step_{i}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{i}")
            self.run_logger.log_data(log, i)

        self.run_logger.save_weights(self.wrapper.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")

    # ------------------------------------------------------------------ stage steps
    def _stage1_step(self) -> dict[str, float]:
        """Stage 1 (C-VAE pretraining): contact_vae-only updates, actor/critic untouched."""
        log: dict[str, float] = {}
        for _ in range(self.cvae_updates_per_pretrain_iter):
            if len(self.replay_buffer) < 1:
                break
            batch = self.replay_buffer.sample().to(self.device)
            log = self._vae_update(batch)
        return log

    def _stage2_step(self) -> dict[str, float]:
        """Stage 2 (joint training): 1 SAC update, then vae_updates_per_sac_update C-VAE updates."""
        log: dict[str, float] = {}
        if len(self.replay_buffer) >= 1:
            batch = self.replay_buffer.sample().to(self.device)
            log.update(self._sac_update(batch))
            for _ in range(self.vae_updates_per_sac_update):
                batch = self.replay_buffer.sample().to(self.device)
                log.update(self._vae_update(batch))
        return log

    # ------------------------------------------------------------------ gradient updates
    def _clip(self, optim: torch.optim.Optimizer) -> None:
        params = [p for g in optim.param_groups for p in g["params"]]
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm, norm_type=self.config.clip_grad_norm_p)

    def _y_enc_hist(self, td: "TensorDict") -> torch.Tensor:
        """Run the shared encoder + history ring buffer on a REPLAY-SAMPLED (uniformly shuffled,
        non-sequential) batch, restricted to the actor-visible keys (never the privileged ones).

        Reuses the per-transition ``HISTORY_KEY``/``is_init`` that
        ``ctrac_policy._FrameHistoryBuffer`` itself wrote into the tensordict at COLLECTION time
        (carried the same way GRUModule/state_machine's recurrent state is) rather than
        re-deriving multi-step temporal order from the batch, which ``RandomSampler``'s shuffling
        destroys — see ``ctrac_policy.py``'s "Multi-frame history" section. One inherent
        consequence of combining a rollout-carried ring buffer with OFF-POLICY replay (not
        specific to this implementation -- the same applies to any recurrent/history state under
        experience replay): the newest slot of the resulting ``y_enc_hist`` reflects the CURRENT
        (possibly since-updated) encoder weights, while the older ``history_len - 1`` slots are
        whatever the encoder produced AT COLLECTION TIME (frozen values carried in the replay
        buffer, no gradient path back through them). At ``history_len=1`` this reduces exactly to
        the previous single-frame ``_y_enc``.
        """
        keys = [*self.wrapper.actor_obs_keys, HISTORY_KEY, "is_init"]
        sub = td.select(*[k for k in keys if k in td.keys()])
        sub = self.wrapper.encoder_operator(sub)
        sub = self.wrapper.history_operator(sub)
        return sub["y_enc_hist"]

    def _vae_update(self, batch: "TensorDict") -> dict[str, float]:
        y_enc_hist = self._y_enc_hist(batch)
        contact_target = batch[self.wrapper.contact_target_key]
        # Clean (noiseless) reconstruction target when the env supplies one (Env.emit_clean_
        # observations, see ctrac_policy.py's "Clean denoising target" section); falls back to
        # plain next-obs (today's behaviour) when absent, e.g. emit_clean_observations=False.
        next_td = batch["next"]
        target_td = next_td["clean"] if "clean" in next_td.keys() else next_td
        next_obs_flat = flatten_observations(target_td, self.wrapper.actor_obs_keys)
        losses = self.wrapper.contact_vae.loss(y_enc_hist, contact_target, next_obs_flat)
        self.vae_optim.zero_grad(set_to_none=True)
        losses["loss_vae_total"].backward()
        self._clip(self.vae_optim)
        self.vae_optim.step()
        return {f"ctrac/{k}": v.item() for k, v in losses.items()}

    def _sac_update(self, batch: "TensorDict") -> dict[str, float]:
        self.actor_optim.zero_grad(set_to_none=True)
        self.qvalue_optim.zero_grad(set_to_none=True)
        self.vae_optim.zero_grad(set_to_none=True)
        if self.alpha_optim is not None:
            self.alpha_optim.zero_grad(set_to_none=True)

        loss_td = self.sac_loss(batch)
        sac_total = loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
        sac_total.backward()

        # Each group only receives gradient if it was actually on the path to sac_total:
        # actor_optim <- loss_actor (head), vae_optim <- loss_actor (shared encoder + C-VAE,
        # the audited fix), qvalue_optim <- loss_qvalue.
        self._clip(self.actor_optim)
        self._clip(self.qvalue_optim)
        self._clip(self.vae_optim)
        self.actor_optim.step()
        self.qvalue_optim.step()
        self.vae_optim.step()
        if self.alpha_optim is not None:
            self.alpha_optim.step()
        self.target_updater.step()

        return {
            "ctrac/loss_actor": loss_td["loss_actor"].item(),
            "ctrac/loss_qvalue": loss_td["loss_qvalue"].item(),
            "ctrac/loss_alpha": loss_td["loss_alpha"].item(),
            "ctrac/alpha": loss_td["alpha"].item(),
            "ctrac/entropy": loss_td["entropy"].item(),
        }

    # ------------------------------------------------------------------ evaluation
    def _get_eval_rollout_results(self) -> dict[str, float]:
        self.env.eval()
        self.actor_operator.eval()
        with (
            set_exploration_type(ExplorationType.DETERMINISTIC),
            torch.inference_mode(),
        ):
            eval_rollout = self.env.rollout(self.config.max_eval_steps, self.actor_operator, break_when_all_done=True, auto_reset=True)
        results = log_from_eval_rollout(eval_rollout)
        del eval_rollout
        return results

    def _post_training_evaluation(self) -> dict[str, float]:
        self.term_logger.info(f"Training finished, evaluating the final policy for {self.config.eval_repeats_after_training} samples.")
        avg_eval_log = self._get_eval_rollout_results()
        for _ in range(self.config.eval_repeats_after_training - 1):
            for k, v in self._get_eval_rollout_results().items():
                avg_eval_log[k] += v
        for k in avg_eval_log.keys():
            avg_eval_log[k] /= self.config.eval_repeats_after_training
        print(f"\nFinal evaluation results ({self.config.eval_repeats_after_training} samples):")
        print("\n".join(make_formatted_str_lines(avg_eval_log, EVAL_LOG_OPT)))
        return avg_eval_log


if __name__ == "__main__":
    cfg = parse_and_load_config()
    trainer = CTRACTrainer(cfg)
    trainer.train()
