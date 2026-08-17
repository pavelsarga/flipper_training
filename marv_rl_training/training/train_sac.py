# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher
import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C-TRAC (Pan et al. 2025) asymmetric SAC + C-VAE policy inside FTR-Benchmark (Isaac Sim)")
    parser.add_argument("--config", type=str, required=True, help="Path to a SAC config yaml (see FtrSACConfig)")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
    parser.add_argument("--terrain", type=str, default=None, help="Override terrain in config")
    parser.add_argument("--task", type=str, default=None, help="Override task in config (e.g. Ftr-Crossing-Direct-v0)")
    AppLauncher.add_app_launcher_args(parser)
    args, unknown_args = parser.parse_known_args()

    # AppLauncher processes some flags (e.g. --gpu) from sys.argv directly without
    # removing them from unknown_args, so they leak into OmegaConf overrides and crash.
    # Strip any --flag / value pairs that are not OmegaConf key=value overrides.
    _filtered, _skip = [], False
    for _a in unknown_args:
        if _skip:
            _skip = False
            continue
        if _a.startswith("--") and "=" not in _a:
            _skip = True
            continue
        _filtered.append(_a)
    unknown_args = _filtered

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

import torch
from omegaconf import OmegaConf
from torchrl.data import LazyMemmapStorage, RandomSampler, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import SACLoss, SoftUpdate, ValueEstimators
from tqdm import tqdm

import gymnasium

import marv_rl_training  # registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.training.env_type_registry import default_num_env_types
from marv_rl_training.training.eval_data import (
    aggregate_per_env,
    aggregate_per_spot,
    load_env_type_names,
    run_tracked_rollout,
    save_per_spot_csv,
)
from marv_rl_training.utils.logutils import RunLogger, get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.ctrac.ctrac_policy import CTRACPolicyConfig
from rl_modules.ctrac.ctrac_observation import (
    CONTACT_POINTS_OFFSET,
    CONTACT_PROB_OFFSET,
    PARTIAL_DIM,
)
from rl_modules.ctrac.ctrac_cvae import contact_est_loss, contact_geo_loss, contact_prob_loss, vae_loss


# ============================================================
# BLOCK 3 — Config dataclass for FTR-specific SAC training
# ============================================================

@dataclass
class FtrSACConfig:
    """Asymmetric SAC + C-VAE config for C-TRAC (Pan et al. 2025). Mirrors FtrD3QNConfig's
    env/run/physics fields but replaces epsilon-greedy/hard-target-sync with SAC's entropy
    coefficient + soft target update (tau), and adds the C-VAE's own optimizer/schedule
    plus the alternating-update ratio (Fig. 2's "1 SAC iteration : 5 C-VAE refinements").
    """

    name: str
    comment: str
    seed: int
    device: str
    training_dtype: torch.dtype
    num_robots: int
    task: str
    terrain: str
    total_frames: int
    time_steps_per_batch: int

    # Off-policy replay + optimisation
    replay_buffer_capacity: int
    min_replay_size: int
    batch_size: int
    updates_per_batch: int
    gamma: float
    tau: float                  # soft target-network update rate (paper Table: 0.01)

    # SAC entropy (paper Table: entropy coefficient 1.5 — a fixed alpha, not auto-tuned,
    # matching how the paper reports it as a single hyperparameter rather than a target
    # entropy). Settable to auto-tune instead via fixed_alpha: false.
    alpha_init: float = 1.5
    fixed_alpha: bool = True
    target_entropy: "str | float" = "auto"
    alpha_optimizer_opts: dict[str, Any] = field(default_factory=lambda: {"lr": 3e-4})
    loss_function: str = "smooth_l1"

    # C-VAE (Eq. 9-14) — alternating update: for every SAC gradient step, run this many
    # additional supervised C-VAE gradient steps (Fig. 2 caption: "1 iteration" SAC : "5
    # iterations" C-VAE contact-model refinement).
    cvae_updates_per_sac_step: int = 5
    cvae_vae_beta: float = 1.0       # beta-VAE KL weight (paper Table: "KL divergence weight (1.0)")
    cvae_prob_weight: float = 1.0    # L_prob's weight in L_C = L_prob + L_est + L_geo
    cvae_est_weight: float = 1.0
    cvae_geo_weight: float = 1.0
    cvae_max_reach: float = 0.8      # L_geo's Omega radius (m) around the contact centroid

    eval_and_save_every: int = 8
    eval_repeats: int = 2
    eval_repeats_after_training: int = 20
    max_grad_norm: float = 0.5       # paper Table: "gradient clipping threshold (0.5)"
    clip_grad_norm_p: "int | str" = 2
    optimizer: type = None
    optimizer_opts: dict[str, Any] = field(default_factory=dict)
    scheduler: type = None
    scheduler_opts: dict[str, Any] = field(default_factory=dict)
    data_collector_opts: dict[str, Any] = field(default_factory=dict)
    policy_opts: dict[str, Any] = field(default_factory=dict)   # CTRACPolicyConfig kwargs
    vecnorm_opts: dict[str, Any] = field(default_factory=dict)
    # Default False, unlike the PPO/D3QN trainers: SAC's actor loss is alpha*log_pi - Q with
    # alpha fixed at the paper's 1.5, so normalising the reward rescales Q out from under a
    # constant alpha and the entropy term takes over (see marv_config_ctrac.yaml's comment).
    vecnorm_on_reward: bool = False
    ftr_obs_encoder_opts: dict[str, Any] = field(default_factory=dict)

    save_weights_every: int = 0
    max_eval_steps: int = 0
    eval_num_env_types: "int | None" = None
    eval_env_names_yaml: "str | None" = None
    use_wandb: bool = False
    use_tensorboard: bool = False
    policy_weights_path: "str | None" = None
    vecnorm_weights_path: "str | None" = None
    extra_env_transforms: list = field(default_factory=list)
    env_cfg_overrides: dict = field(default_factory=dict)
    sim_dt: float = 1 / 400
    decimation: int = 5
    solver_position_iterations: int = 16
    solver_velocity_iterations: int = 4
    max_depenetration_velocity: float = 0.15
    bounce_threshold_velocity: float = 0.2
    robot_linear_damping: float = 0.05
    robot_angular_damping: float = 0.05
    robot_max_linear_velocity: float = 10.0
    robot_max_angular_velocity: float = 720.0
    physx_gpu_heap_capacity: int = 2**28
    physx_gpu_temp_buffer_capacity: int = 2**26
    physx_gpu_max_num_partitions: int = 8
    physx_gpu_found_lost_aggregate_pairs_capacity: int = 2**27


# ============================================================
# BLOCK 4 — FtrSACTrainer
# ============================================================

class FtrSACTrainer:
    """Asymmetric SAC (Sec. V-A) + jointly-trained C-VAE (Sec. IV-C) trainer. Structurally
    mirrors FtrD3QNTrainer (same env setup, RunLogger, replay buffer, crash recovery) but
    swaps epsilon-greedy Double-DQN for torchrl's SACLoss/SoftUpdate, and adds a second,
    alternating supervised update for the C-VAE embedded inside the actor
    (rl_modules/ctrac/ctrac_policy.py's CTRACActorNet.cvae).
    """

    def __init__(self, raw_config: "DictConfig", ftr_gym_env: gymnasium.Env, optuna_trial=None):
        self.optuna_trial = optuna_trial
        self.config = FtrSACConfig(**raw_config)
        self.device = set_device(self.config.device)
        self.rng = seed_all(self.config.seed)

        self.run_logger = RunLogger(
            train_config=raw_config, category="sac", use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard, step_metric_name="collected_frames",
        )
        self._eval_num_env_types = self.config.eval_num_env_types or default_num_env_types(self.config.terrain)
        self._eval_env_type_names = load_env_type_names(self.config.terrain, self.config.eval_env_names_yaml, self._eval_num_env_types)
        self._eval_per_spot_csv = self.run_logger.logpath / "eval_per_spot.csv"
        if self.optuna_trial is not None:
            self.optuna_trial.set_user_attr("logpath", str(self.run_logger.logpath))
        self.term_logger = get_terminal_logger("ftr_sac_train")
        self.term_logger.info(f"Seed: {self.config.seed}")

        # ---- environment ----
        self.ftr_torchrl_env = FtrTorchRLEnv(
            ftr_gym_env, encoder_opts=self.config.ftr_obs_encoder_opts, device=self.device,
            shock_scale=self.config.env_cfg_overrides.get("shock_scale"),
        )
        self.env = self.ftr_torchrl_env

        # ---- policy (actor w/ embedded C-VAE + Q-network template) ----
        policy_cfg = CTRACPolicyConfig(**self.config.policy_opts)
        self.policy_operator, self.qvalue_operator, self.cvae, _optim_groups = policy_cfg.create(
            self.ftr_torchrl_env, device=self.device, weights_path=self.config.policy_weights_path,
        )
        # _optim_groups (policy_operator.parameters()/qvalue_operator.parameters()) is
        # NOT used below — see the SACLoss construction comment for why.

        # ---- transforms + VecNorm ----
        self.env, self.vecnorm = make_transformed_env(self.ftr_torchrl_env, self.config, policy_transforms=[])
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False)
            self.term_logger.info(f"Loaded vecnorm weights from {self.config.vecnorm_weights_path}")

        # ---- SAC loss + soft target updater ----
        self.loss_module = SACLoss(
            actor_network=self.policy_operator,
            qvalue_network=self.qvalue_operator,
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            alpha_init=self.config.alpha_init,
            fixed_alpha=self.config.fixed_alpha,
            target_entropy=self.config.target_entropy,
            action_spec=self.ftr_torchrl_env.action_spec,
        ).to(self.device)
        # gamma is no longer a SACLoss constructor kwarg in this torchrl version — passing
        # it there raises "gamma / lambda parameters through the loss constructor is a
        # deprecated feature" (confirmed via direct smoke test against the installed
        # torchrl). Set it via the value-estimator API instead.
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.config.gamma)
        self.target_updater = SoftUpdate(self.loss_module, tau=self.config.tau)

        # ---- optimizers (actor / qvalue / cvae / alpha, each separate — standard SAC
        # multi-optimizer pattern: each loss term backprops into a disjoint parameter set).
        #
        # Built from loss_module.actor_network_params / .qvalue_network_params, NOT
        # policy_operator.parameters() / qvalue_operator.parameters() directly — confirmed
        # via direct smoke test that SACLoss.convert_to_functional() re-wraps each
        # parameter into a new tensor object that ALIASES the same underlying storage as
        # the original module (in-place edits are mutually visible, so optimizing these
        # correctly updates the live policy_operator/self.cvae/qvalue_operator used for
        # rollout) but is a DIFFERENT object for autograd's `.grad`-accumulation purposes —
        # an optimizer built from the original modules' own .parameters() would silently
        # never receive gradients (loss_module's backward populates .grad only on ITS OWN
        # copies), which cost real debugging time to catch (a plain "loss.backward();
        # optim.step()" runs with no error either way — it would just silently never train).
        actor_params_named = list(self.loss_module.actor_network_params.named_parameters())
        actor_only_params = [p for n, p in actor_params_named if "cvae" not in n]
        qvalue_params = list(self.loss_module.qvalue_network_params.parameters())
        actor_optimizer_opts = self.config.policy_opts.get("actor_optimizer_opts", {}) or {}
        qvalue_optimizer_opts = self.config.policy_opts.get("qvalue_optimizer_opts", {}) or {}
        self.actor_optim = self.config.optimizer(actor_only_params, **actor_optimizer_opts)
        self.qvalue_optim = self.config.optimizer(qvalue_params, **qvalue_optimizer_opts)
        self._actor_only_params = actor_only_params
        self._qvalue_params = qvalue_params
        # The C-VAE update (_cvae_update) is a fully separate forward/backward pass through
        # self.cvae directly (not through loss_module at all), so this optimizer's params
        # are unaffected by the aliasing issue above.
        cvae_optimizer_opts = self.config.policy_opts.get("cvae_optimizer_opts", {}) or {}
        self.cvae_optim = self.config.optimizer(self.cvae.parameters(), **cvae_optimizer_opts)
        self.cvae_scheduler = None
        if self.config.scheduler is not None:
            # Deliberately a RISING schedule (start_factor < end_factor), the inverse of
            # every other schedule in this project — see marv_config_ctrac.yaml's comment:
            # the C-VAE should have little influence on the actor's input distribution
            # while SAC's rollout policy is still close to random (far from the Stage I
            # marv_rl-driven demonstration distribution it was pretrained on), ramping up
            # as the policy converges closer to it.
            self.cvae_scheduler = self.config.scheduler(self.cvae_optim, **(self.config.scheduler_opts or {}))
        self.alpha_optim = None
        if not self.config.fixed_alpha:
            self.alpha_optim = torch.optim.Adam([self.loss_module.log_alpha], **self.config.alpha_optimizer_opts)

        # ---- data collection ----
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.collector = SyncDataCollector(
            self.env, self.policy_operator, frames_per_batch=iteration_size,
            total_frames=self.config.total_frames, **self.config.data_collector_opts, device=self.device,
        )
        # LazyMemmapStorage (disk-backed, OS page cache) instead of LazyTensorStorage
        # (fully RAM-resident) — a CTRAC transition is unusually heavy vs. every other
        # module's replay buffer in this project: the packed observation (1227-dim,
        # dominated by the 960-cell privileged heightmap) is stored TWICE per transition
        # (obs + next_obs) plus an (8, 251) obs_history array needed so the C-VAE can be
        # trained on replayed, temporally-correct history windows (see CTRACActorNet's
        # docstring) — roughly 17.9 KB/transition. At replay_buffer_capacity's full size
        # that's tens of GB held entirely in RAM with LazyTensorStorage, which OOM-killed a
        # real training run (confirmed: throughput degraded steadily from ~1200 to ~500
        # frames/s over 37 minutes before the kill, the classic memory-growth/thrashing
        # signature, dying right around when the buffer neared capacity) — D3QN/PPO's
        # much smaller, single-obs, no-history transitions never hit this at the same
        # nominal capacity, which is why replay_buffer_capacity was copied from those
        # configs without anyone (including this) reconsidering the per-transition size.
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=self.config.replay_buffer_capacity, ndim=1,
                scratch_dir=str(self.run_logger.logpath / "replay_buffer"),
                # train_ctrac.sbatch's respawn loop re-invokes this script fresh after a
                # crash, reusing the same attempt/logpath — existsok=True so a scratch dir
                # left over from the killed attempt doesn't itself crash the respawn
                # (the replay buffer restarting empty after a respawn is expected/fine,
                # same as this project's other off-policy trainers never persisting it
                # across crash-restarts either).
                existsok=True,
            ),
            sampler=RandomSampler(), batch_size=self.config.batch_size,
        )

        self._grad_steps = 0
        self.term_logger.info("Initialized FtrSACTrainer.")

    # ------------------------------------------------------------------
    def train(self):
        try:
            self._train()
            post_log = self._post_training_evaluation()
        except KeyboardInterrupt:
            self.term_logger.info("Training interrupted by user.")
            post_log = None
        except Exception as e:
            self.term_logger.error(f"Training failed: {e}")
            traceback.print_exception(e)
            if "CUDA error" in str(e) or "CUDA out of memory" in str(e) or "CommError" in str(type(e).__name__):
                self.term_logger.error(f"{type(e).__name__} detected — calling os._exit(75) to skip cleanup.")
                import os as _os
                _os._exit(75)
            try:
                self.run_logger.save_weights(self.policy_operator.state_dict(), "policy_crash")
                self.run_logger.save_weights(self.qvalue_operator.state_dict(), "qvalue_crash")
                self.run_logger.save_weights(self.cvae.state_dict(), "cvae_crash")
                self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_crash")
                self.term_logger.info(f"Saved crash checkpoint at frames={getattr(self, '_current_total_frames', 0)}.")
            except Exception:
                pass
            raise
        finally:
            if self.run_logger is not None:
                self.run_logger.close()
        return post_log

    def _cvae_update(self, batch) -> dict[str, float]:
        """One supervised gradient step for the C-VAE (Eq. 10-14), on a batch of
        precomputed obs_history windows (see CTRACActorNet's docstring for why these are
        stored at collection time rather than reconstructed from a random minibatch)."""
        obs_hist = batch["obs_history"].to(self.device)         # (B, H, PARTIAL_DIM)
        obs = batch[OBS_KEY].to(self.device)                    # (B, obs_dim) — this step's ground truth
        next_obs = batch[("next", OBS_KEY)].to(self.device)     # (B, obs_dim) — reconstruction target

        target_points = obs[..., CONTACT_POINTS_OFFSET:CONTACT_PROB_OFFSET].reshape(-1, 4, 3)
        target_prob = obs[..., CONTACT_PROB_OFFSET:CONTACT_PROB_OFFSET + 4]
        target_recon = next_obs[..., :PARTIAL_DIM]

        _z, mu, logvar, pred_points, pred_prob, pred_recon = self.cvae(obs_hist, sample=True)

        loss_vae, recon_l, kl_l = vae_loss(pred_recon, target_recon, mu, logvar, beta=self.config.cvae_vae_beta)
        loss_prob = contact_prob_loss(pred_prob, target_prob)
        loss_est = contact_est_loss(pred_points, target_points, target_prob)
        loss_geo = contact_geo_loss(pred_points, target_prob, target_points.mean(dim=1), self.config.cvae_max_reach)
        loss = (
            loss_vae
            + self.config.cvae_prob_weight * loss_prob
            + self.config.cvae_est_weight * loss_est
            + self.config.cvae_geo_weight * loss_geo
        )

        self.cvae_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), self.config.max_grad_norm, error_if_nonfinite=False, norm_type=self.config.clip_grad_norm_p)
        self.cvae_optim.step()

        return {
            "cvae_loss": loss.item(), "cvae_loss_recon": recon_l.item(), "cvae_loss_kl": kl_l.item(),
            "cvae_loss_prob": loss_prob.item(), "cvae_loss_est": loss_est.item(), "cvae_loss_geo": loss_geo.item(),
        }

    def _sac_update(self, batch) -> dict[str, float]:
        batch = batch.to(self.device)
        loss_td = self.loss_module(batch)

        actor_loss = loss_td["loss_actor"]
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._actor_only_params, self.config.max_grad_norm, error_if_nonfinite=False, norm_type=self.config.clip_grad_norm_p,
        )
        if actor_grad_norm.isfinite():
            self.actor_optim.step()

        qvalue_loss = loss_td["loss_qvalue"]
        self.qvalue_optim.zero_grad()
        qvalue_loss.backward()
        qvalue_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._qvalue_params, self.config.max_grad_norm, error_if_nonfinite=False, norm_type=self.config.clip_grad_norm_p,
        )
        if qvalue_grad_norm.isfinite():
            self.qvalue_optim.step()

        log = {
            "loss_actor": actor_loss.item(), "loss_qvalue": qvalue_loss.item(),
            "alpha": self.loss_module._alpha.item() if hasattr(self.loss_module, "_alpha") else self.config.alpha_init,
        }
        if self.alpha_optim is not None and "loss_alpha" in loss_td.keys():
            alpha_loss = loss_td["loss_alpha"]
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            log["loss_alpha"] = alpha_loss.item()

        self.target_updater.step()
        self._grad_steps += 1
        return log

    def _train(self):
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.term_logger.info(f"Iteration size: {iteration_size}, batch_size: {self.config.batch_size}, updates/iter: {self.config.updates_per_batch}")
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        pbar = tqdm(total=self.config.total_frames, desc="C-TRAC SAC Training", unit="frames", leave=False)

        for i, tensordict_data in enumerate(self.collector):
            total_collected_frames = (i + 1) * iteration_size
            self._current_total_frames = total_collected_frames
            pbar.update(iteration_size)

            self.policy_operator.train()
            self.env.train()

            flat = tensordict_data.reshape(-1)
            explosion_flags = flat.get(("next", "explosion"), None)
            n_dirty = 0
            if explosion_flags is not None:
                clean_mask = ~explosion_flags.reshape(-1)
                n_dirty = int((~clean_mask).sum().item())
                if n_dirty > 0:
                    flat = flat[clean_mask]

            if flat.shape[0] == 0:
                self.term_logger.warning("All transitions exploded this iteration — skipping.")
                log = {**self.ftr_torchrl_env.pop_reward_info(), **self.ftr_torchrl_env.pop_termination_info(), **self.ftr_torchrl_env.pop_state_stats()}
                self.run_logger.log_data(log, total_collected_frames)
                continue

            nan_count = 0
            for key in flat.keys(include_nested=True, leaves_only=True):
                t = flat[key]
                if t.is_floating_point():
                    bad = ~t.isfinite()
                    if bad.any():
                        nan_count += int(bad.sum().item())
                        flat[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if nan_count > 0:
                self.term_logger.warning(f"Sanitized {nan_count} non-finite values in rollout tensordict.")

            rollout_mean_reward = flat[("next", "reward")].mean().item()
            actions = flat["action"]
            action_log = {
                "action/v_mean": actions[:, 0].mean().item(), "action/w_mean": actions[:, 1].mean().item(),
                "action/front_left_mean": actions[:, 2].mean().item(), "action/front_right_mean": actions[:, 3].mean().item(),
                "action/rear_left_mean": actions[:, 4].mean().item(), "action/rear_right_mean": actions[:, 5].mean().item(),
            }

            # SACLoss evaluates the actor on the "next" sub-tensordict to build the target
            # action, and _CTRACActorTDModule feeds whatever "obs_history" it finds there to
            # the C-VAE. Only the root carries one (the actor wrote it at collection time),
            # so without this the target action's z would come from CTRACObsHistory's
            # constant-frame fallback — the same defect the root key fixes for the online
            # action. The next window is exactly the current one shifted by one frame with
            # next_obs's partial slice appended, so it is computed here rather than stored
            # again (which would add another (H, 251) array per transition to an already
            # heavy replay entry).
            hist = flat["obs_history"]
            next_partial = flat[("next", OBS_KEY)][..., :PARTIAL_DIM].unsqueeze(-2)
            flat.set(("next", "obs_history"), torch.cat([hist[..., 1:, :], next_partial], dim=-2))

            self.replay_buffer.extend(flat)
            del tensordict_data, flat
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            sac_log, cvae_log = {}, {}
            if len(self.replay_buffer) >= self.config.min_replay_size:
                for _ in range(self.config.updates_per_batch):
                    sub_batch = self.replay_buffer.sample()
                    sac_log = self._sac_update(sub_batch)
                    for _ in range(self.config.cvae_updates_per_sac_step):
                        cvae_batch = self.replay_buffer.sample()
                        cvae_log = self._cvae_update(cvae_batch)

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            if self.cvae_scheduler is not None:
                self.cvae_scheduler.step()

            # Key order matters: RunLogger._write_row groups a row's keys by topic (the
            # prefix before the last "/") via itertools.groupby, which only merges
            # CONSECUTIVE same-topic keys rather than sorting first. Putting
            # "explosions/dirty_transitions" between the two "train/*" runs below would
            # split them into two non-adjacent groups -> two half-populated rows per step
            # in train.csv (same underlying bug fixed in _get_eval_rollout_results for
            # eval.csv) — all "train/*" keys must stay contiguous.
            log = {
                **action_log,
                **self.ftr_torchrl_env.pop_reward_info(), **self.ftr_torchrl_env.pop_termination_info(), **self.ftr_torchrl_env.pop_state_stats(),
                "explosions/dirty_transitions": n_dirty,
                "train/mean_reward": rollout_mean_reward, "train/replay_buffer_size": len(self.replay_buffer),
                **{f"train/{k}": v for k, v in sac_log.items()}, **{f"train/{k}": v for k, v in cvae_log.items()},
            }

            save_every = self.config.save_weights_every or self.config.eval_and_save_every
            if i % save_every == 0:
                self.run_logger.save_weights(self.policy_operator.state_dict(), f"policy_step_{total_collected_frames}")
                self.run_logger.save_weights(self.qvalue_operator.state_dict(), f"qvalue_step_{total_collected_frames}")
                self.run_logger.save_weights(self.cvae.state_dict(), f"cvae_step_{total_collected_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
            if i % self.config.eval_and_save_every == 0 and i > 0:
                try:
                    eval_log = self._get_eval_rollout_results()
                    for _ in range(self.config.eval_repeats - 1):
                        for k, v in self._get_eval_rollout_results().items():
                            eval_log[k] += v
                    if self.config.eval_repeats > 1:
                        for k in eval_log:
                            eval_log[k] /= self.config.eval_repeats
                    log.update(eval_log)
                    if self.optuna_trial is not None:
                        success_rate = eval_log.get("eval/success_rate", 0.0)
                        self.optuna_trial.report(success_rate, i // self.config.eval_and_save_every)
                        if self.optuna_trial.should_prune():
                            raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.term_logger.warning("Eval CUDA error — GPU context corrupted. Exiting immediately.")
                        import os as _os
                        _os._exit(75)
                    self.term_logger.warning(f"Eval rollout failed (physics explosion): {e}. Skipping eval metrics.")

            self.run_logger.log_data(log, total_collected_frames)

        self.run_logger.save_weights(self.policy_operator.state_dict(), "policy_final")
        self.run_logger.save_weights(self.qvalue_operator.state_dict(), "qvalue_final")
        self.run_logger.save_weights(self.cvae.state_dict(), "cvae_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")

    def _get_eval_rollout_results(self) -> dict[str, float]:
        self.env.eval()
        self.policy_operator.eval()
        max_eval_steps = self.config.max_eval_steps or (self.ftr_torchrl_env.ftr_env.unwrapped.max_episode_length * 2)
        self.ftr_torchrl_env.enable_per_env_tracking()
        results, episode_records = run_tracked_rollout(
            self.env, self.ftr_torchrl_env, self.ftr_torchrl_env.ftr_env, self.policy_operator, max_eval_steps,
            num_env_types=self._eval_num_env_types, env_type_names=self._eval_env_type_names, terrain=self.config.terrain,
        )
        self.ftr_torchrl_env.disable_per_env_tracking()
        # RunLogger._write_row groups a logged row's keys by topic (the prefix before the
        # last "/") using itertools.groupby, which only merges CONSECUTIVE same-topic
        # keys — it does not sort first. run_tracked_rollout's returned dict interleaves
        # "rew/"/"shock/" keys between two separate runs of "eval/" keys, which splits a
        # single eval step into two non-adjacent groupby groups -> two half-populated CSV
        # rows for the same collected_frames value (confirmed against a real run's
        # eval.csv: eval/mean_step_reward etc. on one row, eval/success_rate etc. on the
        # next). train_d3qn.py's own _get_eval_rollout_results already pops these same
        # keys out for exactly this reason — replicated here (this project's other
        # trainers keep them, but self._eval_reward_info was never actually consumed
        # anywhere in this trainer either, matching D3QN's own usage).
        self._eval_reward_info = {k: results.pop(k) for k in list(results) if k.startswith("rew/") or k.startswith("shock/")}

        if episode_records:
            per_env_rows = aggregate_per_env(
                episode_records=episode_records, env_type_names=self._eval_env_type_names,
                eval_id="train", policy=self.run_logger.run_name, terrain=self.config.terrain, repeat=1, obs_stats=results,
            )
            for row in per_env_rows:
                results[f"eval_per_env/{row.env_type_name}_success_rate"] = row.success_rate
            per_spot_rows = aggregate_per_spot(
                episode_records=episode_records, env_type_names=self._eval_env_type_names,
                num_depth_cols=10, eval_id="train", policy=self.run_logger.run_name, terrain=self.config.terrain, repeat=1,
            )
            save_per_spot_csv(self._eval_per_spot_csv, per_spot_rows)
        return results

    def _post_training_evaluation(self) -> dict[str, float]:
        self.term_logger.info(f"Training finished. Running {self.config.eval_repeats_after_training} final eval(s).")
        avg = self._get_eval_rollout_results()
        for _ in range(self.config.eval_repeats_after_training - 1):
            for k, v in self._get_eval_rollout_results().items():
                avg[k] += v
        for k in avg:
            avg[k] /= self.config.eval_repeats_after_training
        print("\nFinal evaluation results:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        return avg


# ============================================================
# BLOCK 5 — Entry point
# ============================================================

def _load_raw_config(config_path: str, cli_overrides: list[str]):
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


if __name__ == "__main__":
    prev_cfg_path = RunLogger.latest_attempt_config()
    if prev_cfg_path is not None:
        print(f"[INFO] Respawn detected — loading config from previous attempt: {prev_cfg_path}", flush=True)
        raw_cfg = _load_raw_config(str(prev_cfg_path), unknown_args)
        if not raw_cfg:
            print(f"[WARNING] Previous attempt config at {prev_cfg_path} is empty — falling back to {args.config}", flush=True)
            raw_cfg = _load_raw_config(args.config, unknown_args)
    else:
        raw_cfg = _load_raw_config(args.config, unknown_args)

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.terrain is not None:
        raw_cfg.terrain = args.terrain
    if args.task is not None:
        raw_cfg.task = args.task

    import os
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() returned False after AppLauncher init.", flush=True)
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    _cfg = FtrSACConfig(**raw_cfg)

    spec = gymnasium.spec(_cfg.task)
    _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
        import importlib
        _mod_path, _cls_name = _env_cfg_entry.rsplit(":", 1)
        _EnvCfgClass = getattr(importlib.import_module(_mod_path), _cls_name)
    elif isinstance(_env_cfg_entry, type):
        _EnvCfgClass = _env_cfg_entry
    else:
        from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg
        _EnvCfgClass = CrossingEnvCfg

    env_cfg = _EnvCfgClass()
    env_cfg.scene.num_envs = _cfg.num_robots
    env_cfg.terrain_name = _cfg.terrain

    env_cfg.sim.dt = _cfg.sim_dt
    env_cfg.decimation = _cfg.decimation

    env_cfg.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = _cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = _cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = _cfg.robot_angular_damping

    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = _cfg.solver_velocity_iterations

    env_cfg.sim.physx.min_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = _cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = _cfg.bounce_threshold_velocity
    env_cfg.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = _cfg.physx_gpu_found_lost_aggregate_pairs_capacity

    # Scale down GPU PhysX buffers for small env counts (e.g. local debug runs on laptop
    # GPUs) — mirrors eval_ftr.py's own scaling exactly, never carried over into this file
    # originally. FTR_SIM_CFG's defaults are sized for 4096 envs on server GPUs; this is
    # what actually caused local ctrac runs to fail scene creation regardless of env count
    # or ContactSensor/terrain settings (confirmed by direct comparison against
    # eval_ftr.py, which already scales these and works locally).
    if _cfg.num_robots <= 64:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22
    elif _cfg.num_robots > 512:
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 27

    # Must set module_name: ctrac here (via env_cfg_overrides in the yaml) so FtrEnv routes
    # observations/rewards through CTRACModule and attaches the ContactSensor (ftr_env.py).
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    trainer = FtrSACTrainer(raw_cfg, ftr_gym_env)
    trainer.train()

    import os as _os
    _os._exit(0)
