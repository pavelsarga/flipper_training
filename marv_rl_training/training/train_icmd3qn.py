# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
# Imported here, before AppLauncher, to match the load order train_d3qn.py uses when it is
# the __main__ script; the inherited _train() is what actually reports to the trial.
import optuna  # noqa: F401

from marv_rl_training.training.cli import launch_isaac_app, train_arg_parser

if __name__ == "__main__":
    parser = train_arg_parser("Train ICM-D3QN (Pan et al. 2023) flipper policy inside FTR-Benchmark (Isaac Sim)", play=True)
    args, unknown_args, simulation_app = launch_isaac_app(parser)

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
from dataclasses import dataclass, field
from typing import Any

import torch

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.training.env_setup import build_ftr_gym_env, import_ftr_tasks, require_cuda
from marv_rl_training.training.entrypoint import resolve_train_config, run_trainer
from marv_rl_training.training.eval_common import exit_flushed
from marv_rl_training.training.train_d3qn import FtrD3QNConfig, FtrD3QNTrainer

from rl_modules.icmd3qn.icmd3qn_icm import ICMD3QNCuriosityModule
from rl_modules.icmd3qn.icmd3qn_policy import ICMD3QNPolicy


# ============================================================
# BLOCK 3 — Config
# ============================================================

@dataclass
class FtrICMD3QNConfig(FtrD3QNConfig):
    """AT-D3QN's config plus the curiosity-module fields Eq. 11-14 need.

    Everything else — env, replay buffer, epsilon schedule, physics, checkpointing — is
    inherited unchanged from FtrD3QNConfig, which is the honest description of the two
    methods: ICM-D3QN is AT-D3QN with an intrinsic reward term.

    env_cfg_overrides must set `module_name: icmd3qn` so FtrEnv computes ICM-D3QN's 18-D
    obs/reward (see rl_modules/registry.py) and FtrTorchRLEnv picks ICMD3QNObservation.
    """

    icm_opts: dict[str, Any] = field(default_factory=dict)            # ICMD3QNCuriosityModule kwargs
    icm_optimizer_opts: dict[str, Any] = field(default_factory=dict)  # reuses `optimizer` (e.g. AdamW)
    icm_beta_forward: float = 1.0   # beta_F (Eq. 15)
    icm_beta_inverse: float = 1.0   # beta_I (Eq. 15)
    icm_weights_path: str | None = None


# ============================================================
# BLOCK 4 — FtrICMD3QNTrainer
# ============================================================

class FtrICMD3QNTrainer(FtrD3QNTrainer):
    """AT-D3QN plus an Intrinsic Curiosity Module trained jointly on every sampled batch.

    R_t = R^e_t + R^i_t (Eq. 13), where R^e_t comes from ICMD3QNModule.get_reward_components()
    and R^i_t from the curiosity module here. The DQN loop, env setup, RunLogger, crash
    recovery and replay-buffer persistence are FtrD3QNTrainer's, unchanged: the ICM enters
    through the aux-module / aux-optimizer / reward hooks, so it is checkpointed, resumed and
    logged with no separate copy of any of that machinery.
    """

    CONFIG_CLASS = FtrICMD3QNConfig
    POLICY_CLASS = ICMD3QNPolicy
    RUN_CATEGORY = "icmd3qn"
    TERM_LOGGER_NAME = "ftr_icmd3qn_train"

    def _build_aux_modules(self) -> dict[str, torch.nn.Module]:
        icm = ICMD3QNCuriosityModule(**self.config.icm_opts).to(self.device)
        if self.config.icm_weights_path is not None:
            icm.load_state_dict(torch.load(self.config.icm_weights_path, map_location=self.device), strict=False)
            self.term_logger.info(f"Loaded ICM weights from {self.config.icm_weights_path}")
        return {"icm": icm}

    def _build_aux_optimizers(self) -> dict[str, Any]:
        opts = self.config.icm_optimizer_opts or self.config.optimizer_opts or {}
        return {"icm": self.config.optimizer(self.aux_modules["icm"].parameters(), **opts)}

    @property
    def icm(self) -> torch.nn.Module:
        return self.aux_modules["icm"]

    def _augment_reward(self, obs, action_idx, next_obs, reward):
        """Train psi/F/I on this batch (Eq. 11, 14 -> Eq. 15), then add R^i_t (Eq. 12, 13).

        The intrinsic reward is recomputed under no_grad after the ICM step, so the TD target
        uses the just-updated curiosity estimate rather than the pre-update one.
        """
        icm = self.icm
        psi_t1, psi_t1_hat, action_logits = icm(obs, action_idx, next_obs)
        l_forward, l_inverse = icm.losses(psi_t1, psi_t1_hat, action_logits, action_idx)
        icm_loss = self.config.icm_beta_forward * l_forward + self.config.icm_beta_inverse * l_inverse

        icm_optim = self.aux_optimizers["icm"]
        icm_optim.zero_grad()
        icm_loss.backward()
        icm_optim.step()

        with torch.no_grad():
            psi_t1, psi_t1_hat, _ = icm(obs, action_idx, next_obs)
            intrinsic_reward = icm.intrinsic_reward(psi_t1, psi_t1_hat)

        return reward + intrinsic_reward, {
            "icm_loss_forward": l_forward.item(),
            "icm_loss_inverse": l_inverse.item(),
            "icm_intrinsic_reward_mean": intrinsic_reward.mean().item(),
            "extrinsic_reward_mean": reward.mean().item(),
        }


# ============================================================
# BLOCK 5 — Entry point
# ============================================================

if __name__ == "__main__":
    raw_cfg = resolve_train_config(args, unknown_args)

    require_cuda()
    import_ftr_tasks()

    # FtrICMD3QNConfig is built here only to read the env fields build_ftr_gym_env needs.
    _cfg = FtrICMD3QNConfig(**raw_cfg)
    ftr_gym_env = build_ftr_gym_env(_cfg)

    if args.play is not None:
        FtrICMD3QNTrainer.play(_cfg, ftr_gym_env, simulation_app)
    else:
        run_trainer(FtrICMD3QNTrainer, raw_cfg, ftr_gym_env)

    # Skip simulation_app.close() — Isaac Sim's shutdown re-initialises GPU foundation and
    # frequently deadlocks, keeping the SLURM slot busy for hours.
    exit_flushed()
