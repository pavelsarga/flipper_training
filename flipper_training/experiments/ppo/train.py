from typing import TYPE_CHECKING
import torch
import traceback
from flipper_training.experiments.ppo.common import (
    prepare_env,
    log_from_eval_rollout,
    parse_and_load_config,
    make_transformed_env,
    EVAL_LOG_OPT,
    make_formatted_str_lines,
)
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
# Was `from config import PPOExperimentConfig, OmegaConf` -- a bare `config` module only resolves
# when this file is run as a plain script from within experiments/ppo/ (script-dir sys.path[0]
# aliases the sibling config.py to top-level `config`). It raises `ModuleNotFoundError: No module
# named 'config'` under the CLAUDE.md-documented `python -m flipper_training.experiments.ppo.train`
# invocation, since `-m` does not add the submodule's own directory to sys.path. Fully-qualified
# imports work under both invocation styles.
from flipper_training.experiments.ppo.config import PPOExperimentConfig
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from flipper_training.environment.env import Env
from flipper_training.utils.logutils import RunLogger, get_terminal_logger
from torchrl.modules import ActorValueOperator

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict


def train_step_to_log(rollout_td: "TensorDict", loss_td: "TensorDict", grad_norm: torch.Tensor, optim: torch.optim.Optimizer) -> dict[str, float]:
    """
    Extract important data from the training tensordicts to log.
    """
    return {
        "train/mean_action_sample_log_prob": rollout_td["sample_log_prob"].mean().item(),
        "train/mean_critic_loss": loss_td["loss_critic"].mean().item(),
        "train/mean_objective_loss": loss_td["loss_objective"].mean().item(),
        "train/mean_entropy_loss": loss_td["loss_entropy"].mean().item(),
        "train/mean_entropy": loss_td["entropy"].mean().item(),
        "train/mean_kl_approx": loss_td["kl_approx"].mean().item(),
        "train/mean_clip_fraction": loss_td["clip_fraction"].mean().item(),
        "train/mean_advantage": rollout_td["advantage"].mean().item(),
        "train/std_advantage": rollout_td["advantage"].std().item(),
        "train/total_grad_norm": grad_norm.item(),
        **{f"train/{g['name']}_lr": g["lr"] for g in optim.param_groups},
    }


class PPOTrainer:
    def __init__(self, config: "DictConfig"):
        self.config = PPOExperimentConfig(**config)
        self.run_logger = RunLogger(
            train_config=config,
            category="ppo",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="collected_frames",
        )
        self.term_logger = get_terminal_logger("ppo_train")
        self.env, self.device, self.rng = prepare_env(self.config, mode="train")
        self.policy_config = self.config.policy_config(**self.config.policy_opts)
        self.actor_value_wrapper, self.optim_groups, self.policy_transforms = self.policy_config.create(
            env=self.env,
            weights_path=self.config.policy_weights_path,
            device=self.device,
        )
        self.actor_operator = self.actor_value_wrapper.get_policy_operator()
        self.value_operator = self.actor_value_wrapper.get_value_operator()
        self.env, self.vecnorm = make_transformed_env(self.env, self.config, self.policy_transforms)
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False)
        self.collector = SyncDataCollector(
            self.env,
            self.actor_operator,
            frames_per_batch=self.config.time_steps_per_batch * self.config.num_robots,  # iteration size
            total_frames=self.config.total_frames,
            **self.config.data_collector_opts,
            device=self.device,
        )
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.config.time_steps_per_batch * self.config.num_robots,  # iteration size
                ndim=1,  # This storage will mix the batch and time dimensions
                device=self.device,
                compilable=True,
            ),
            sampler=SamplerWithoutReplacement(drop_last=True, shuffle=True),
            batch_size=self.config.frames_per_sub_batch,  # sample flattened and mixed data
            dim_extend=0,
            compilable=True,
        )
        self.advantage_module = GAE(**self.config.gae_opts, value_network=self.value_operator, time_dim=1, device=self.device, differentiable=False)
        self.advantage_module = self.advantage_module.to(self.config.training_dtype)
        self.loss_module = ClipPPOLoss(
            self.actor_operator,
            self.actor_value_wrapper.get_value_head() if isinstance(self.actor_value_wrapper, ActorValueOperator) else self.value_operator,
            **self.config.ppo_opts,
        )
        self.loss_module = self.loss_module.to(self.config.training_dtype)
        self.optim = self.config.optimizer(
            self.optim_groups,
            **(self.config.optimizer_opts or {}),
        )
        self.scheduler = self.config.scheduler(
            self.optim,
            **(self.config.scheduler_opts or {}),
        )
        self.term_logger.info("Initialized PPOTrainer with the following configuration:")
        print(OmegaConf.to_yaml(config, sort_keys=True))

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
        iteration_size = (
            self.config.time_steps_per_batch * self.config.num_robots
        )  # total number of frames per iteration (stable baselines convention)
        if not iteration_size % self.config.frames_per_sub_batch == 0:
            raise ValueError(
                f"The iteration size must be divisible by the frames per sub-batch: {iteration_size % self.config.frames_per_sub_batch=}"
            )
        self.term_logger.info(
            f"Iteration size: {iteration_size}, frames per sub-batch: {self.config.frames_per_sub_batch}, "
            f"optimization steps: {iteration_size // self.config.frames_per_sub_batch}"
        )
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        # Training loop
        pbar = tqdm(total=self.config.total_frames, desc="Training", unit="frames", leave=False)
        for i, tensordict_data in enumerate(self.collector):
            total_collected_frames = (i + 1) * iteration_size
            pbar.update(iteration_size)
            # collected (B, T, *specs) where B is the batch size and T the number of steps
            tensordict_data.pop(Env.STATE_KEY)  # we don't need this
            tensordict_data.pop(("next", Env.STATE_KEY))  # we don't need this
            self.actor_operator.train()
            self.value_operator.train()
            self.env.train()
            # Optimization
            for j in range(self.config.epochs_per_batch):
                self.advantage_module(tensordict_data)
                self.replay_buffer.extend(tensordict_data.reshape(-1))  # we can now safely flatten the data
                for k in range(iteration_size // self.config.frames_per_sub_batch):
                    sub_batch = self.replay_buffer.sample()
                    loss_vals = self.loss_module(sub_batch)
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    loss_value.backward()
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor_value_wrapper.parameters(),
                            self.config.max_grad_norm,
                            error_if_nonfinite=True,
                            norm_type=self.config.clip_grad_norm_p,
                        )
                    except Exception as e:
                        self.term_logger.error(f"Gradient norm calculation failed: {e}")
                        self.run_logger.save_weights(
                            self._full_policy_state_with_grads(),
                            f"actor_value_wrapper_failed_grad_norm_total_frames_{total_collected_frames}_sub_batch_{j}_step_{k}",
                        )
                        raise e

                    self.optim.step()
                    self.optim.zero_grad()
            self.scheduler.step()
            log = train_step_to_log(tensordict_data, loss_vals, grad_norm, self.optim)
            if i % self.config.eval_and_save_every == 0:
                eval_log = self._get_eval_rollout_results()
                log.update(eval_log)
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{total_collected_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
            self.run_logger.log_data(log, total_collected_frames)

        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")
        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{self.config.total_frames}")
        self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{self.config.total_frames}")

    def _full_policy_state_with_grads(self):
        """
        Returns the full state of the loss module, including the state of the actor and critic.
        """
        return {
            "loss_module": self.loss_module.state_dict(),
            "actor_value_wrapper": self.actor_value_wrapper.state_dict(),
            "actor_value_wrapper_grads": {k: v.grad for k, v in self.actor_value_wrapper.named_parameters() if v.grad is not None},
            "vecnorm": self.vecnorm.state_dict(),
        }

    def _get_eval_rollout_results(self) -> dict[str, int | float]:
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

    def _post_training_evaluation(self):
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
    trainer = PPOTrainer(cfg)
    trainer.train()
