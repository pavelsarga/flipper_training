import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass, field
from flipper_training.environment.env import Env
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    ActorCriticWrapper,
    ActorValueOperator,
)
from flipper_training.utils.logutils import get_terminal_logger
from rich.console import Console
from rich.table import Table
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["MLPPolicyConfig"]


def count_parameters(module: nn.Module) -> int:
    """Counts the total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class MLPPolicyConfig(PolicyConfig):
    share_encoder: bool
    actor_mlp_opts: dict
    value_mlp_opts: dict
    actor_optimizer_opts: dict
    value_optimizer_opts: dict
    apply_baselines_init: bool = False
    extra_distribution_kwargs: dict = field(default_factory=dict)
    distribution_class: type = TanhNormal

    def __post_init__(self):
        self.logger = get_terminal_logger("MLPPolicyConfig")

    def create(self, env: Env, **kwargs):
        # Fetch the environment data
        action_spec = env.action_spec
        encoders = {o.name: o.get_encoder() for o in env.observations}
        # Create the encoder
        encoder = EncoderCombiner(encoders)
        if self.share_encoder:
            actor_value_wrapper = self._create_shared(action_spec, encoder)
            optim_groups = [
                {
                    "params": actor_value_wrapper.get_policy_operator().parameters(),
                    "name": "policy_and_encoder",
                    **self.actor_optimizer_opts,
                },
                {
                    "params": actor_value_wrapper.get_value_head().parameters(),  # this is only the value MLP head
                    "name": "value_head",
                    **self.value_optimizer_opts,
                },
            ]
            self.logger.info("Using shared encoder for actor and critic. Actor's optimizer settings will be used for the encoder.")
        else:
            actor_value_wrapper = self._create_separate(action_spec, encoder)
            optim_groups = [
                {
                    "params": actor_value_wrapper.get_policy_operator().parameters(),
                    "name": "policy_operator",
                    **self.actor_optimizer_opts,
                },
                {
                    "params": actor_value_wrapper.get_value_operator().parameters(),
                    "name": "value_operator",
                    **self.value_optimizer_opts,
                },
            ]
        if kwargs.get("device", None) is not None:
            actor_value_wrapper.to(kwargs["device"])

        # Apply initialization if needed
        if (
            self.apply_baselines_init
        ):  # Apply orthogonal initialization to the actor and value operators before loading weights (we might have some new weights)
            self._apply_baselines_init(
                actor_value_wrapper.get_policy_operator(),
                actor_value_wrapper.get_value_operator(),
                action_spec.shape[1],
            )
            self.logger.info("Applied orthogonal initialization to the actor and value operators.")

        # Load weights if path provided
        if weights_path := kwargs.get("weights_path", None):
            missing_unexpected = actor_value_wrapper.load_state_dict(torch.load(weights_path, map_location=actor_value_wrapper.device), strict=False)
            self.logger.info(f"Loaded weights from {kwargs['weights_path']}")
            if missing_unexpected.missing_keys:
                self.logger.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        # Log parameter counts before returning
        MLPPolicyConfig._log_parameter_counts(actor_value_wrapper, self.share_encoder)

        return actor_value_wrapper, optim_groups, []

    def _create_separate(self, action_spec, combined_encoder: EncoderCombiner):
        if "in_features" in self.actor_mlp_opts or "out_features" in self.actor_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the policy MLP options. They are dictated by the environment and the encoder."
            )
        actor_encoder_module = TensorDictModule(
            deepcopy(combined_encoder),
            in_keys={k: k for k in combined_encoder.encoders.keys()},
            out_keys=["y_actor"],
            out_to_in_map=True,
        )
        actor_module = TensorDictModule(
            module=nn.Sequential(
                MLP(in_dim=combined_encoder.output_dim, out_dim=2 * action_spec.shape[1], **self.actor_mlp_opts),
                NormalParamExtractor(),
            ),
            in_keys=["y_actor"],
            out_keys=["loc", "scale"],
        )
        actor_module = ProbabilisticActor(
            module=TensorDictSequential([actor_encoder_module, actor_module]),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=self.distribution_class,
            distribution_kwargs={
                "low": action_spec.space.low[0],  # pass only the values without a batch dimension
                "high": action_spec.space.high[0],  # pass only the values without a batch dimension
                **self.extra_distribution_kwargs,
            },
            return_log_prob=True,
        )
        if "in_features" in self.value_mlp_opts or "out_features" in self.value_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the value MLP options. They are dictated by the environment and the encoder."
            )
        value_encoder_module = TensorDictModule(
            deepcopy(combined_encoder),
            in_keys={k: k for k in combined_encoder.encoders.keys()},
            out_keys=["y_value"],
            out_to_in_map=True,
        )
        value_module = TensorDictModule(
            MLP(in_dim=combined_encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_value"],  # pass the observations as input
            out_keys=["state_value"],
        )
        value_operator = TensorDictSequential(
            [value_encoder_module, value_module],
        )
        wrapper = ActorCriticWrapper(
            policy_operator=actor_module,
            value_operator=value_operator,
        )
        if id(actor_encoder_module.module) == id(value_encoder_module.module):
            raise ValueError("The encoder module for the policy and value operators should not be the same instance. This is a bug in the code.")
        return wrapper

    def _create_shared(self, action_spec, combined_encoder: EncoderCombiner):
        if "in_features" in self.actor_mlp_opts or "out_features" in self.actor_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the policy MLP options. They are dictated by the environment and the encoder."
            )
        actor_td = TensorDictModule(
            nn.Sequential(
                MLP(in_dim=combined_encoder.output_dim, out_dim=2 * action_spec.shape[1], **self.actor_mlp_opts),
                NormalParamExtractor(),
            ),
            in_keys=["y_shared"],
            out_keys=["loc", "scale"],
        )
        actor_module = ProbabilisticActor(
            module=actor_td,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],  # pass only the values without a batch dimension
                "high": action_spec.space.high[0],  # pass only the values without a batch dimension
                **self.extra_distribution_kwargs,
            },
            return_log_prob=True,
        )
        value_operator = ValueOperator(
            module=MLP(in_dim=combined_encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_shared"],  # pass the observations as input
        )
        encoder_module = TensorDictModule(
            combined_encoder,
            in_keys={k: k for k in combined_encoder.encoders.keys()},
            out_keys=["y_shared"],
            out_to_in_map=True,
        )
        return ActorValueOperator(
            policy_operator=actor_module,
            value_operator=value_operator,
            common_operator=encoder_module,
        )

    @staticmethod
    def _log_parameter_counts(actor_value_wrapper, share_encoder: bool):
        """Logs the parameter counts of the policy components using a rich table."""
        console = Console()
        table = Table(title="Policy Parameter Counts")
        table.add_column("Component", justify="right", style="cyan", no_wrap=True)
        table.add_column("Parameters", justify="right", style="magenta")

        total_params = count_parameters(actor_value_wrapper)
        actor_params = count_parameters(actor_value_wrapper.get_policy_operator())
        value_params = count_parameters(actor_value_wrapper.get_value_operator())

        if share_encoder:
            # ActorValueOperator structure
            encoder_params = count_parameters(actor_value_wrapper.module[0])
            actor_head_params = count_parameters(actor_value_wrapper.module[1])
            value_head_params = count_parameters(actor_value_wrapper.module[2])
            table.add_row("Shared Encoder", f"{encoder_params:,}")
            table.add_row("Actor Head", f"{actor_head_params:,}")
            table.add_row("Value Head", f"{value_head_params:,}")
        else:
            # ActorCriticWrapper structure
            policy_operator = actor_value_wrapper.get_policy_operator()
            value_operator = actor_value_wrapper.get_value_operator()
            actor_encoder_params = count_parameters(policy_operator.module[0].module[0])
            actor_head_params = count_parameters(policy_operator.module[0].module[1])
            value_encoder_params = count_parameters(value_operator[0])
            value_head_params = count_parameters(value_operator[1])
            table.add_row("Actor Encoder", f"{actor_encoder_params:,}")
            table.add_row("Actor Head", f"{actor_head_params:,}")
            table.add_row("Value Encoder", f"{value_encoder_params:,}")
            table.add_row("Value Head", f"{value_head_params:,}")

        table.add_row("---", "---")
        table.add_row("Total Actor", f"{actor_params:,}")
        table.add_row("Total Value", f"{value_params:,}")
        table.add_row("---", "---")
        table.add_row("Total Parameters", f"{total_params:,}")
        console.print(table)

    @staticmethod
    def _apply_baselines_init(
        actor_operator: TensorDictModule,
        value_operator: TensorDictModule,
        action_size: int,
    ):
        """
        Apply the orthogonal initialization to the actor and value operators like in StableBaselines3.
        """

        def init_baselines(m):
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                output_dim = m.weight.shape[0]
                if output_dim == action_size * 2:
                    # Initialize the output layer of the policy network
                    nn.init.normal_(m.weight, 0, 0.01)
                elif output_dim == 1:
                    # Initialize the output layer of the value network
                    nn.init.normal_(m.weight, 0, 1)
                else:
                    # Initialize hidden layers
                    nn.init.orthogonal_(m.weight, gain=2**0.5)
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # initialize hidden layers orthogonally
        actor_operator.apply(init_baselines)
        value_operator.apply(init_baselines)
