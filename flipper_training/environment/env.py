import time
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, assert_allclose_td
from torchrl.data import Bounded, Composite, Unbounded, Binary
from torchrl.envs import EnvBase, make_composite_from_td

from flipper_training.configs.terrain_config import TerrainConfig
from flipper_training.configs.engine_config import PhysicsEngineConfig
from flipper_training.configs.robot_config import RobotModelConfig
from flipper_training.engine.engine import DPhysicsEngine
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.logutils import get_terminal_logger
from flipper_training.vis.static_vis import plot_heightmap_3d
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from flipper_training.observations import ObservationFactory
    from flipper_training.rl_objectives import ObjectiveFactory
    from flipper_training.rl_rewards import RewardFactory


class Env(EnvBase):
    _batch_locked = True
    STATE_KEY = "curr_state"
    PREV_STATE_DER_KEY = "prev_state_der"

    def __init__(
        self,
        objective_factory: "ObjectiveFactory",
        reward_factory: "RewardFactory",
        observation_factories: list["ObservationFactory"],
        terrain_config: TerrainConfig,
        physics_config: PhysicsEngineConfig,
        robot_model_config: RobotModelConfig,
        device: torch.device | str = "cpu",
        out_dtype: torch.dtype = torch.float32,
        differentiable: bool = False,
        engine_compile_opts: dict | None = None,
        return_derivative: bool = False,
        engine_iters_per_step: int = 1,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Misc
        self.n_robots = self.batch_size[0]
        self.differentiable = differentiable
        self.out_dtype = out_dtype
        self.logger = get_terminal_logger("environment")
        self.return_derivative = return_derivative
        self.engine_iters_per_step = engine_iters_per_step
        self.generator = generator
        # Physics configs
        self.phys_cfg = physics_config.to(device)
        self.robot_cfg = robot_model_config.to(device)
        self.terrain_cfg = terrain_config.to(device)
        # Engine
        self.engine = DPhysicsEngine(physics_config, robot_model_config, device)
        # RL components
        self.objective = objective_factory(self)
        self.reward = reward_factory(self)
        self.observations = [o(self) for o in observation_factories]
        # RL State variables
        self.step_count = torch.zeros((self.n_robots,), device=self.device, dtype=torch.int32)
        self.step_limits = torch.zeros((self.n_robots,), device=self.device, dtype=torch.int32)
        self.start = PhysicsState.dummy(batch_size=self.n_robots, robot_model=robot_model_config, device=self.device)
        self.goal = PhysicsState.dummy(batch_size=self.n_robots, robot_model=robot_model_config, device=self.device)
        self.truncate_mode = True
        # Specs
        self.action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()
        self.reward_spec = self._make_reward_spec()
        self.done_spec = self._make_done_spec()
        # Reset the environment
        if engine_compile_opts is not None:
            self._compile_engine(**engine_compile_opts)
        self.reset()
        self._print_summary()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["step_count"] = self.step_count
        state_dict["step_limits"] = self.step_limits
        state_dict["start"] = self.start
        state_dict["goal"] = self.goal
        state_dict["objective"] = self.objective.state_dict()
        state_dict["reward"] = self.reward.state_dict()
        return state_dict

    def _set_truncate_mode(self, truncate_mode: bool) -> None:
        self.truncate_mode = truncate_mode

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.step_count = state_dict["step_count"]
        self.step_limits = state_dict["step_limits"]
        self.start = state_dict["start"]
        self.goal = state_dict["goal"]
        self.objective.load_state_dict(state_dict["objective"])
        self.reward.load_state_dict(state_dict["reward"])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def _print_summary(self) -> None:
        t = Table(title="Environment Summary")
        t.add_column("Key", justify="right", style="cyan", no_wrap=True)
        t.add_column("Value", justify="right", style="magenta")
        t.add_row("Number of robots", str(self.n_robots))
        t.add_row("Observations", ", ".join([o.name for o in self.observations]))
        t.add_row("Reward", str(self.reward.name))
        t.add_row("Objective", str(self.objective.name))
        t.add_row("Physics frequency", f"{1 / self.phys_cfg.dt: .2f} Hz")
        t.add_row("Engine iters/step", str(self.engine_iters_per_step))
        t.add_row("Effective frequency", f"{1 / self.effective_dt: .2f} Hz")
        c = Console()
        c.print(t)

    @property
    def effective_dt(self) -> float:
        return self.engine_iters_per_step * self.phys_cfg.dt

    def _set_differentiable(self, differentiable: bool) -> None:
        self.differentiable = differentiable

    def _compile_engine(
        self,
        correctness_iters: int = 100,
        benchmark_iters: int = 1000,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        **kwargs,
    ) -> None:
        act = self.action_spec.rand()
        state = self.start.clone()  # Dummy state
        # Capture the return tensors from the engine for correctness
        states, prev_state_ders = [], []
        for _ in range(correctness_iters):
            state, prev_state_der = self.engine(state, act, self.terrain_cfg)
            states.append(state)
            prev_state_ders.append(prev_state_der)
        # Compile the engine
        torch._dynamo.reset()
        self.engine.compile(**kwargs)
        self.engine(state, act, self.terrain_cfg)  # Dummy forward pass to compile the engine, record the return tensors
        state = self.start.clone()  # Reset the state
        self.logger.info(f"Engine compiled successfully, testing correctness with {atol=}, {rtol=}")
        for _ in range(correctness_iters):
            curr_state, prev_state_der = self.engine(state, act, self.terrain_cfg)
            # Check correctness
            assert_allclose_td(curr_state, states.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state")
            assert_allclose_td(prev_state_der, prev_state_ders.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state der")
            state = curr_state.clone()
        self.logger.info("Compiled engine passed correctness test")
        # Benchmark the compiled engine
        start_time = time.perf_counter_ns()
        for _ in range(benchmark_iters):
            _ = self.engine(state, act, self.terrain_cfg)
        end_time = time.perf_counter_ns()
        self.logger.info(f"Compiled engine takes {((end_time - start_time) / benchmark_iters) / 1e6} ms per step")

    def _set_seed(self, seed: int | None):
        self.logger.warning(f"This environment is not seedable, ignoring seed {seed}, please set the default generator seed")

    def _make_action_spec(self) -> Composite:
        track_low = torch.full(
            (self.n_robots, self.robot_cfg.num_driving_parts),
            -self.robot_cfg.v_max,
            device=self.device,
        )
        track_high = -track_low
        joint_low = self.robot_cfg.joint_limits[0].repeat(self.n_robots, 1)
        joint_high = self.robot_cfg.joint_limits[1].repeat(self.n_robots, 1)
        return Bounded(
            low=torch.cat([track_low, joint_low], dim=1),
            high=torch.cat([track_high, joint_high], dim=1),
            shape=(self.n_robots, 2 * self.robot_cfg.num_driving_parts),
            device=self.device,
            dtype=torch.float32,
        )

    def visualize(self, robot_points: torch.Tensor | None = None, return_figures: bool = False) -> None | list:
        figures = []
        for i in range(self.n_robots):
            f = plot_heightmap_3d(
                self.terrain_cfg.x_grid[i],
                self.terrain_cfg.y_grid[i],
                self.terrain_cfg.z_grid[i],
                start=self.start.x[i],
                goal=self.goal.x[i],
                robot_points=robot_points[i] if robot_points is not None else None,
            )
            if return_figures:
                figures.append(f)
            else:
                f.show()
        if return_figures:
            return figures

    def _make_observation_spec(self) -> Composite:
        obs_specs = {o.name: o.get_spec() for o in self.observations}
        state_spec = {Env.STATE_KEY: make_composite_from_td(self.start.to_tensordict())}
        if self.return_derivative:
            der_spec = {
                Env.PREV_STATE_DER_KEY: make_composite_from_td(
                    PhysicsStateDer.dummy(self.robot_cfg, device=self.device, batch_size=self.n_robots).to_tensordict()
                )
            }
        else:
            der_spec = {}
        return Composite(
            obs_specs | state_spec | der_spec,  # Include the physics state in the observation spec
            device=self.device,
            shape=(self.n_robots,),
        )

    def _make_reward_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.n_robots, 1),
            dtype=self.out_dtype,
            device=self.device,
        )

    def _make_done_spec(self) -> Bounded:
        bool_spec = Binary(
            shape=(self.n_robots, 1),
            dtype=torch.bool,
            device=self.device,
        )
        return Composite(
            {
                "succeeded": bool_spec,
                "failed": bool_spec,
                "truncated": bool_spec,
                "terminated": bool_spec,
            },
            device=self.device,
            shape=(self.n_robots,),
        )

    def _get_observations(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> TensorDict:
        obs_td = TensorDict(
            {o.name: o(prev_state=prev_state, action=action, prev_state_der=prev_state_der, curr_state=curr_state) for o in self.observations},
            device=self.device,
            batch_size=[self.n_robots],
        )
        obs_td[Env.STATE_KEY] = curr_state.to_tensordict()
        if self.return_derivative:
            obs_td[Env.PREV_STATE_DER_KEY] = prev_state_der.to_tensordict()
        return obs_td

    def _to_realistic_env(self) -> "Env":
        """
        Convert this environment to a realistic world interface environment.
        This is done by replacing the _step method to instead take in a tensordict with the necessary
        observation data from the realistic world, and outputting a tensordict in the same format
        as this environment such that it can be used with the trained policy.
        Returns:
            Env: A new environment instance that interfaces with a realistic world.
        """

        def realistic_step(self, tensordict: TensorDict) -> TensorDict:
            obs_td = TensorDict(
                {o.name: o.from_realistic_world(tensordict) for o in self.observations},
                device=self.device,
                batch_size=[1],
            )
            obs_td[Env.STATE_KEY] = PhysicsState.dummy(self.robot_cfg, device=self.device, batch_size=1).to_tensordict()
            obs_td["succeeded"] = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
            obs_td["failed"] = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
            obs_td["terminated"] = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
            obs_td["truncated"] = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
            obs_td["reward"] = torch.zeros((1, 1), device=self.device, dtype=self.out_dtype)
            return obs_td

        self._step = realistic_step.__get__(self, Env)
        return self

    def _step_engine(self, prev_state: PhysicsState, action: torch.Tensor) -> tuple[PhysicsStateDer, PhysicsState]:
        curr_state = prev_state
        first_prev_state_der = None
        with torch.inference_mode(not self.differentiable):
            for _ in range(self.engine_iters_per_step):
                curr_state, prev_state_der = self.engine(curr_state, action, self.terrain_cfg)
                if first_prev_state_der is None:
                    first_prev_state_der = prev_state_der.clone()
                curr_state = curr_state.clone()
        return first_prev_state_der, curr_state

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        # Generate start and goal states, iteration limits for done/terminated robots
        if tensordict is not None and "_reset" in tensordict:  # this is passed in training
            reset_mask = tensordict["_reset"].squeeze(-1)
        else:
            reset_mask = torch.full((self.n_robots,), True, device=self.device, dtype=torch.bool)
        training = tensordict is not None and "_step" in tensordict
        self.objective.reset(reset_mask, training)
        self.reward.reset(reset_mask, training)
        # Generate new start and goal states
        new_start, new_goal, new_step_limits = self.objective.generate_start_goal_states()
        # Update the state variables for the done robots
        self.start[reset_mask] = new_start[reset_mask]
        self.goal[reset_mask] = new_goal[reset_mask]
        self.step_limits[reset_mask] = new_step_limits[reset_mask]
        self.step_count[reset_mask] = 0
        # Take a dummy step to get the first observation
        if tensordict is not None and Env.STATE_KEY in tensordict:
            prev_state = PhysicsState(**tensordict.get(Env.STATE_KEY))
        else:
            prev_state = self.start.clone()
        zeros_action = self.action_spec.zeros()
        prev_state_der, curr_state = self._step_engine(prev_state, zeros_action)
        # Output tensordict
        obs_td = self._get_observations(prev_state=prev_state, action=zeros_action, prev_state_der=prev_state_der, curr_state=curr_state)
        return obs_td

    def _step(self, tensordict) -> TensorDict:
        self.step_count += 1
        action = tensordict.get("action").to(self.device)
        prev_state = PhysicsState.from_tensordict(tensordict.get(Env.STATE_KEY))
        # Step the engine
        prev_state_der, curr_state = self._step_engine(prev_state=prev_state, action=action)
        # Check if the robots have reached the goal or terminated
        reached_goal = self.objective.check_reached_goal(prev_state, curr_state, self.goal)
        failed = self.objective.check_terminated_wrong(prev_state, curr_state, self.goal)
        truncated = (self.step_count >= self.step_limits) * self.truncate_mode  # All zeros if truncate_mode is False
        # Output tensordict
        obs_td = self._get_observations(prev_state=prev_state, action=action, prev_state_der=prev_state_der, curr_state=curr_state)
        obs_td["succeeded"] = reached_goal
        obs_td["failed"] = failed
        obs_td["terminated"] = failed | reached_goal
        obs_td["truncated"] = truncated
        obs_td["reward"] = self.reward(
            prev_state=prev_state,
            action=action,
            prev_state_der=prev_state_der,
            curr_state=curr_state,
            success=reached_goal,
            fail=failed,
            start_state=self.start,
            goal_state=self.goal,
        )
        return obs_td
