import optuna
import traceback
from typing import Any
from dataclasses import dataclass
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback
from omegaconf import OmegaConf
import argparse
import os
from flipper_training import ROOT
from flipper_training.utils.logutils import get_terminal_logger
from flipper_training.experiments.ppo.train import PPOTrainer


DB_SECRET = OmegaConf.load(ROOT / "optuna_db.yaml")

def _build_conn_str(db_secret) -> str:
    if "url" in db_secret:
        conn_str = db_secret["url"]
        if conn_str.startswith("sqlite:///"):
            import pathlib
            pathlib.Path(conn_str[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        return conn_str
    sslmode = db_secret.get("sslmode", "require")
    return (
        f"postgresql+psycopg2://{db_secret['db_user']}:{db_secret['db_password']}"
        f"@{db_secret['db_host']}:{db_secret['db_port']}/{db_secret['db_name']}?sslmode={sslmode}"
    )
TERM_LOGGER = get_terminal_logger("optuna_train")


@dataclass
class OptunaConfig:
    study_name: str
    directions: list[str]
    metrics_to_optimize: list[str]
    num_trials: int
    gpu: int
    train_config_overrides: dict[str, Any]
    optuna_keys: list[str]
    optuna_types: list[str]
    optuna_values: list[str]
    slurm_params: dict[str, Any] | None = None


def define_search_space(trial, keys, types, values):
    params = {}
    for key, typ, val in zip(keys, types, values):
        if typ == "float":
            params[key] = trial.suggest_float(key, val[0], val[1])
        elif typ == "int":
            params[key] = trial.suggest_int(key, val[0], val[1])
        elif typ == "categorical":
            params[key] = trial.suggest_categorical(key, val)
        elif typ == "bool":
            params[key] = trial.suggest_categorical(key, [True, False])
    return params


class FailedTrialException(Exception):
    pass


def objective(trial, base_config, keys, types, values, metrics_to_optimize):
    params = define_search_space(trial, keys, types, values)
    dotlist = [f"{k}={v}" for k, v in params.items()]
    updated_config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))
    try:
        trainer = PPOTrainer(updated_config)  # Returns a dict like {"eval/mean_reward": value}
        metrics = trainer.train()
    except Exception as e:
        traceback.print_exception(e)
        raise FailedTrialException(f"Trial failed with exception: {e}") from e
    return tuple(metrics[metric] for metric in metrics_to_optimize)


def perform_study(optuna_config: OptunaConfig, train_config):
    # Set up Optuna study
    storage = RDBStorage(_build_conn_str(DB_SECRET))
    study = optuna.create_study(
        study_name=optuna_config.study_name,
        storage=storage,
        directions=optuna_config.directions,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(optuna_config.gpu)

    train_config["device"] = "cuda:0"
    callback = MaxTrialsCallback(optuna_config.num_trials)
    study.optimize(
        lambda trial: objective(
            trial,
            train_config,
            optuna_config.optuna_keys,
            optuna_config.optuna_types,
            optuna_config.optuna_values,
            optuna_config.metrics_to_optimize,
        ),
        callbacks=[callback],
        catch=[FailedTrialException],
        gc_after_trial=True,
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optuna optimization for PPO training")
    parser.add_argument("--train_config", "-t", type=str, required=True)
    parser.add_argument("--optuna_config", "-o", type=str, required=True)
    args, unknown = parser.parse_known_args()

    optuna_config = OmegaConf.merge(OmegaConf.load(args.optuna_config), OmegaConf.from_dotlist(unknown))
    optuna_config = OptunaConfig(**optuna_config)
    train_config = OmegaConf.load(args.train_config)
    train_config = OmegaConf.merge(train_config, optuna_config.train_config_overrides)

    # Validate argument lengths
    if len(optuna_config.optuna_keys) != len(optuna_config.optuna_types) or len(optuna_config.optuna_keys) != len(optuna_config.optuna_values):
        raise ValueError("optuna_keys, optuna_types, and optuna_values must have the same length")
    if len(optuna_config.directions) != len(optuna_config.metrics_to_optimize):
        raise ValueError("directions and metrics_to_optimize must have the same length")
    # Run
    TERM_LOGGER.info(f"Running with bare parameters: {optuna_config}")
    perform_study(optuna_config, train_config)


if __name__ == "__main__":
    main()
