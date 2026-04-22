import csv
import logging
import os
import threading
import time
import sys
import hashlib
from dataclasses import dataclass, field
from itertools import groupby
from queue import Queue
from typing import Any
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from flipper_training import ROOT

PROJECT = os.environ.get("WANDB_PROJECT", "flipper_training")


def red(s):
    return f"\033[91m{s}\033[00m"


def green(s):
    return f"\033[92m{s}\033[00m"


def yellow(s):
    return f"\033[93m{s}\033[00m"


def blue(s):
    return f"\033[94m{s}\033[00m"


def bold_red(s):
    return f"\033[31;1m{s}\033[00m"


class ColoredFormatter(logging.Formatter):
    base_fmt = "%(asctime)s [%(name)s][%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"

    def __init__(self):
        super().__init__()
        self.formatters = {}
        for fun, level in zip([blue, green, yellow, red, bold_red], [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]):
            level_fmt = self.base_fmt.replace("%(levelname)s", fun("%(levelname)s"))
            self.formatters[level] = logging.Formatter(level_fmt)

    def format(self, record):
        formatter = self.formatters.get(record.levelno)
        return formatter.format(record)


def get_terminal_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    return logger


@dataclass
class RunLogger:
    train_config: DictConfig
    use_wandb: bool
    use_tensorboard: bool
    category: str
    logfiles: dict = field(default_factory=dict)
    writers: dict = field(default_factory=dict)
    log_queue: Queue = field(default_factory=Queue)
    step_metric_name: str = "log_step"
    known_wandb_metrics: set = field(default_factory=set)
    tensorboard_writer: SummaryWriter | None = field(init=False, default=None)  # Added placeholder

    def __post_init__(self):
        self.terminal_logger = get_terminal_logger("RunLogger")
        import os
        ts = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{os.getpid()}"
        run_name = f"{self.train_config['name']}_{ts}"

        # Under SLURM jobs, store everything in the SLURM log directory.
        # Otherwise fall back to runs/<category>/<run_name>.
        slurm_logpath = self._slurm_log_dir()
        if slurm_logpath is not None:
            self.logpath = slurm_logpath
        else:
            self.logpath = ROOT / f"runs/{self.category}/{run_name}"

        self.run_name = self.logpath.name  # use logdir name so W&B matches filesystem
        self.terminal_logger.info(f"RunLogger initialized for run {self.run_name} at {self.logpath}")
        self.logpath.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.logpath / "weights"
        self.weights_path.mkdir(exist_ok=True)
        if self.use_wandb:
            # Check if this is a restart by looking for saved W&B run ID
            wandb_id_file = self.logpath / ".wandb_run_id"
            is_restart = wandb_id_file.exists()
            if is_restart:
                with open(wandb_id_file, "r") as f:
                    self.wandb_run_id = f.read().strip()
                self.terminal_logger.info(f"Detected restart — resuming W&B run {self.wandb_run_id}")
                resume_mode = "allow"
            else:
                self.wandb_run_id = hashlib.sha256(self.run_name.encode()).hexdigest()
                with open(wandb_id_file, "w") as f:
                    f.write(self.wandb_run_id)
                resume_mode = "never"

            confdict = OmegaConf.to_container(self.train_config, resolve=False)
            wandb.init(
                project=PROJECT,
                id=self.wandb_run_id,
                name=self.run_name,
                tags=[self.category],
                notes=self.wandb_run_id,
                config=confdict,
                save_code=True,
                resume=resume_mode,
            )
            wandb.define_metric(self.step_metric_name)
        if self.use_tensorboard:
            tb_log_dir = self.logpath / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
            self.terminal_logger.info(f"TensorBoard logs will be saved to {tb_log_dir}")
        self._save_config()
        self.write_thread = threading.Thread(target=self._write, daemon=True)
        self.write_thread.start()

    @staticmethod
    def _slurm_log_dir() -> Path | None:
        """Return the SLURM log subdirectory for this job, or None if not in a SLURM job.

        Array jobs:  logs/<name>_<array_job_id>/<name>_<array_job_id>_<task_id>/
        Single jobs: logs/<name>_<job_id>/
        """
        job_name = os.environ.get("SLURM_JOB_NAME", "")
        array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        job_id = os.environ.get("SLURM_JOB_ID")
        if array_job_id and task_id:
            # Array job: logs/optuna_ftr_123/optuna_ftr_123_45/
            log_dir_name = f"{job_name}_{array_job_id}" if job_name else f"slurm_{array_job_id}"
            task_dir_name = f"{log_dir_name}_{task_id}"
            return ROOT / f"logs/{log_dir_name}/{task_dir_name}"
        elif job_id:
            # Single job: logs/train_ftr_456/
            log_dir_name = f"{job_name}_{job_id}" if job_name else f"slurm_{job_id}"
            return ROOT / f"logs/{log_dir_name}"
        return None

    def _save_config(self):
        OmegaConf.save(self.train_config, self.logpath / "config.yaml")

    def _init_logfile(self, name: str, sample_row: dict[str, Any]):
        self.logfiles[name] = open(self.logpath / f"{name}.csv", "w")
        writer = csv.DictWriter(self.logfiles[name], fieldnames=[self.step_metric_name] + list(sample_row.keys()))
        self.writers[name] = writer
        writer.writeheader()
        if self.use_wandb:
            for k in sample_row.keys():
                if k not in self.known_wandb_metrics:
                    wandb.define_metric(k, step_metric=self.step_metric_name)
                    self.known_wandb_metrics.add(k)
        return writer

    def log_data(self, row: dict[str, Any], step: int):
        self.log_queue.put((step, row))

    def _write_row(self, row: dict[str, Any], step: int):
        for topic, names in groupby(row.items(), key=lambda x: x[0].rsplit("/", maxsplit=1)[0]):
            topic_row = dict(names)
            writer = self.writers.get(topic, None) or self._init_logfile(topic, topic_row)
            # Handle new fields not in original fieldnames by extending the writer
            output_row = topic_row | {self.step_metric_name: step}
            if set(output_row.keys()) - set(writer.fieldnames):
                # New fields detected — extend fieldnames and rewrite header
                writer.fieldnames = list(dict.fromkeys(writer.fieldnames + list(output_row.keys())))
                self.logfiles[topic].seek(0)
                self.logfiles[topic].truncate()
                writer.writeheader()
                self.logfiles[topic].flush()
            writer.writerow(output_row)
            self.logfiles[topic].flush()

    def _write(self):
        while True:
            (step, row) = self.log_queue.get()
            if step == -1:
                break
            if self.use_wandb:
                wandb.log(data=row | {self.step_metric_name: step})
            if self.use_tensorboard and self.tensorboard_writer is not None:
                for tag, scalar_value in row.items():
                    # Basic check if value is likely a scalar
                    if isinstance(scalar_value, (int, float, torch.Tensor)) and not isinstance(
                        scalar_value, bool
                    ):  # check avoids bools which SummaryWriter dislikes
                        # Ensure tensor values are converted to scalar CPU values
                        if isinstance(scalar_value, torch.Tensor):
                            if scalar_value.numel() == 1:  # Log only if it's a single value tensor
                                scalar_value = scalar_value.item()
                            else:
                                continue  # Skip multi-element tensors for add_scalar
                        try:
                            self.tensorboard_writer.add_scalar(tag, scalar_value, step)
                        except Exception as e:
                            self.terminal_logger.warning(f"TensorBoard logging failed for tag '{tag}' with value {scalar_value}: {e}")
            self._write_row(row, step)

    def close(self):
        self.log_queue.put((-1, {}))
        self.write_thread.join()
        for f in self.logfiles.values():
            f.close()
        if self.use_wandb:
            wandb.finish()
            self.terminal_logger.info(
                f"W&B run finished (ID: {self.wandb_run_id}). If the job restarts, W&B logging will resume in the same run."
            )
        if self.use_tensorboard and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        self.terminal_logger.info("RunLogger closed.")

    def save_weights(self, state_dict: dict, name: str):
        model_path = self.weights_path / f"{name}.pth"
        torch.save(state_dict, model_path)
        if self.use_wandb:
            wandb.save(str(model_path), base_path=str(self.weights_path))

    def __del__(self):
        self.close()


@dataclass
class LocalRunReader:
    source: str | Path

    def __post_init__(self):
        self.path = Path(self.source).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        csvs = list(self.path.glob("*.csv"))
        print(f"Found available logs: {', '.join(map(str, csvs))}")

    def get_weights_path(self, name: str) -> Path:
        return self.path / "weights" / f"{name}.pth"

    def load_config(self) -> DictConfig:
        return OmegaConf.load(self.path / "config.yaml")


@dataclass
class WandbRunReader:
    run_id: str
    category: str
    default_weigths_path: Path = ROOT / "runs"

    def __post_init__(self):
        self.api = wandb.Api()
        self.run = self.api.run(f"{PROJECT}/{self.run_id}")
        self.history = self.run.scan_history()
        self.weights_root = self.default_weigths_path / f"{self.category}/{self.run.name}/wandb_weights"
        self.weights_root.mkdir(parents=True, exist_ok=True)

    def get_weights_path(self, name: str) -> Path:
        full_model_name = f"{PROJECT}/{name}:{self.run_id}"
        weight_artifact = self.api.artifact(full_model_name, type="model")
        weight_artifact.download(self.weights_root, skip_cache=False)
        return self.weights_root / f"{name}.pth"

    def load_config(self) -> DictConfig:
        return OmegaConf.create(self.run.config)

    def get_metric(self, name: str) -> list:
        return [x[name] for x in self.history]
