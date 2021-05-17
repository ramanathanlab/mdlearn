"""Configurations and utilities for model building and training."""
import json
import yaml
import torch
import wandb
import argparse
from pathlib import Path
from pydantic import BaseSettings as _BaseSettings
from typing import TypeVar, Type, Union, Optional, Dict, Any

PathLike = Union[str, Path]
_T = TypeVar("_T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path: PathLike):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class WandbConfig(BaseSettings):
    # Project name for wandb logging
    wandb_project_name: Optional[str] = None
    # Team name for wandb logging
    wandb_entity_name: Optional[str] = None
    # Model tag for wandb labeling
    model_tag: Optional[str] = None

    def init(
        self,
        cfg: BaseSettings,
        model: torch.nn.Module,
        wandb_path: PathLike,
    ) -> Optional[wandb.config]:
        """Initialize wandb with model and config.

        Parameters
        ----------
        cfg : BaseSettings
            Model configuration with hyperparameters and training settings.
        model : torch.nn.Module
            Model to train, passed to :obj:`wandb.watch(model)` for logging.
        wandb_path : PathLike
            Path to write :obj:`wandb/` directory containing training logs.

        Returns
        -------
        Optional[wandb.config]
            wandb config object or None if :obj:`wandb_project_name` is None.
        """

        if self.wandb_project_name is not None:
            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity_name,
                name=self.model_tag,
                id=self.model_tag,
                dir=str(wandb_path),
                config=cfg.dict(),
                resume=False,
            )
            wandb.watch(model)
            return wandb.config


class OptimizerConfig(BaseSettings):
    """pydantic schema for PyTorch optimizer which allows for arbitrary
    optimizer hyperparameters."""

    class Config:
        extra = "allow"

    # Name of optimizer
    name: str = "Adam"
    # Arbitrary optimizer hyperparameters
    hparams: Dict[str, Any] = {}

    def get_torch_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Construct a PyTorch optimizer specified by :obj:`name` and :obj:`hparams`."""
        from torch import optim

        if self.name == "Adadelta":
            optimizer = optim.Adadelta
        elif self.name == "Adagrad":
            optimizer = optim.Adagrad
        elif self.name == "Adam":
            optimizer = optim.Adam
        elif self.name == "AdamW":
            optimizer = optim.AdamW
        elif self.name == "SparseAdam":
            optimizer = optim.SparseAdam
        elif self.name == "Adamax":
            optimizer = optim.Adamax
        elif self.name == "ASGD":
            optimizer = optim.ASGD
        elif self.name == "LBFGS":
            optimizer = optim.LBFGS
        elif self.name == "RMSprop":
            optimizer = optim.RMSprop
        elif self.name == "Rprop":
            optimizer = optim.Rprop
        elif self.name == "SGD":
            optimizer = optim.SGD
        else:
            raise ValueError(f"Invalid optimizer name: {self.name}")

        try:
            return optimizer(parameters, **self.hparams)

        except TypeError:
            raise Exception(
                f"Invalid parameter in hparams: {self.hparams}"
                f" for optimizer {self.name}.\nSee PyTorch docs."
            )


class SchedulerConfig(BaseSettings):
    """pydantic schema for PyTorch scheduler which allows for arbitrary
    scheduler hyperparameters."""

    class Config:
        extra = "allow"

    # Name of scheduler
    name: str = "ReduceLROnPlateau"
    # Arbitrary scheduler hyperparameters
    hparams: Dict[str, Any] = {}

    def get_torch_scheduler(self, optimizer: torch.optim.Optimizer):
        """Construct a PyTorch lr_scheduler specified by :obj:`name` and :obj:`hparams`."""
        from torch.optim import lr_scheduler

        if self.name == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau
        elif self.name == "LambdaLR":
            raise ValueError("LambdaLR not supported")
        elif self.name == "MultiplicativeLR":
            raise ValueError("MultiplicativeLR not supported")
        elif self.name == "StepLR":
            scheduler = lr_scheduler.StepLR
        elif self.name == "MultiStepLR":
            scheduler = lr_scheduler.MultiStepLR
        elif self.name == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR
        elif self.name == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR
        elif self.name == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau
        elif self.name == "CyclicLR":
            scheduler = lr_scheduler.CyclicLR
        elif self.name == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR
        elif self.name == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts
        else:
            raise ValueError(f"Invalid scheduler name: {self.name}")

        try:
            return scheduler(optimizer, **self.hparams)

        except TypeError:
            raise Exception(
                f"Invalid parameter in hparams: {self.hparams}"
                f" for scheduler {self.name}.\nSee PyTorch docs."
            )


def log_checkpoint(
    checkpoint_file: PathLike,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
):
    """Write a torch .pt file containing the epoch, model, optimizer,
    and scheduler.

    Parameters
    ----------
    checkpoint_file: PathLike
        Path to save checkpoint file.
    epoch : int
        The current training epoch.
    model : torch.nn.Module
        The model whose parameters are saved.
    optimizer : torch.optim.Optimizer
        The optimizer whose parameters are saved.
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Optional scheduler whose parameters are saved.
    """
    checkpoint = {
        "epoch": epoch,  # To resume training, (see resume_checkpoint)
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_file)


def resume_checkpoint(
    checkpoint_file: PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """Modifies :obj:`model`, :obj:`optimizer`, and :obj:`scheduler` with
    values stored in torch .pt file :obj:`checkpoint_file` to resume from
    a previous training checkpoint.

    Parameters
    ----------
    checkpoint_file : PathLike
        Path to checkpoint file to resume from.
    model : torch.nn.Module
        Module to update the parameters of.
    optimizer : torch.optim.Optimizer
        Optimizer to update.
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Optional scheduler to update.

    Returns
    -------
    int :
        The epoch the checkpoint is saved plus one i.e. the current
        training epoch to start from.
    """
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
    return start_epoch
