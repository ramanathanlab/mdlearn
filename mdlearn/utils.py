"""Configurations and utilities for model building and training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

import torch
import yaml
from pydantic import BaseModel as _BaseModel

PathLike = Union[str, Path]
_T = TypeVar('_T')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments using argparse library

    Returns
    -------
        argparse.Namespace:
            Dict like object containing a path to a YAML file
            accessed via the config property.

    Example
    -------
    >>> from mdlearn.utils import parse_args
    >>> args = parse_args()
    >>> # MyConfig should inherit from BaseModel
    >>> cfg = MyConfig.from_yaml(args.config)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        help='YAML config file',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


class BaseModel(_BaseModel):
    def dump_yaml(self, cfg_path: PathLike):
        with open(cfg_path, mode='w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class WandbConfig(BaseModel):
    # Project name for wandb logging
    wandb_project_name: Optional[str] = None
    # Team name for wandb logging
    wandb_entity_name: Optional[str] = None
    # Model tag for wandb labeling
    wandb_model_tag: Optional[str] = None

    def init(
        self,
        cfg: BaseModel,
        model: torch.nn.Module,
        wandb_path: PathLike,
    ) -> Optional[wandb.config]:  # noqa: F821
        """Initialize wandb with model and config.

        Parameters
        ----------
        cfg : BaseModel
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
            import wandb

            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity_name,
                name=self.wandb_model_tag,
                id=self.wandb_model_tag,
                dir=str(wandb_path),
                config=cfg.dict(),
                resume=False,
            )
            wandb.watch(model)
            return wandb.config


class OptimizerConfig(BaseModel):
    """pydantic schema for PyTorch optimizer which allows
    for arbitrary optimizer hyperparameters.
    """

    class Config:
        extra = 'allow'

    # Name of optimizer
    name: str = 'Adam'
    # Arbitrary optimizer hyperparameters
    hparams: dict[str, Any] = {}


class SchedulerConfig(BaseModel):
    """pydantic schema for PyTorch scheduler which allows for arbitrary
    scheduler hyperparameters.
    """

    class Config:
        extra = 'allow'

    # Name of scheduler
    name: str = 'ReduceLROnPlateau'
    # Arbitrary scheduler hyperparameters
    hparams: dict[str, Any] = {}


def get_torch_optimizer(
    name: str,
    hparams: dict[str, Any],
    parameters,
) -> torch.optim.Optimizer:
    """Construct a PyTorch optimizer specified by :obj:`name` and :obj:`hparams`."""
    from torch import optim

    if name == 'Adadelta':
        optimizer = optim.Adadelta
    elif name == 'Adagrad':
        optimizer = optim.Adagrad
    elif name == 'Adam':
        optimizer = optim.Adam
    elif name == 'AdamW':
        optimizer = optim.AdamW
    elif name == 'SparseAdam':
        optimizer = optim.SparseAdam
    elif name == 'Adamax':
        optimizer = optim.Adamax
    elif name == 'ASGD':
        optimizer = optim.ASGD
    elif name == 'LBFGS':
        optimizer = optim.LBFGS
    elif name == 'RMSprop':
        optimizer = optim.RMSprop
    elif name == 'Rprop':
        optimizer = optim.Rprop
    elif name == 'SGD':
        optimizer = optim.SGD
    else:
        raise ValueError(f'Invalid optimizer name: {name}')

    try:
        return optimizer(parameters, **hparams)

    except TypeError:
        raise Exception(
            f'Invalid parameter in hparams: {hparams}'
            f' for optimizer {name}.\nSee PyTorch docs.',
        )


def get_torch_scheduler(  # noqa: C901
    name: Optional[str],
    hparams: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Construct a PyTorch lr_scheduler specified by :obj:`name` and :obj:`hparams`.

    Parameters
    ----------
    name : Optional[str]
        Name of PyTorch lr_scheduler class to use. If :obj:`name` is :obj:`None`,
        simply return None.
    hparams : Dict[str, Any]
        Hyperparameters to pass to the lr_scheduler.
    optimizer : torch.optim.Optimizer
        The initialized optimizer.

    Returns
    -------
    Optional[torch.optim.lr_scheduler._LRScheduler]
        The initialized PyTorch scheduler, or None if :obj:`name` is :obj:`None`.
    """
    if name is None:
        return None

    from torch.optim import lr_scheduler

    if name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau
    elif name == 'LambdaLR':
        raise ValueError('LambdaLR not supported')
    elif name == 'MultiplicativeLR':
        raise ValueError('MultiplicativeLR not supported')
    elif name == 'StepLR':
        scheduler = lr_scheduler.StepLR
    elif name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR
    elif name == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR
    elif name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR
    elif name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau
    elif name == 'CyclicLR':
        scheduler = lr_scheduler.CyclicLR
    elif name == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts
    else:
        raise ValueError(f'Invalid scheduler name: {name}')

    try:
        return scheduler(optimizer, **hparams)

    except TypeError:
        raise Exception(
            f'Invalid parameter in hparams: {hparams}'
            f' for scheduler {name}.\nSee PyTorch docs.',
        )


def log_checkpoint(
    checkpoint_file: PathLike,
    epoch: int,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
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
    optimizers : Dict[str, torch.optim.Optimizer]
        The optimizers whose parameters are saved.
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Optional scheduler whose parameters are saved.
    """
    checkpoint = {
        'epoch': epoch,  # To resume training, (see resume_checkpoint)
        'model_state_dict': model.state_dict(),
    }
    for name, optimizer in optimizers.items():
        checkpoint[name + '_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_file)


def resume_checkpoint(
    checkpoint_file: PathLike,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
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
    optimizers : Dict[str, torch.optim.Optimizer]
        Optimizers to update.
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Optional scheduler to update.

    Returns
    -------
    int :
        The epoch the checkpoint is saved plus one i.e. the current
        training epoch to start from.
    """
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[name + '_state_dict'])
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
    return start_epoch
