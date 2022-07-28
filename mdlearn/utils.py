"""Configurations and utilities for model building and training."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import yaml
from pydantic import BaseSettings as _BaseSettings

PathLike = Union[str, Path]
_T = TypeVar("_T")


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
    >>> # MyConfig should inherit from BaseSettings
    >>> cfg = MyConfig.from_yaml(args.config)
    """
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
    ) -> Optional["wandb.config"]:  # noqa: F821
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
            import wandb

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
    """pydantic schema for PyTorch optimizer which allows
    for arbitrary optimizer hyperparameters."""

    class Config:
        extra = "allow"

    # Name of optimizer
    name: str = "Adam"
    # Arbitrary optimizer hyperparameters
    hparams: Dict[str, Any] = {}


class SchedulerConfig(BaseSettings):
    """pydantic schema for PyTorch scheduler which allows for arbitrary
    scheduler hyperparameters."""

    class Config:
        extra = "allow"

    # Name of scheduler
    name: str = "ReduceLROnPlateau"
    # Arbitrary scheduler hyperparameters
    hparams: Dict[str, Any] = {}


def get_torch_optimizer(
    name: str, hparams: Dict[str, Any], parameters
) -> torch.optim.Optimizer:
    """Construct a PyTorch optimizer specified by :obj:`name` and :obj:`hparams`."""
    from torch import optim

    if name == "Adadelta":
        optimizer = optim.Adadelta
    elif name == "Adagrad":
        optimizer = optim.Adagrad
    elif name == "Adam":
        optimizer = optim.Adam
    elif name == "AdamW":
        optimizer = optim.AdamW
    elif name == "SparseAdam":
        optimizer = optim.SparseAdam
    elif name == "Adamax":
        optimizer = optim.Adamax
    elif name == "ASGD":
        optimizer = optim.ASGD
    elif name == "LBFGS":
        optimizer = optim.LBFGS
    elif name == "RMSprop":
        optimizer = optim.RMSprop
    elif name == "Rprop":
        optimizer = optim.Rprop
    elif name == "SGD":
        optimizer = optim.SGD
    else:
        raise ValueError(f"Invalid optimizer name: {name}")

    try:
        return optimizer(parameters, **hparams)

    except TypeError:
        raise Exception(
            f"Invalid parameter in hparams: {hparams}"
            f" for optimizer {name}.\nSee PyTorch docs."
        )


def get_torch_scheduler(  # noqa: C901
    name: Optional[str], hparams: Dict[str, Any], optimizer: torch.optim.Optimizer
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

    if name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau
    elif name == "LambdaLR":
        raise ValueError("LambdaLR not supported")
    elif name == "MultiplicativeLR":
        raise ValueError("MultiplicativeLR not supported")
    elif name == "StepLR":
        scheduler = lr_scheduler.StepLR
    elif name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR
    elif name == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR
    elif name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR
    elif name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau
    elif name == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR
    elif name == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR
    elif name == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts
    else:
        raise ValueError(f"Invalid scheduler name: {name}")

    try:
        return scheduler(optimizer, **hparams)

    except TypeError:
        raise Exception(
            f"Invalid parameter in hparams: {hparams}"
            f" for scheduler {name}.\nSee PyTorch docs."
        )


def log_checkpoint(
    checkpoint_file: PathLike,
    epoch: int,
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
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
        "epoch": epoch,  # To resume training, (see resume_checkpoint)
        "model_state_dict": model.state_dict(),
    }
    for name, optimizer in optimizers.items():
        checkpoint[name + "_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_file)


def resume_checkpoint(
    checkpoint_file: PathLike,
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
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
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[name + "_state_dict"])
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)
    return start_epoch


def plot_scatter(
    data: np.ndarray,
    color_dict: Dict[str, np.ndarray] = {},
    color: Optional[str] = None,
):

    import pandas as pd
    import plotly.express as px

    df_dict = color_dict.copy()

    dim = data.shape[1]
    assert dim in [2, 3]
    for i, name in zip(range(dim), ["x", "y", "z"]):
        df_dict[name] = data[:, i]

    df = pd.DataFrame(df_dict)
    scatter_kwargs = dict(
        x="x",
        y="y",
        color=color,
        width=1000,
        height=1000,
        size_max=7,
        hover_data=list(df_dict.keys()),
    )
    if dim == 2:
        fig = px.scatter(df, **scatter_kwargs)
    else:  # dim == 3
        fig = px.scatter_3d(df, z="z", **scatter_kwargs)
    return fig


def log_latent_visualization(
    data: np.ndarray,
    colors: Dict[str, np.ndarray],
    output_path: PathLike,
    epoch: int = 0,
    n_samples: Optional[int] = None,
    method: str = "TSNE",
) -> Dict[str, str]:
    """Make scatter plots of the latent space using the specified
    method of dimensionality reduction.

    Parameters
    ----------
    data : np.ndarray
        The latent embeddings to visualize of shape (N, D) where
        N  is the number of examples and D is the number of dimensions.
    colors : Dict[str, np.ndarray]
        Each item in the dictionary will generate a different plot labeled
        with the key name. Each inner array should be of size N.
    output_path : PathLike
        The output directory path to save plots to.
    epoch : int, default=0
        The current epoch of training to label plots with.
    n_samples : Optional[int], default=None
        Number of samples to plot, will take a random sample of the
        :obj:`data` if :obj:`n_samples < N`. Otherwise, if :obj:`n_samples`
        is None, use all the data.
    method : str, default="TSNE"
        Method of dimensionality reduction used to plot. Currently supports:
        "PCA", "TSNE", "LLE", or "raw" for plotting the raw embeddings (or
        up to the first 3 dimensions if D > 3). If "TSNE" is specified, then
        the GPU accelerated RAPIDS.ai implementation will be tried first and
        if it is unavailable then the sklearn version will be used instead.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping each key in color to a raw HTML string containing
        the scatter plot data. These can be saved directly for visualization
        and logged to wandb during training.

    Raises
    ------
    ValueError
        If dimensionality reduction :obj:`method` is not supported.
    """
    from plotly.io import to_html

    # Make temp variables to not mutate input data
    if n_samples is not None:
        inds = np.random.choice(len(data), n_samples)
        _data = data[inds]
        _colors = {name: color[inds] for name, color in colors.items()}
    else:
        _data = data
        _colors = colors

    if method == "PCA":
        from sklearn.decomposition import PCA

        model = PCA(n_components=3)
        data_proj = model.fit_transform(_data)

    elif method == "TSNE":
        try:
            # Attempt to use rapidsai
            from cuml.manifold import TSNE

            # rapidsai only supports 2 dimensions
            model = TSNE(n_components=2, method="barnes_hut")
        except ImportError:
            from sklearn.manifold import TSNE

            model = TSNE(n_components=3, n_jobs=1)

        data_proj = model.fit_transform(_data)

    elif method == "LLE":
        from sklearn import manifold

        data_proj, _ = manifold.locally_linear_embedding(
            _data, n_neighbors=12, n_components=3
        )
    elif method == "raw":
        if _data.shape[1] <= 3:
            # If _data only has 2 or 3 dimensions, use it directly.
            data_proj = _data
        else:
            # Use the first 3 dimensions of the raw data.
            data_proj = _data[:, :3]
    else:
        raise ValueError(f"Invalid dimensionality reduction method {method}")

    html_strings = {}
    for color in _colors:
        fig = plot_scatter(data_proj, _colors, color)
        html_string = to_html(fig)
        html_strings[color] = html_string

        fname = Path(output_path) / f"latent_space-{method}-{color}-epoch-{epoch}.html"
        with open(fname, "w") as f:
            f.write(html_string)

    return html_strings
