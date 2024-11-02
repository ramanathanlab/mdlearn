from __future__ import annotations

import random
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import nn

from mdlearn.utils import PathLike


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def conv_output_dim(input_dim, kernel_size, stride, padding, transpose=False):
    """
    Parameters
    ----------
    input_dim : int
        input size. may include padding
    kernel_size : int
        filter size
    stride : int
        stride length
    padding : int
        length of 0 pad
    """
    if transpose:
        # TODO: see symmetric decoder _conv_layers,
        #       may have bugs for transpose layers
        output_padding = 1 if stride > 1 else 0
        output_padding = 0
        return (
            (input_dim - 1) * stride
            + kernel_size
            - 2 * padding
            + output_padding
        )

    return (2 * padding + input_dim - kernel_size) // stride + 1


def conv_output_shape(
    input_dim,
    kernel_size,
    stride,
    padding,
    num_filters,
    transpose=False,
    dim=2,
):
    """
    Parameters
    ----------
    input_dim : tuple
        (height, width) dimensions for convolution input
    kernel_size : int
        filter size
    stride : int
        stride length
    padding : tuple
        (height, width) length of 0 pad
    num_filters : int
        number of filters
    transpose : bool
        signifies whether Conv or ConvTranspose
    dim : int
        1 or 2, signifies Conv1d or Conv2d

    Returns
    -------
    (channels, height, width) tuple
    """
    if isinstance(input_dim, int):
        input_dim, padding = [input_dim], [padding]

    dims = [
        conv_output_dim(d, kernel_size, stride, p, transpose)
        for d, p in zip(input_dim, padding)
    ]

    if dim == 1:
        return num_filters, dims[0]
    if dim == 2:
        return num_filters, dims[0], dims[1]

    raise ValueError(f'Invalid dim: {dim}')


def _same_padding(
    input_dim: Union[int, tuple[int, int]],
    kernel_size: int,
    stride: int,
) -> int:
    """
    Implements Keras-like same padding.
    If the stride is one then use same_padding.
    Otherwise, select the smallest pad such that the
    kernel_size fits evenly within the input_dim.
    """
    if stride == 1:
        # In this case we want output_dim = input_dim
        # input_dim = output_dim = (2*pad + input_dim - kernel_size) // stride + 1
        return (input_dim * (stride - 1) - stride + kernel_size) // 2

    # Largest i such that: alpha = kernel_size + i*stride <= input_dim
    # Then input_dim - alpha is the pad
    # i <= (input_dim - kernel_size) // stride
    for i in reversed(range((input_dim - kernel_size) // stride + 1)):
        alpha = kernel_size + i * stride
        if alpha <= input_dim:
            # TODO: see symmetric decoder
            # adjustment = int(input_dim % 2 == 0)
            return input_dim - alpha  # + adjustment

    raise Exception('No padding found')


def same_padding(
    input_dim: Union[int, tuple[int, int]],
    kernel_size: int,
    stride: int,
) -> Union[int, tuple[int, int]]:
    """Returns Keras-like same padding. Works for rectangular input_dim.

    Parameters
    ----------
    input_dim : tuple, int
        (height, width) dimensions for Conv2d input
        int for Conv1d input
    kernel_size : int
        filter size
    stride : int
        stride length

    Returns
    -------
    int:
        height of padding
    int:
        width of padding
    """
    # Handle Conv1d case
    if isinstance(input_dim, int):
        return _same_padding(input_dim, kernel_size, stride)

    h_pad = _same_padding(input_dim[0], kernel_size, stride)
    # If square input, no need to compute width padding
    if input_dim[0] == input_dim[1]:
        return h_pad, h_pad
    w_pad = _same_padding(input_dim[1], kernel_size, stride)
    return h_pad, w_pad


def get_activation(activation, *args, **kwargs):
    """
    Parameters
    ----------
    activation : str
        type of activation e.g. 'ReLU', etc
    """
    if activation == 'ReLU':
        return nn.ReLU(*args, **kwargs)
    if activation == 'LeakyReLU':
        return nn.LeakyReLU(*args, **kwargs)
    if activation == 'Sigmoid':
        return nn.Sigmoid(*args, **kwargs)
    if activation == 'Tanh':
        return nn.Tanh(*args, **kwargs)
    if activation == 'None':
        return nn.Identity(*args, **kwargs)
    raise ValueError(f'Invalid activation type: {activation}')


# TODO: generalize this more.
def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_uniform_(m.weight)


class Trainer:
    """Trainer base class which implements training utility functions."""

    def __init__(
        self,
        seed: int = np.random.default_rng().integers(2**31 - 1, dtype=int),
        in_gpu_memory: bool = False,
        num_data_workers: int = 0,
        prefetch_factor: int = 2,
        split_pct: float = 0.8,
        split_method: str = 'random',
        batch_size: int = 128,
        inference_batch_size: int = 128,
        shuffle: bool = True,
        device: str = 'cpu',
        epochs: int = 100,
        verbose: bool = False,
        clip_grad_max_norm: float = 10.0,
        checkpoint_log_every: int = 10,
        plot_log_every: int = 10,
        plot_n_samples: int = 10000,
        plot_method: Optional[str] = 'TSNE',
        train_subsample_pct: float = 1.0,
        valid_subsample_pct: float = 1.0,
        use_wandb: bool = False,
    ):
        """
        Parameters
        ----------
        seed : int, default=np.random.default_rng().integers(2**31 - 1, dtype=int)
            Random seed for torch, numpy, and random module.
        in_gpu_memory : bool, default=False
            If True, will pre-load the entire :obj:`data` array to GPU memory.
        num_data_workers : int, default=0
            How many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
        prefetch_factor : int, by default=2
            Number of samples loaded in advance by each worker. 2 means there will be a
            total of 2 * num_workers samples prefetched across all workers.
        split_pct : float, default=0.8
            Proportion of data set to use for training. The rest goes to validation.
        split_method : str, default="random"
            Method to split the data. For random split use "random", for a simple
            partition, use "partition".
        batch_size : int, default=128
            Mini-batch size for training.
        inference_batch_size : int, default=128
            Mini-batch size for inference.
        shuffle : bool, default=True
            Whether to shuffle training data or not.
        device : str, default="cpu"
            Specify training hardware either :obj:`cpu` or :obj:`cuda` for GPU devices.
        epochs : int, default=100
            Number of epochs to train for.
        verbose : bool, default False
            If True, will print training and validation loss at each epoch.
        clip_grad_max_norm : float, default=10.0
            Max norm of the gradients for gradient clipping for more information
            see: :obj:`torch.nn.utils.clip_grad_norm_` documentation.
        checkpoint_log_every : int, default=10
            Epoch interval to log a checkpoint file containing the model
            weights, optimizer, and scheduler parameters.
        plot_log_every : int, default=10
            Epoch interval to log a visualization plot of the latent space.
        plot_n_samples : int, default=10000
            Number of validation samples to use for plotting.
        plot_method : Optional[str], default="TSNE"
            The method for visualizing the latent space or if visualization
            should not be run, set :obj:`plot_method=None`. If using :obj:`"TSNE"`,
            it will attempt to use the RAPIDS.ai GPU implementation and
            will fallback to the sklearn CPU implementation if RAPIDS.ai
            is unavailable.
        train_subsample_pct : float, default=1.0
            Percentage of training data to use during hyperparameter sweeps.
        valid_subsample_pct : float, default=1.0
            Percentage of validation data to use during hyperparameter sweeps.
        use_wandb : bool, default=False
            If True, will log results to wandb.

        Raises
        ------
        ValueError
            :obj:`split_pct` should be between 0 and 1.
        ValueError
            :obj:`train_subsample_pct` should be between 0 and 1.
        ValueError
            :obj:`valid_subsample_pct` should be between 0 and 1.
        ValueError
            Specified :obj:`device` as :obj:`cuda`, but it is unavailable.

        Note
        ----
        This base class does not receive optimizer or scheduler settings
        because in general there could be multiple optimizers.
        """
        if split_pct < 0 or split_pct > 1:
            raise ValueError('split_pct should be between 0 and 1.')
        if train_subsample_pct < 0 or train_subsample_pct > 1:
            raise ValueError('train_subsample_pct should be between 0 and 1')
        if valid_subsample_pct < 0 or valid_subsample_pct > 1:
            raise ValueError('valid_subsample_pct should be between 0 and 1')
        if 'cuda' in device and not torch.cuda.is_available():
            raise ValueError('Specified cuda, but it is unavailable.')

        self.seed = seed
        self.scalar_dset_names = []
        self.in_gpu_memory = in_gpu_memory
        self.num_data_workers = 0 if in_gpu_memory else num_data_workers
        self.persistent_workers = (
            self.num_data_workers > 0
        ) and not self.in_gpu_memory
        self.prefetch_factor = prefetch_factor
        self.split_pct = split_pct
        self.split_method = split_method
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.epochs = epochs
        self.verbose = verbose
        self.clip_grad_max_norm = clip_grad_max_norm
        self.checkpoint_log_every = checkpoint_log_every
        self.plot_log_every = plot_log_every
        self.plot_n_samples = plot_n_samples
        self.plot_method = plot_method
        self.train_subsample_pct = train_subsample_pct
        self.valid_subsample_pct = valid_subsample_pct
        self.use_wandb = use_wandb

        # Default scheduler to None
        self.scheduler = None

        # Set random seeds
        self._set_seed()

    def _set_seed(self) -> None:
        """Set random seed of torch, numpy, and random."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _set_num_threads(self) -> None:
        """Set available number of cores."""
        torch.set_num_threads(
            1 if self.num_data_workers == 0 else self.num_data_workers,
        )

    def _make_output_dir(
        self,
        output_path: PathLike,
        exist_ok: bool = False,
    ) -> tuple[Path, Path, Path]:
        """Creates output directory structure.

        Parameters
        ----------
        output_path : PathLike
            The root output path to store training results.
        exist_ok : bool, default=False
            Set to True if resuming from a checkpoint, otherwise
            should be False for a fresh training run.

        Returns
        -------
        Path
            Root directory path for the training results.
        Path
            Directory path to store checkpoint files :obj:`output_path/checkpoints`.
        Path
            Directory path to store plotting results :obj:`output_path/plots`.
        """
        output_path = Path(output_path).resolve()
        output_path.mkdir(exist_ok=exist_ok)
        # Create checkpoint directory
        checkpoint_path = output_path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=exist_ok)
        # Create plot directory
        plot_path = output_path / 'plots'
        plot_path.mkdir(exist_ok=exist_ok)
        return output_path, checkpoint_path, plot_path

    def _load_checkpoint(self, checkpoint: PathLike) -> int:
        """Load parameters from a checkpoint file.

        Parameters
        ----------
        checkpoint : PathLike
            PyTorch checkpoint file (.pt) to load model, optimizer
            and scheduler parameters from.

        Returns
        -------
        int
            Epoch where training left off.

        Note
        ----
        This function works in the case of a single optimizer,
        single model and single (or None) scheduler. If additional
        functionality is needed, it can be implemented in the child class.
        """
        from mdlearn.utils import resume_checkpoint

        return resume_checkpoint(
            checkpoint,
            self.model,
            {'optimizer': self.optimizer},
            self.scheduler,
        )

    def _resume_training(self, checkpoint: Optional[PathLike] = None) -> int:
        """Optionally resume training from a checkpoint.

        Parameters
        ----------
        checkpoint : Optional[PathLike], default=None
            PyTorch checkpoint file (.pt) to resume training from.

        Returns
        -------
        int
            Epoch where training left off or 1 if :obj:`checkpoint` is :obj:`None`.

        Note
        ----
        Requires :obj:`self._load_checkpoint()` to be implemented.
        """
        if checkpoint is not None:
            start_epoch = self._load_checkpoint(checkpoint)
            if self.verbose:
                print(
                    f'Resume training at epoch {start_epoch} from {checkpoint}',
                )
        else:
            start_epoch = 1

        return start_epoch

    def step_scheduler(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_valid_loss: float,
    ):
        """Implements the logic to step the learning rate scheduler.
        Different schedulers may have different update logic. Please
        subclass :obj:`LinearAETrainer` and re-implement this function
        for support of additional logic.

        Parameters
        ----------
        epoch : int
            The current training epoch.
        avg_train_loss : float
            The current epochs average training loss.
        avg_valid_loss : float
            The current epochs average valiation loss.

        Raises
        ------
        NotImplementedError
            If using a learning rate scheduler other than :obj:`ReduceLROnPlateau`,
            a step function will need to be added.
        """
        if self.scheduler is None:
            return
        elif self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(avg_valid_loss)
        else:
            raise NotImplementedError(
                f'scheduler {self.scheduler_name} step function.',
            )

    def fit(self):
        """Trains the model on the input dataset.

        Raises
        ------
        NotImplementedError
            Child class must implement this method.
        """
        raise NotImplementedError('Child class must implement this method')

    def predict(self):
        """Predicts using the trained model.

        Raises
        ------
        NotImplementedError
            Child class must implement this method.
        """
        raise NotImplementedError('Child class must implement this method')
