from __future__ import annotations

from pathlib import Path
from typing import Optional

from mdlearn.utils import BaseModel
from mdlearn.utils import OptimizerConfig
from mdlearn.utils import SchedulerConfig


class SymmetricConv2dWAEConfig(BaseModel):
    # Path to HDF5 training file
    input_path: Path = Path('TODO')
    # Path to write sweep to
    output_path: Path = Path('TODO')
    # Input image shapes
    input_shape: tuple[int, ...] = (1, 28, 28)
    # Name of the dataset in the HDF5 file.
    dataset_name: str = 'contact_map'
    # Name of scalar datasets in the HDF5 file.
    scalar_dset_names: list[str] = []
    # Name of optional values field in the HDF5 file.
    values_dset_name: Optional[str] = None
    # Sets requires_grad torch.Tensor parameter for scalars specified
    # by scalar_dset_names. Set to True, to use scalars for multi-task
    # learning. If scalars are only required for plotting, then set it as False.
    scalar_requires_grad: bool = False
    # Whether to pull all the training data into memory or read each
    # batch from disk on the fly
    in_memory: bool = True
    # Percentage of data to be used as training data after a random split.
    split_pct: float = 0.8
    # Random seed for shuffling train/validation data
    seed: int = 333
    # Whether or not to shuffle train/validation data
    shuffle: bool = True
    # Number of epochs to train
    epochs: int = 10
    # Training batch size
    batch_size: int = 64
    # Gradient clipping (max_norm parameter of torch.nn.utils.clip_grad_norm_)
    clip_grad_max_norm: float = 5.0
    # Pretrained model weights
    init_weights: Optional[str] = None
    # Optimizer params
    optimizer: OptimizerConfig = OptimizerConfig(
        name='Adam',
        hparams={'lr': 0.0001},
    )
    # Learning rate scheduler params
    scheduler: Optional[SchedulerConfig] = None

    # Model hyperparameters
    latent_dim: int = 64
    filters: list[int] = [100, 100, 100, 100]
    kernels: list[int] = [5, 5, 5, 5]
    strides: list[int] = [1, 2, 1, 1]
    latent_dim: int = 10
    affine_widths: list[int] = [64]
    affine_dropouts: list[float] = [0.0]
    activation: str = 'ReLU'
    output_activation: str = 'Sigmoid'  # None is Identity function
    lambda_rec: float = 1.0
    # Regurization parameters
    # Should be ~ sqrt(latent_dim)
    sigma: float = 3.0  # Set by code
    kernel: str = 'gaussian'
    rf_dim: int = 10
    rf_resample: bool = False

    # Training settings
    # Number of data loaders for training
    num_data_workers: int = 0
    # Number of samples loaded in advance by each worker
    prefetch_factor: int = 2
    # Whether or not to ignore the GPU while training.
    ignore_gpu: bool = False
    # Log checkpoint file every `checkpoint_log_every` epochs
    checkpoint_log_every: int = 1
    # Log latent space plot `plot_log_every` epochs
    plot_log_every: int = 1

    # Validation settings
    # Method used to visualize latent space
    plot_method: str = 'TSNE'
    # Number of validation samples to run visualization on
    plot_n_samples: Optional[int] = None

    # Hyperparameter sweep optimizations
    # Percent of training batches to use (batches are shuffled each epoch)
    train_subsample_pct: float = 1.0
    # Percent of validation batches to use (batches are shuffled each epoch)
    valid_subsample_pct: float = 1.0


if __name__ == '__main__':
    SymmetricConv2dWAEConfig().dump_yaml(
        'symmetric_conv2d_wae_sweep_default.yaml',
    )
