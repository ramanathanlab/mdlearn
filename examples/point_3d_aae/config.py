from __future__ import annotations

from pathlib import Path
from typing import Optional

from mdlearn.utils import BaseModel
from mdlearn.utils import OptimizerConfig
from mdlearn.utils import SchedulerConfig
from mdlearn.utils import WandbConfig


class Point3dAAEConfig(BaseModel):
    # File paths
    # Path to HDF5 training file
    input_path: Path = Path('TODO')
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path('TODO')
    # Optionally resume training from a checkpoint file
    resume_checkpoint: Optional[Path] = None

    # Number of points per sample. Should be smaller or equal
    # than the total number of points.
    num_points: int = 28
    # Number of additional per-point features in addition to xyz coords.
    num_features: int = 0
    # Name of the dataset in the HDF5 file.
    dataset_name: str = 'point_cloud'
    # Name of scalar datasets in the HDF5 file.
    scalar_dset_names: list[str] = []
    # If True, subtract center of mass from batch and shift and scale
    # batch by the full dataset statistics.
    cms_transform: bool = False
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
    # Pretrained model weights
    init_weights: Optional[str] = None
    # AE (encoder/decoder) optimizer params
    ae_optimizer: OptimizerConfig = OptimizerConfig(
        name='Adam',
        hparams={'lr': 0.0001},
    )
    # Discriminator optimizer params
    disc_optimizer: OptimizerConfig = OptimizerConfig(
        name='Adam',
        hparams={'lr': 0.0001},
    )
    # Learning rate scheduler params
    scheduler: Optional[SchedulerConfig] = None
    # Wandb params
    wandb: Optional[WandbConfig] = WandbConfig()

    # Model hyperparameters
    latent_dim: int = 20
    encoder_bias: bool = True
    encoder_relu_slope: float = 0.0
    encoder_filters: list[int] = [64, 128, 256, 256, 512]
    encoder_kernels: list[int] = [5, 5, 3, 1, 1]
    decoder_bias: bool = True
    decoder_relu_slope: float = 0.0
    decoder_affine_widths: list[int] = [64, 128, 512, 1024]
    discriminator_bias: bool = True
    discriminator_relu_slope: float = 0.0
    discriminator_affine_widths: list[int] = [512, 128, 64]
    # Mean of the prior distribution
    noise_mu: float = 0.0
    # Standard deviation of the prior distribution
    noise_std: float = 1.0
    # Releative weight to put on gradient penalty
    lambda_gp: float = 10.0
    # Releative weight to put on reconstruction loss
    lambda_rec: float = 0.5

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


if __name__ == '__main__':
    Point3dAAEConfig().dump_yaml('point_3d_aae_template.yaml')
