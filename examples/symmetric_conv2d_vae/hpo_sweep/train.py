from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import wandb

from mdlearn.nn.models.vae.symmetric_conv2d_vae import (
    SymmetricConv2dVAETrainer,
)
from mdlearn.utils import BaseModel


class SymmetricConv2dVAEConfig(BaseModel):
    # File paths
    # Path to HDF5 training file
    input_path: Path = Path('TODO')
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path('TODO')
    # Optionally resume training from a checkpoint file
    checkpoint_path: Optional[Path] = None

    input_shape: tuple[int, int, int] = (1, 28, 28)
    filters: list[int] = [100, 100, 100, 100]
    kernels: list[int] = [5, 5, 5, 5]
    strides: list[int] = [1, 2, 1, 2]
    affine_widths: list[int] = [128, 64]
    affine_dropouts: list[float] = [0.0, 0.0]
    latent_dim: int = 10
    activation: str = 'ReLU'
    output_activation: str = 'Sigmoid'
    lambda_rec: float = 1.0
    seed: int = 42
    num_data_workers: int = 0
    prefetch_factor: int = 2
    split_pct: float = 0.8
    split_method: str = 'random'
    shuffle: bool = True
    epochs: int = 50
    batch_size: int = 64
    device: str = 'cuda'
    clip_grad_max_norm: float = 10.0
    optimizer_name: str = 'RMSprop'
    optimizer_hparams: dict[str, Any] = {'lr': 0.001, 'weight_decay': 0.00001}
    scheduler_name: Optional[str] = None
    scheduler_hparams: dict[str, Any] = {}
    verbose: bool = False
    checkpoint_log_every: int = 10
    plot_log_every: int = 50
    plot_n_samples: int = 10000
    plot_method: Optional[str] = None
    train_subsample_pct: float = 0.5
    valid_subsample_pct: float = 0.5
    use_wandb: bool = True
    inference_batch_size: int = 128


def main(cfg: SymmetricConv2dVAEConfig):
    # Initialize the model
    trainer = SymmetricConv2dVAETrainer(
        input_shape=cfg.input_shape,
        filters=cfg.filters,
        kernels=cfg.kernels,
        strides=cfg.strides,
        affine_widths=cfg.affine_widths,
        affine_dropouts=cfg.affine_dropouts,
        latent_dim=cfg.latent_dim,
        activation=cfg.activation,
        output_activation=cfg.output_activation,
        lambda_rec=cfg.lambda_rec,
        seed=cfg.seed,
        num_data_workers=cfg.num_data_workers,
        prefetch_factor=cfg.prefetch_factor,
        split_pct=cfg.split_pct,
        split_method=cfg.split_method,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        device=cfg.device,
        optimizer_name=cfg.optimizer_name,
        optimizer_hparams=cfg.optimizer_hparams,
        scheduler_name=cfg.scheduler_name,
        scheduler_hparams=cfg.scheduler_hparams,
        epochs=cfg.epochs,
        verbose=cfg.verbose,
        clip_grad_max_norm=cfg.clip_grad_max_norm,
        checkpoint_log_every=cfg.checkpoint_log_every,
        plot_log_every=cfg.plot_log_every,
        plot_n_samples=cfg.plot_n_samples,
        plot_method=cfg.plot_method,
        train_subsample_pct=cfg.train_subsample_pct,
        valid_subsample_pct=cfg.valid_subsample_pct,
        use_wandb=cfg.use_wandb,
    )

    print(trainer.model)

    # Load input data from HDF5 file
    with h5py.File(cfg.input_path) as f:
        contact_maps = f['contact_map'][...]
        scalars = {'rmsd': f['rmsd'][...]}

    print(f'Number of contact maps: {len(contact_maps)}')

    # Train model
    trainer.fit(
        X=contact_maps,
        scalars=scalars,
        output_path=cfg.output_path,
        checkpoint=cfg.checkpoint_path,
    )

    pd.DataFrame(trainer.loss_curve_).to_csv(cfg.output_path / 'loss.csv')

    # Generate latent embeddings in inference mode
    (
        z,
        loss,
        recon_loss,
        kld_loss,
    ) = trainer.predict(
        X=contact_maps,
        inference_batch_size=cfg.inference_batch_size,
    )

    np.save(cfg.output_path / 'z.npy', z)

    print(
        f'Final loss on the full dataset is: {loss}, recon: {recon_loss}, kld: {kld_loss}',
    )


if __name__ == '__main__':
    # Generate sample yaml
    # SymmetricConv2dVAEConfig().dump_yaml("symmetric_conv2d_vae_template.yaml")
    # exit()

    wandb.init()
    cfg = SymmetricConv2dVAEConfig.from_yaml(wandb.config.default_yaml)
    cfg.use_wandb = True

    # Update cfg with sweep parameters
    cfg.batch_size = wandb.config.batch_size
    cfg.optimizer_name = wandb.config.optimizer
    cfg.optimizer_hparams['lr'] = wandb.config.lr
    cfg.latent_dim = wandb.config.latent_dim
    cfg.lambda_rec = wandb.config.lambda_rec

    main(cfg)
