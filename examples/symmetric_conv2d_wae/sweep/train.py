from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from config import SymmetricConv2dWAEConfig
from torchsummary import summary

from mdlearn.data.datasets.contact_map import ContactMapDataset
from mdlearn.data.utils import train_valid_split
from mdlearn.metrics import metric_cluster_quality
from mdlearn.nn.models.wae.symmetric_conv2d_wae import SymmetricConv2dWAE
from mdlearn.utils import get_torch_optimizer
from mdlearn.utils import get_torch_scheduler
from mdlearn.utils import log_checkpoint
from mdlearn.visualize import log_latent_visualization


def main(cfg: SymmetricConv2dWAEConfig):
    # Create checkpoint directory
    checkpoint_path = Path(wandb.run.dir) / 'checkpoints'
    checkpoint_path.mkdir()
    # Create plot directory
    plot_path = Path(wandb.run.dir) / 'plots'
    plot_path.mkdir()

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_num_threads(cfg.num_data_workers)

    # Load training and validation data
    dataset = ContactMapDataset(
        path=cfg.input_path,
        shape=cfg.input_shape,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        values_dset_name=cfg.values_dset_name,
        scalar_requires_grad=cfg.scalar_requires_grad,
        in_memory=cfg.in_memory,
    )
    train_loader, valid_loader = train_valid_split(
        dataset,
        cfg.split_pct,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_data_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
    )

    # Hardware
    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available() and not cfg.ignore_gpu
        else 'cpu',
    )

    # Create model
    model = SymmetricConv2dWAE(
        cfg.input_shape,
        cfg.init_weights,
        cfg.filters,
        cfg.kernels,
        cfg.strides,
        cfg.affine_widths,
        cfg.affine_dropouts,
        cfg.latent_dim,
        cfg.activation,
        cfg.output_activation,
    )
    model = model.to(device)

    # Diplay model
    print(model)
    summary(model, cfg.input_shape)
    wandb.watch(model)  # Must run after summary()

    optimizer = get_torch_optimizer(
        cfg.optimizer.name,
        cfg.optimizer.hparams,
        model.parameters(),
    )
    if cfg.scheduler is not None:
        scheduler = get_torch_scheduler(
            cfg.scheduler.name,
            cfg.scheduler.hparams,
            optimizer,
        )
    else:
        scheduler = None

    # Optionally initialize model with pre-trained weights
    if cfg.init_weights is not None:
        checkpoint = torch.load(cfg.init_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {cfg.init_weights}')

    # Start training
    epoch_times = []
    print(
        f'Start training with {round(len(train_loader) * cfg.train_subsample_pct)} batches '
        f'and {round(len(valid_loader) * cfg.valid_subsample_pct)} validation batches.',
    )
    for epoch in range(cfg.epochs):
        start = time.time()

        # Training
        model.train()
        avg_train_losses = train(train_loader, model, optimizer, device)

        print(
            '====> Epoch: {} Train:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}\tAvg mmd loss {:.4f}'.format(
                epoch,
                *avg_train_losses,
            ),
        )

        # Validation
        model.eval()
        with torch.no_grad():
            (
                avg_loss,
                avg_recon_loss,
                avg_kld_loss,
                avg_mmd_loss,
                latent_vectors,
                scalars,
            ) = validate(valid_loader, model, device)

        print(
            '====> Epoch: {} Valid:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}\tAvg mmd loss {:.4f}'.format(
                epoch,
                avg_loss,
                avg_recon_loss,
                avg_kld_loss,
                avg_mmd_loss,
            ),
        )
        elapsed = time.time() - start
        print(f'Epoch: {epoch} Time: {elapsed}\n')
        epoch_times.append(elapsed)

        start = time.time()
        cluster_quality = metric_cluster_quality(
            latent_vectors,
            scalars['rmsd'],
        )
        print('cluster quality metric time: ', time.time() - start)

        metrics = {
            'train_loss': avg_train_losses[0],
            'train_recon_loss': avg_train_losses[1],
            'train_kld_loss': avg_train_losses[2],
            'valid_loss': avg_loss,
            'valid_recon_loss': avg_recon_loss,
            'valid_kld_loss': avg_kld_loss,
            'cluster_quality': cluster_quality,
        }

        # Visualize latent space
        if epoch % cfg.plot_log_every == 0:
            html_strings = log_latent_visualization(
                latent_vectors,
                scalars,
                plot_path,
                epoch,
                cfg.plot_n_samples,
                cfg.plot_method,
            )
            for name, html_string in html_strings.items():
                metrics[name] = wandb.Html(html_string, inject=False)

        wandb.log(metrics)

        # Step the learning rate scheduler
        if scheduler is None:
            pass
        elif cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(avg_loss)
        else:
            raise NotImplementedError(
                f'scheduler {cfg.scheduler.name} step function.',
            )

        if epoch % cfg.checkpoint_log_every == 0:
            log_checkpoint(
                checkpoint_path / f'checkpoint-epoch-{epoch}.pt',
                epoch,
                model,
                {'optimizer': optimizer},
                scheduler,
            )

    print('Elapsed avg time', np.mean(epoch_times))

    # Output directory structure
    # output_path
    # ├── checkpoint
    # │     ├── epoch-1-20200606-125334.pt
    # │     ├── epoch-2-20200606-125338.pt
    # ├── wandb/


def train(train_loader, model, optimizer, device):
    avg_loss, avg_recon_loss, avg_kld_loss, avg_mmd_loss, i = (
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    )
    for i, batch in enumerate(train_loader):
        if i / len(train_loader) > cfg.train_subsample_pct:
            break  # Early stop for sweeps

        x = batch['X'].to(device, non_blocking=True)

        # Forward pass
        z, recon_x = model(x)
        kld_loss = model.kld_loss()  # For logging
        mmd_loss = model.mmdrf_loss(
            z,
            cfg.sigma,
            cfg.kernel,
            cfg.rf_dim,
            cfg.rf_resample,
        )
        recon_loss = model.recon_loss(x, recon_x)
        loss = cfg.lambda_rec * recon_loss + mmd_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.clip_grad_max_norm,
        )
        optimizer.step()

        # Collect loss
        avg_loss += loss.item()
        avg_recon_loss += recon_loss.item()
        avg_kld_loss += kld_loss.item()
        avg_mmd_loss += mmd_loss.item()

    avg_loss /= i + 1
    avg_recon_loss /= i + 1
    avg_kld_loss /= i + 1
    avg_mmd_loss /= i + 1

    return avg_loss, avg_recon_loss, avg_kld_loss, avg_mmd_loss


def validate(
    valid_loader,
    model,
    device,
) -> tuple[float, float, float, float, np.ndarray, dict[str, np.ndarray]]:
    scalars = defaultdict(list)
    latent_vectors = []
    avg_loss, avg_recon_loss, avg_kld_loss, avg_mmd_loss, i = (
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    )
    for i, batch in enumerate(valid_loader):
        if i / len(valid_loader) > cfg.valid_subsample_pct:
            break  # Early stop for sweeps

        x = batch['X'].to(device, non_blocking=True)

        # Forward pass
        z, recon_x = model(x)
        kld_loss = model.kld_loss()
        mmd_loss = model.mmdrf_loss(
            z,
            cfg.sigma,
            cfg.kernel,
            cfg.rf_dim,
            cfg.rf_resample,
        )
        recon_loss = model.recon_loss(x, recon_x)
        loss = cfg.lambda_rec * recon_loss + kld_loss

        # Collect loss
        avg_loss += loss.item()
        avg_recon_loss += recon_loss.item()
        avg_kld_loss += kld_loss.item()
        avg_mmd_loss += mmd_loss.item()

        # Collect latent vectors for visualization
        latent_vectors.append(z.cpu().numpy())
        for name in cfg.scalar_dset_names:
            scalars[name].append(batch[name].cpu().numpy())

    avg_loss /= i + 1
    avg_recon_loss /= i + 1
    avg_kld_loss /= i + 1
    avg_mmd_loss /= i + 1

    latent_vectors = np.concatenate(latent_vectors)
    scalars = {
        name: np.concatenate(scalar) for name, scalar in scalars.items()
    }

    return (
        avg_loss,
        avg_recon_loss,
        avg_kld_loss,
        avg_mmd_loss,
        latent_vectors,
        scalars,
    )


if __name__ == '__main__':
    wandb.init()
    cfg = SymmetricConv2dWAEConfig.from_yaml(wandb.config.default_yaml)

    # Update cfg with sweep parameters
    cfg.batch_size = wandb.config.batch_size
    cfg.optimizer.name = wandb.config.optimizer
    cfg.optimizer.hparams['lr'] = wandb.config.lr
    cfg.latent_dim = wandb.config.latent_dim
    cfg.lambda_rec = wandb.config.lambda_rec
    cfg.sigma = np.floor(np.sqrt(wandb.config.latent_dim))
    cfg.rf_dim = wandb.config.rf_dim

    main(cfg)
