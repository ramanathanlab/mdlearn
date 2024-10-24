from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import wandb
from config import SymmetricConv2dVDEConfig
from torchsummary import summary
from tqdm import tqdm

from mdlearn.data.datasets.time_contact_map import ContactMapTimeSeriesDataset
from mdlearn.data.utils import train_valid_split
from mdlearn.nn.models.vde.symmetric_conv2d_vde import SymmetricConv2dVDE
from mdlearn.utils import get_torch_optimizer
from mdlearn.utils import get_torch_scheduler
from mdlearn.utils import log_checkpoint
from mdlearn.utils import log_latent_visualization
from mdlearn.utils import parse_args
from mdlearn.utils import resume_checkpoint


def main(cfg: SymmetricConv2dVDEConfig):
    # Create directory for new run, or use old directory if resuming from a checkpoint
    exist_ok = cfg.resume_checkpoint is not None
    cfg.output_path.mkdir(exist_ok=exist_ok)
    checkpoint_path = cfg.output_path.joinpath('checkpoints')
    checkpoint_path.mkdir(exist_ok=exist_ok)
    plot_path = cfg.output_path / 'plots'
    plot_path.mkdir(exist_ok=exist_ok)

    # Copy training data to output directory to not slow down other
    # training processes using the same data.
    # cfg.input_path = shutil.copy(cfg.input_path, cfg.output_path)

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_num_threads(cfg.num_data_workers)

    # Load training and validation data
    dataset = ContactMapTimeSeriesDataset(
        path=cfg.input_path,
        shape=cfg.input_shape,
        lag_time=cfg.lag_time,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        values_dset_name=cfg.values_dset_name,
        scalar_requires_grad=cfg.scalar_requires_grad,
        in_memory=cfg.in_memory,
    )
    train_loader, valid_loader = train_valid_split(
        dataset,
        cfg.split_pct,
        drop_last=True,
        pin_memory=True,
        shuffle=cfg.shuffle,
        batch_size=cfg.batch_size,
        persistent_workers=True,
        num_workers=cfg.num_data_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    # Hardware
    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available() and not cfg.ignore_gpu
        else 'cpu',
    )

    # Create model
    model = SymmetricConv2dVDE(
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

    if cfg.wandb:
        cfg.wandb.init(cfg, model, cfg.output_path)

    # Diplay model
    print(model)
    summary(model, cfg.input_shape)

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

    # Optionally resume training from a checkpoint
    if cfg.resume_checkpoint is not None:
        start_epoch = resume_checkpoint(
            cfg.resume_checkpoint,
            model,
            {'optimizer': optimizer},
            scheduler,
        )
        print(
            f'Resume training at epoch {start_epoch} from {cfg.resume_checkpoint}',
        )
    else:
        start_epoch = 0

    # Start training
    for epoch in range(start_epoch, cfg.epochs):
        # Training
        model.train()
        avg_train_losses = train(train_loader, model, optimizer, device)

        print(
            '====> Epoch: {} Train:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}\tAvg ac loss {:.4f}'.format(
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
                avg_ac_loss,
                latent_vectors,
                scalars,
            ) = validate(valid_loader, model, device)

        print(
            '====> Epoch: {} Valid:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}\tAvg ac loss {:.4f}\n'.format(
                epoch,
                avg_loss,
                avg_recon_loss,
                avg_kld_loss,
                avg_ac_loss,
            ),
        )

        # Step the learning rate scheduler
        if scheduler is None:
            pass
        elif cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(avg_loss)
        else:
            raise NotImplementedError(
                f'scheduler {cfg.scheduler.name} step function.',
            )

        metrics = {
            'train_loss': avg_train_losses[0],
            'train_recon_loss': avg_train_losses[1],
            'train_kld_loss': avg_train_losses[2],
            'train_ac_loss': avg_train_losses[3],
            'valid_loss': avg_loss,
            'valid_recon_loss': avg_recon_loss,
            'valid_kld_loss': avg_kld_loss,
            'valid_ac_loss': avg_ac_loss,
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
            if cfg.wandb:
                for name, html_string in html_strings.items():
                    metrics[name] = wandb.Html(html_string, inject=False)

        if cfg.wandb:
            wandb.log(metrics)

        if epoch % cfg.checkpoint_log_every == 0:
            log_checkpoint(
                checkpoint_path / f'checkpoint-epoch-{epoch}.pt',
                epoch,
                model,
                {'optimizer': optimizer},
                scheduler,
            )


def train(train_loader, model: SymmetricConv2dVDE, optimizer, device):
    avg_loss, avg_recon_loss, avg_kld_loss, avg_ac_loss = 0.0, 0.0, 0.0, 0.0
    for batch in tqdm(train_loader):
        x_t = batch['X_t'].to(device, non_blocking=True)
        x_t_tau = batch['X_t_tau'].to(device, non_blocking=True)

        # Forward pass
        z_t, recon_x_t_tau = model(x_t)
        kld_loss = model.kld_loss()
        recon_loss = model.recon_loss(x_t_tau, recon_x_t_tau)
        z_t_tau = model.encode(x_t_tau)
        ac_loss = model.ac_loss(z_t, z_t_tau)
        loss = cfg.lambda_rec * recon_loss + kld_loss + ac_loss

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
        avg_ac_loss += ac_loss.item()

    avg_loss /= len(train_loader)
    avg_recon_loss /= len(train_loader)
    avg_kld_loss /= len(train_loader)
    avg_ac_loss /= len(train_loader)

    return avg_loss, avg_recon_loss, avg_kld_loss, avg_ac_loss


def validate(valid_loader, model, device):
    scalars = defaultdict(list)
    latent_vectors = []
    avg_loss, avg_recon_loss, avg_kld_loss, avg_ac_loss = 0.0, 0.0, 0.0, 0.0
    for batch in valid_loader:
        x_t = batch['X_t'].to(device, non_blocking=True)
        x_t_tau = batch['X_t_tau'].to(device, non_blocking=True)

        # Forward pass
        z_t, recon_x_t_tau = model(x_t)
        kld_loss = model.kld_loss()
        recon_loss = model.recon_loss(x_t_tau, recon_x_t_tau)
        z_t_tau = model.encode(x_t_tau)
        ac_loss = model.ac_loss(z_t, z_t_tau)
        loss = cfg.lambda_rec * recon_loss + kld_loss + ac_loss

        # Collect loss
        avg_loss += loss.item()
        avg_recon_loss += recon_loss.item()
        avg_kld_loss += kld_loss.item()
        avg_ac_loss += ac_loss.item()

        # Collect latent vectors for visualization
        latent_vectors.append(z_t.cpu().numpy())
        for name in cfg.scalar_dset_names:
            scalars[name].append(batch[name].cpu().numpy())

    avg_loss /= len(valid_loader)
    avg_recon_loss /= len(valid_loader)
    avg_kld_loss /= len(valid_loader)
    avg_ac_loss /= len(valid_loader)
    latent_vectors = np.concatenate(latent_vectors)
    scalars = {
        name: np.concatenate(scalar) for name, scalar in scalars.items()
    }

    return (
        avg_loss,
        avg_recon_loss,
        avg_kld_loss,
        avg_ac_loss,
        latent_vectors,
        scalars,
    )


if __name__ == '__main__':
    args = parse_args()
    cfg = SymmetricConv2dVDEConfig.from_yaml(args.config)
    main(cfg)
