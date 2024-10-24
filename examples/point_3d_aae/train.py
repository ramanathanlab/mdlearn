from __future__ import annotations

import itertools
import random
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from config import Point3dAAEConfig
from torchsummary import summary
from tqdm import tqdm

from mdlearn.data.datasets.point_cloud import PointCloudDataset
from mdlearn.data.utils import train_valid_split
from mdlearn.nn.models.aae.point_3d_aae import AAE3d
from mdlearn.utils import get_torch_optimizer
from mdlearn.utils import log_checkpoint
from mdlearn.utils import parse_args
from mdlearn.utils import resume_checkpoint
from mdlearn.visualize import log_latent_visualization


def main(cfg: Point3dAAEConfig):
    # Create directory for new run, or use old directory if resuming from a checkpoint
    exist_ok = cfg.resume_checkpoint is not None
    cfg.output_path.mkdir(exist_ok=exist_ok)
    checkpoint_path = cfg.output_path / 'checkpoints'
    checkpoint_path.mkdir(exist_ok=exist_ok)
    # Create plot directory
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
    dataset = PointCloudDataset(
        path=cfg.input_path,
        num_points=cfg.num_points,
        num_features=cfg.num_features,
        dataset_name=cfg.dataset_name,
        scalar_dset_names=cfg.scalar_dset_names,
        seed=cfg.seed,
        cms_transform=cfg.cms_transform,
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
    model = AAE3d(
        cfg.num_points,
        cfg.num_features,
        cfg.latent_dim,
        cfg.encoder_bias,
        cfg.encoder_relu_slope,
        cfg.encoder_filters,
        cfg.encoder_kernels,
        cfg.decoder_bias,
        cfg.decoder_relu_slope,
        cfg.decoder_affine_widths,
        cfg.discriminator_bias,
        cfg.discriminator_relu_slope,
        cfg.discriminator_affine_widths,
    )
    model = model.to(device)

    if cfg.wandb:
        cfg.wandb.init(cfg, model, cfg.output_path)

    # Diplay model
    print(model)
    summary(model, (3 + cfg.num_features, cfg.num_points))

    disc_optimizer = get_torch_optimizer(
        cfg.disc_optimizer.name,
        cfg.disc_optimizer.hparams,
        model.discriminator.parameters(),
    )
    ae_optimizer = get_torch_optimizer(
        cfg.ae_optimizer.name,
        cfg.ae_optimizer.hparams,
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
        ),
    )

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
            {'disc_optimizer': disc_optimizer, 'ae_optimizer': ae_optimizer},
        )
        print(
            f'Resume training at epoch {start_epoch} from {cfg.resume_checkpoint}',
        )
    else:
        start_epoch = 0

    # Start training
    for epoch in range(start_epoch, cfg.epochs):
        train_start = time.time()
        # Training
        model.train()
        avg_train_disc_loss, avg_train_ae_loss = train(
            train_loader,
            model,
            disc_optimizer,
            ae_optimizer,
            device,
        )

        print(
            '====> Epoch: {} Train:\tAvg Disc loss: {:.4f}\tAvg AE loss: {:.4f}\tTime: {:.4f}'.format(
                epoch,
                avg_train_disc_loss,
                avg_train_ae_loss,
                time.time() - train_start,
            ),
        )

        valid_start = time.time()
        # Validation
        model.eval()
        with torch.no_grad():
            avg_valid_recon_loss, latent_vectors, scalars = validate(
                valid_loader,
                model,
                device,
            )

        print(
            '====> Epoch: {} Valid:\tAvg recon loss: {:.4f}\tTime: {:.4f}\n'.format(
                epoch,
                avg_valid_recon_loss,
                time.time() - valid_start,
            ),
        )

        print(f'Total time: {time.time() - train_start:.4f}')

        metrics = {
            'train_disc_loss': avg_train_disc_loss,
            'train_ae_loss': avg_train_ae_loss,
            'valid_recon_loss': avg_valid_recon_loss,
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
                {
                    'disc_optimizer': disc_optimizer,
                    'ae_optimizer': ae_optimizer,
                },
            )


def train(train_loader, model: AAE3d, disc_optimizer, ae_optimizer, device):
    avg_disc_loss, avg_ae_loss = 0.0, 0.0
    # Create prior noise buffer array
    noise = torch.FloatTensor(cfg.batch_size, cfg.latent_dim).to(device)
    for batch in tqdm(train_loader):
        x = batch['X'].to(device, non_blocking=True)

        # Encoder/Discriminator forward
        # Get latent vectors
        z = model.encode(x)
        # Get prior noise
        noise.normal_(mean=cfg.noise_mu, std=cfg.noise_std)
        # Get discriminator logits
        real_logits = model.discriminate(noise)
        fake_logits = model.discriminate(z)
        # Discriminator loss
        critic_loss = model.critic_loss(real_logits, fake_logits)
        gp_loss = model.gp_loss(noise, z)
        disc_loss = critic_loss + cfg.lambda_gp * gp_loss

        # Discriminator backward
        disc_optimizer.zero_grad()
        model.discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_optimizer.step()

        # Decoder forward
        recon_x = model.decode(z)
        recon_loss = model.recon_loss(x, recon_x)
        # Discriminator forward
        fake_logit = model.discriminate(z)
        decoder_loss = model.decoder_loss(fake_logit)
        ae_loss = decoder_loss + cfg.lambda_rec * recon_loss

        # AE backward
        ae_optimizer.zero_grad()
        model.decoder.zero_grad()
        model.encoder.zero_grad()
        ae_loss.backward()

        # Collect loss
        avg_disc_loss += disc_loss.item()
        avg_ae_loss += ae_loss.item()

    avg_disc_loss /= len(train_loader)
    avg_ae_loss /= len(train_loader)

    return avg_disc_loss, avg_ae_loss


def validate(valid_loader, model: AAE3d, device):
    scalars = defaultdict(list)
    latent_vectors = []
    avg_ae_loss = 0.0
    for batch in valid_loader:
        x = batch['X'].to(device)
        z = model.encode(x)
        recon_x = model.decode(z)
        avg_ae_loss += model.recon_loss(x, recon_x).item()

        # Collect latent vectors for visualization
        latent_vectors.append(z.cpu().numpy())
        for name in cfg.scalar_dset_names:
            scalars[name].append(batch[name].cpu().numpy())

    avg_ae_loss /= len(valid_loader)
    latent_vectors = np.concatenate(latent_vectors)
    scalars = {
        name: np.concatenate(scalar) for name, scalar in scalars.items()
    }

    return avg_ae_loss, latent_vectors, scalars


if __name__ == '__main__':
    args = parse_args()
    cfg = Point3dAAEConfig.from_yaml(args.config)
    main(cfg)
