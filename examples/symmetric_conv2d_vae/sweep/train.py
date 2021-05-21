import time
import wandb
import torch
import random
import numpy as np
from pathlib import Path
from torchsummary import summary
from mdlearn.utils import (
    log_checkpoint,
    get_torch_optimizer,
    get_torch_scheduler,
)
from mdlearn.data.utils import train_valid_split
from mdlearn.data.datasets.contact_map import ContactMapDataset
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAE
from config import SymmetricConv2dVAEConfig


def main(cfg: SymmetricConv2dVAEConfig):

    # Create checkpoint directory
    checkpoint_path = Path(wandb.run.dir) / "checkpoints"
    checkpoint_path.mkdir()

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_num_threads(cfg.num_data_workers)

    # Hardware
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu"
    )

    # Create model
    model = SymmetricConv2dVAE(
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
    wandb.watch(model) # Must run after summary()

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

    optimizer = get_torch_optimizer(
        cfg.optimizer.name, cfg.optimizer.hparams, model.parameters()
    )
    if cfg.scheduler is not None:
        scheduler = get_torch_scheduler(
            cfg.scheduler.name, cfg.scheduler.hparams, optimizer
        )
    else:
        scheduler = None

    # Optionally initialize model with pre-trained weights
    if cfg.init_weights is not None:
        checkpoint = torch.load(cfg.init_weights, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {cfg.init_weights}")

    epoch_times = []
    # Start training
    for epoch in range(cfg.epochs):
        start = time.time()

        # Training
        model.train()
        avg_train_losses = train(train_loader, model, optimizer, device)

        print(
            "====> Epoch: {} Train:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}".format(
                epoch, *avg_train_losses
            )
        )

        # Validation
        model.eval()
        with torch.no_grad():
            avg_valid_losses = validate(valid_loader, model, device)

        print(
            "====> Epoch: {} Valid:\tAvg loss: {:.4f}\tAvg recon loss {:.4f}\tAvg kld loss {:.4f}".format(
                epoch, *avg_valid_losses
            )
        )
        elapsed = time.time() - start
        print(f"Epoch: {epoch} Time: {elapsed}\n")
        epoch_times.append(elapsed)

        metrics = {
            "train_loss": avg_train_losses[0],
            "train_recon_loss": avg_train_losses[1],
            "train_kld_loss": avg_train_losses[2],
            "valid_loss": avg_valid_losses[0],
            "valid_recon_loss": avg_valid_losses[1],
            "valid_kld_loss": avg_valid_losses[2],
        }
        wandb.log(metrics)

        # Step the learning rate scheduler
        if scheduler is None:
            pass
        elif cfg.scheduler.name == "ReduceLROnPlateau":
            scheduler.step(avg_valid_losses[0])
        else:
            raise NotImplementedError(f"scheduler {cfg.scheduler.name} step function.")

        if epoch % cfg.checkpoint_log_every == 0:
            log_checkpoint(
                checkpoint_path / f"checkpoint-epoch-{epoch}.pt",
                epoch,
                model,
                optimizer,
                scheduler,
            )

    print("Elapsed avg time", np.mean(elapsed))

    # Output directory structure
    # output_path
    # ├── checkpoint
    # │     ├── epoch-1-20200606-125334.pt
    # │     ├── epoch-2-20200606-125338.pt
    # ├── wandb/


def train(train_loader, model, optimizer, device):
    avg_loss, avg_recon_loss, avg_kld_loss = 0.0, 0.0, 0.0
    for batch in train_loader:

        x = batch["X"].to(device, non_blocking=True)

        # Forward pass
        _, recon_x = model(x)
        kld_loss = model.kld_loss()
        recon_loss = model.recon_loss(x, recon_x)
        loss = cfg.lambda_rec * recon_loss + kld_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_max_norm)
        optimizer.step()

        # Collect loss
        avg_loss += loss.item()
        avg_recon_loss += recon_loss.item()
        avg_kld_loss += kld_loss.item()

    avg_loss /= len(train_loader)
    avg_recon_loss /= len(train_loader)
    avg_kld_loss /= len(train_loader)

    return avg_loss, avg_recon_loss, avg_kld_loss


def validate(valid_loader, model, device):
    avg_loss, avg_recon_loss, avg_kld_loss = 0.0, 0.0, 0.0
    for batch in valid_loader:

        x = batch["X"].to(device, non_blocking=True)

        # Forward pass
        _, recon_x = model(x)
        kld_loss = model.kld_loss()
        recon_loss = model.recon_loss(x, recon_x)
        loss = cfg.lambda_rec * recon_loss + kld_loss

        # Collect loss
        avg_loss += loss.item()
        avg_recon_loss += recon_loss.item()
        avg_kld_loss += kld_loss.item()

    avg_loss /= len(valid_loader)
    avg_recon_loss /= len(valid_loader)
    avg_kld_loss /= len(valid_loader)

    return avg_loss, avg_recon_loss, avg_kld_loss


if __name__ == "__main__":
    #wandb.init(dir=wandb.config.output_path)
    wandb.init()
    cfg = SymmetricConv2dVAEConfig.from_yaml(wandb.config.default_yaml)

    # Update cfg with sweep parameters
    cfg.batch_size = wandb.config.batch_size
    cfg.optimizer.name = wandb.config.optimizer
    cfg.optimizer.hparams["lr"] = wandb.config.lr
    cfg.latent_dim = wandb.config.latent_dim

    main(cfg)
