import torch
import random
import numpy as np
from torchsummary import summary
from mdlearn.utils import (
    parse_args,
    log_checkpoint,
    resume_checkpoint,
)
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAE
from mdlearn.data.utils import train_valid_split
from mdlearn.data.datasets.contact_map import ContactMapDataset
from config import SymmetricConv2dVAEConfig


def main(cfg: SymmetricConv2dVAEConfig):

    # Create directory for new run, or use old directory if resuming from a checkpoint
    exist_ok = cfg.resume_checkpoint is not None
    cfg.output_path.mkdir(exist_ok=exist_ok)
    checkpoint_path = cfg.output_path.joinpath("checkpoints")
    checkpoint_path.mkdir(exist_ok=exist_ok)

    # Copy training data to output directory to not slow down other
    # training processes using the same data.
    # cfg.input_path = shutil.copy(cfg.input_path, cfg.output_path)

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

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

    if cfg.wandb:
        cfg.wandb.init(cfg, model, cfg.output_path)

    # Diplay model
    print(model)
    summary(model, cfg.input_shape)

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
    )

    optimizer = cfg.optimizer.get_torch_optimizer(model.parameters())
    if cfg.scheduler is not None:
        scheduler = cfg.scheduler.get_torch_scheduler(optimizer)
    else:
        scheduler = None

    # Optionally initialize model with pre-trained weights
    if cfg.init_weights is not None:
        checkpoint = torch.load(cfg.init_weights, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {cfg.init_weights}")

    # Optionally resume training from a checkpoint
    if cfg.resume_checkpoint is not None:
        start_epoch = resume_checkpoint(
            cfg.resume_checkpoint, model, optimizer, scheduler
        )
        print(f"Resume training at epoch {start_epoch} from {cfg.resume_checkpoint}")
    else:
        start_epoch = 0

    # Start training
    for epoch in range(start_epoch, cfg.epochs):
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

        # Step the learning rate scheduler
        if scheduler is None:
            pass
        elif cfg.scheduler.name == "ReduceLROnPlateau":
            scheduler.step(avg_valid_losses[0])
        else:
            raise NotImplementedError(f"scheduler {cfg.scheduler.name} step function.")

        log_checkpoint(
            checkpoint_path / f"checkpoint-epoch-{epoch}.pt",
            epoch,
            model,
            optimizer,
            scheduler,
        )

    # Output directory structure
    # output_path
    # ├── checkpoint
    # │     ├── epoch-1-20200606-125334.pt
    # │     ├── epoch-2-20200606-125338.pt
    # |__ wandb/


def train(train_loader, model, optimizer, device):
    avg_loss, avg_recon_loss, avg_kld_loss = 0.0, 0.0, 0.0
    for batch in train_loader:

        x = batch["X"].to(device)

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

        x = batch["X"].to(device)

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
    args = parse_args()
    cfg = SymmetricConv2dVAEConfig.from_yaml(args.config)
    main(cfg)
