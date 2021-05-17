import torch
import random
import numpy as np
from torchsummary import summary
from pathlib import Path
from typing import Optional, List, Tuple
from mdlearn.utils import (
    BaseSettings,
    WandbConfig,
    OptimizerConfig,
    SchedulerConfig,
    parse_args,
    log_checkpoint,
    resume_checkpoint,
)
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAE
from mdlearn.data.utils import train_valid_split
from mdlearn.data.datasets.contact_map import ContactMapDataset


class SymmetricConv2dVAEConfig(BaseSettings):
    # File paths
    # Path to HDF5 training file
    input_path: Path = Path("TODO")
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("TODO")
    # Optionally resume training from a checkpoint file
    resume_checkpoint: Optional[Path] = None

    # Input image shapes
    input_shape: Tuple[int, ...] = (1, 28, 28)
    # Name of the dataset in the HDF5 file.
    dataset_name: str = "contact_map"
    # Name of scalar datasets in the HDF5 file.
    scalar_dset_names: List[str] = []
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
    optimizer: OptimizerConfig = OptimizerConfig(name="Adam", hparams={"lr": 0.0001})
    # Learning rate scheduler params
    scheduler: Optional[SchedulerConfig] = None
    # Wandb params
    wandb: Optional[WandbConfig] = WandbConfig()

    # Model hyperparameters
    latent_dim: int = 64
    filters: List[int] = [100, 100, 100, 100]
    kernels: List[int] = [5, 5, 5, 5]
    strides: List[int] = [1, 2, 1, 1]
    latent_dim: int = 10
    affine_widths: List[int] = [64]
    affine_dropouts: List[float] = [0.0]
    activation: str = "ReLU"
    output_activation: str = "None"  # Identity function
    lambda_rec: float = 1.0

    # Training settings
    # Number of data loaders for training
    num_data_workers: int = 0
    # Whether or not to ignore the GPU while training.
    ignore_gpu: bool = False


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
