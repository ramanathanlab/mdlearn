import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from torch.nn import functional as F
from mdlearn.utils import PathLike
from mdlearn.nn.models.ae import AE
from mdlearn.nn.modules.dense_net import DenseNet


class LinearAE(AE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        neurons: List[int] = [128],
        bias: bool = True,
        relu_slope: float = 0.0,
        inplace_activation: bool = False,
    ):
        """A symmetric autoencoder with all linear layers.
        Applies a ReLU activation between encoder and decoder.

        Parameters
        ----------
        input_dim : int
            Dimension of input tensor (should be flattened).
        latent_dim: int
            Dimension of the latent space.
        neurons : List[int], optional
            Linear layers :obj:`in_features`, by default [128].
        bias : bool, optional
            Use a bias term in the Linear layers, by default True.
        relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0.
        inplace_activation : bool, optional
            Sets the inplace option for the activation function.
        """

        neurons = neurons.copy() + [latent_dim]
        encoder = DenseNet(input_dim, neurons, bias, relu_slope, inplace_activation)
        decoder_neurons = list(reversed(neurons))[1:] + [input_dim]
        decoder = DenseNet(
            neurons[-1], decoder_neurons, bias, relu_slope, inplace_activation
        )

        super().__init__(encoder, decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z = F.relu(z)
        recon_x = self.decode(z)
        return z, recon_x

    def recon_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        r"""Compute the MSE reconstruction loss between :obj:`x` and :obj:`recon_x`.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        recon_x : torch.Tensor
            The reconstruction of the input data :obj:`x`

        Returns
        -------
        torch.Tensor
            The reconstruction loss between :obj:`x` and :obj:`recon_x`.
        """
        return F.mse_loss(recon_x, x, reduction=reduction)


class LinearAETrainer:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        neurons: List[int] = [128],
        bias: bool = True,
        relu_slope: float = 0.0,
        inplace_activation: bool = False,
        seed: int = 42,
        num_data_workers: int = 2,
        split_pct: float = 8.0,
        batch_size: int = 128,
        shuffle: bool = True,
        device: str = "cuda",
        optimizer_name: str = "RMSprop",
        optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001},
        scheduler_name: Optional[str] = None,
        scheduler_hparams: Dict[str, Any] = {},
        epochs: int = 100,
        verbose: bool = False,
        clip_grad_max_norm: float = 10.0,
        checkpoint_log_every: int = 10,
        plot_log_every: int = 10,
        plot_n_samples: int = 10000,
        plot_method: str = "TSNE",
    ):

        if "cuda" in device and not torch.cuda.is_available():
            raise ValueError("Specified cuda, but it is unavailable.")

        self.seed = seed
        self.num_data_workers = num_data_workers
        self.split_pct = split_pct
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_name = scheduler_name
        self.scheduler_hparams = scheduler_hparams
        self.epochs = epochs
        self.verbose = verbose
        self.clip_grad_max_norm = clip_grad_max_norm
        self.checkpoint_log_every = checkpoint_log_every
        self.plot_log_every = plot_log_every
        self.plot_n_samples = plot_n_samples
        self.plot_method = plot_method

        from mdlearn.utils import get_torch_optimizer, get_torch_scheduler

        self.model = LinearAE(
            input_dim, latent_dim, neurons, bias, relu_slope, inplace_activation
        ).to(self.device)

        # Setup optimizer
        self.optimizer = get_torch_optimizer(
            self.optimizer_name, self.optimizer_hparams, self.model.parameters()
        )

        # Setup learning rate scheduler
        if self.scheduler_name is not None:
            self.scheduler = get_torch_scheduler(
                self.scheduler_name, self.scheduler_hparams, self.optimizer
            )
        else:
            self.scheduler = None

    def fit(
        self,
        X: np.ndarray,
        scalars: Dict[str, np.ndarray] = {},
        checkpoint: Optional[PathLike] = None,
        output_path: PathLike = "./",
    ):

        from mdlearn.utils import (
            resume_checkpoint,
            log_checkpoint,
            log_latent_visualization,
        )
        from mdlearn.data.utils import train_valid_split
        from mdlearn.data.datasets.feature_vector import FeatureVectorDataset

        exist_ok = checkpoint is not None
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=exist_ok)
        # Create checkpoint directory
        checkpoint_path = output_path / "checkpoints"
        checkpoint_path.mkdir(exist_ok=exist_ok)
        # Create plot directory
        plot_path = output_path / "plots"
        plot_path.mkdir(exist_ok=exist_ok)

        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Set available number of cores
        torch.set_num_threads(
            1 if self.num_data_workers == 0 else self.num_data_workers
        )

        # Load training and validation data
        dataset = FeatureVectorDataset(X, scalars)
        train_loader, valid_loader = train_valid_split(
            dataset,
            self.split_pct,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_data_workers,
            drop_last=True,
            pin_memory=True,
        )
        self.scalar_dset_names = list(scalars.keys())

        # Optionally resume training from a checkpoint
        if checkpoint is not None:
            start_epoch = resume_checkpoint(
                checkpoint, self.model, {"optimizer": self.optimizer}, self.scheduler
            )
            if self.verbose:
                print(f"Resume training at epoch {start_epoch} from {checkpoint}")
        else:
            start_epoch = 0

        # Start training
        for epoch in range(start_epoch, self.epochs):
            # Training
            self.model.train()
            avg_train_loss = self._train(train_loader)

            if self.verbose:
                print(
                    "====> Epoch: {} Train:\tAvg loss: {:.4f}".format(
                        epoch, avg_train_loss
                    )
                )

            # Validation
            self.model.eval()
            with torch.no_grad():
                avg_valid_loss, z, paints = self._validate(valid_loader)

            if self.verbose:
                print(
                    "====> Epoch: {} Valid:\tAvg loss: {:.4f}\n".format(
                        epoch, avg_valid_loss
                    )
                )

            # Step the learning rate scheduler
            if self.scheduler is None:
                pass
            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(avg_valid_loss)
            else:
                raise NotImplementedError(
                    f"scheduler {self.scheduler_name} step function."
                )

            # Log a model checkpoint file
            if epoch % self.checkpoint_log_every == 0:
                log_checkpoint(
                    checkpoint_path / f"checkpoint-epoch-{epoch}.pt",
                    epoch,
                    self.model,
                    {"optimizer": self.optimizer},
                    self.scheduler,
                )

            # Log a visualization of the latent space
            if epoch % self.plot_log_every == 0:
                log_latent_visualization(
                    z,
                    paints,
                    plot_path,
                    epoch,
                    self.plot_n_samples,
                    self.plot_method,
                )

    def _train(self, train_loader) -> float:
        avg_loss = 0.0
        for batch in train_loader:

            x = batch["X"].to(self.device)

            # Forward pass
            _, recon_x = self.model(x)
            loss = self.model.recon_loss(x, recon_x)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_max_norm
            )
            self.optimizer.step()

            # Collect loss
            avg_loss += loss.item()

        avg_loss /= len(train_loader)

        return avg_loss

    def _validate(
        self, valid_loader
    ) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        paints = defaultdict(list)
        latent_vectors = []
        avg_loss = 0.0
        for batch in valid_loader:

            x = batch["X"].to(self.device)

            # Forward pass
            z, recon_x = self.model(x)
            loss = self.model.recon_loss(x, recon_x)

            # Collect loss
            avg_loss += loss.item()

            # Collect latent vectors for visualization
            latent_vectors.append(z.cpu().numpy())
            for name in self.scalar_dset_names:
                paints[name].append(batch[name].cpu().numpy())

        avg_loss /= len(valid_loader)
        # Group latent vectors and paints
        latent_vectors = np.concatenate(latent_vectors)
        paints = {name: np.concatenate(scalar) for name, scalar in paints.items()}

        return avg_loss, latent_vectors, paints
