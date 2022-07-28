from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mdlearn.nn.models.vae import VAE
from mdlearn.nn.modules.conv2d_decoder import Conv2dDecoder
from mdlearn.nn.modules.conv2d_encoder import Conv2dEncoder
from mdlearn.nn.utils import PathLike, Trainer


class SymmetricConv2dVAE(VAE):
    """Convolutional variational autoencoder from the
    `"Deep clustering of protein folding simulations"
    <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2507-5>`_ paper.
    Inherits from :obj:`mdlearn.nn.models.vae.VAE`."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        init_weights: Optional[str] = None,
        filters: List[int] = [64, 64, 64],
        kernels: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 1],
        affine_widths: List[int] = [128],
        affine_dropouts: List[float] = [0.0],
        latent_dim: int = 3,
        activation: str = "ReLU",
        output_activation: str = "Sigmoid",
    ):
        """
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            (height, width) input dimensions of input image.
        init_weights : Optional[str]
            .pt weights file to initial weights with.
        filters : List[int]
            Convolutional filter dimensions.
        kernels : List[int]
            Convolutional kernel dimensions (assumes square kernel).
        strides : List[int]
            Convolutional stride lengths (assumes square strides).
        affine_widths : List[int]
            Number of neurons in each linear layer.
        affine_dropouts : List[float]
            Dropout probability for each linear layer. Dropout value
            of 0.0 will skip adding the dropout layer.
        latent_dim : int
            Latent dimension for :math:`mu` and :math:`logstd` layers.
        activation : str
            Activation function to use between convultional and linear layers.
        output_activation : str
            Output activation function for last decoder layer.
        """
        self._check_hyperparameters(
            kernels, strides, filters, affine_widths, affine_dropouts
        )

        encoder = Conv2dEncoder(
            input_shape,
            init_weights,
            filters,
            kernels,
            strides,
            affine_widths,
            affine_dropouts,
            latent_dim,
            activation,
        )

        decoder = Conv2dDecoder(
            input_shape,
            encoder.shapes,
            init_weights,
            list(reversed(filters)),
            list(reversed(kernels)),
            list(reversed(strides)),
            list(reversed(affine_widths)),
            list(reversed(affine_dropouts)),
            latent_dim,
            activation,
            output_activation,
        )

        super().__init__(encoder, decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of variational autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input :obj:`x` data to encode and reconstruct.

        Returns
        -------
        torch.Tensor
            :math:`z`-latent space batch tensor.
        torch.Tensor
            :obj:`recon_x` reconstruction of :obj:`x`.
        """
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x

    def _check_hyperparameters(
        self,
        kernels: List[int],
        strides: List[int],
        filters: List[int],
        affine_widths: List[int],
        affine_dropouts: List[float],
    ):
        """Check that hyperparameters are consistent and logical."""
        if not (len(kernels) == len(strides) == len(filters)):
            raise ValueError("Number of filters, kernels and strides must be equal.")

        if len(affine_dropouts) != len(affine_widths):
            raise ValueError(
                "Number of dropouts must equal the number of affine widths."
            )

        # Common convention: allows for filter center and for even padding
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Only odd valued kernel sizes allowed.")

        if any(p < 0 or p > 1 for p in affine_dropouts):
            raise ValueError("Dropout probabilities, p, must be 0 <= p <= 1.")


class SymmetricConv2dVAETrainer(Trainer):
    """Trainer class to fit a convolutional variational autoencoder
    to a set of contact maps."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: List[int] = [64, 64, 64],
        kernels: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 1],
        affine_widths: List[int] = [128],
        affine_dropouts: List[float] = [0.0],
        latent_dim: int = 3,
        activation: str = "ReLU",
        output_activation: str = "Sigmoid",
        lambda_rec: float = 1.0,
        seed: int = 42,
        num_data_workers: int = 0,
        prefetch_factor: int = 2,
        split_pct: float = 0.8,
        split_method: str = "random",
        batch_size: int = 128,
        shuffle: bool = True,
        device: str = "cpu",
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
        plot_method: Optional[str] = "TSNE",
        train_subsample_pct: float = 1.0,
        valid_subsample_pct: float = 1.0,
        use_wandb: bool = False,
    ):
        """
        Parameters
        ----------
        input_dim : int, default=40
            Dimension of input tensor (should be flattened).
        latent_dim : int, default=3
            Dimension of the latent space.
        hidden_neurons : List[int], default=[32, 16, 8]
            Linear layers :obj:`in_features`. Defines the shape of the autoencoder
            (does not include latent dimension). The encoder and decoder are symmetric.
        bias : bool, default=True
            Use a bias term in the Linear layers.
        relu_slope : float, default=0.0
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`.
        inplace_activation : bool, default=False
            Sets the inplace option for the activation function.
        seed : int, default=42
            Random seed for torch, numpy, and random module.
        num_data_workers : int, default=0
            How many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
        prefetch_factor : int, by default=2
            Number of samples loaded in advance by each worker. 2 means there will be a
            total of 2 * num_workers samples prefetched across all workers.
        split_pct : float, default=0.8
            Proportion of data set to use for training. The rest goes to validation.
        split_method : str, default="random"
            Method to split the data. For random split use "random", for a simple
            partition, use "partition".
        batch_size : int, default=128
            Mini-batch size for training.
        shuffle : bool, default=True
            Whether to shuffle training data or not.
        device : str, default="cpu"
            Specify training hardware either :obj:`cpu` or :obj:`cuda` for GPU devices.
        optimizer_name : str, default="RMSprop"
            Name of the PyTorch optimizer to use. Matches PyTorch optimizer class name.
        optimizer_hparams : Dict[str, Any], default={"lr": 0.001, "weight_decay": 0.00001}
            Dictionary of hyperparameters to pass to the chosen PyTorch optimizer.
        scheduler_name : Optional[str], default=None
            Name of the PyTorch learning rate scheduler to use.
            Matches PyTorch optimizer class name.
        scheduler_hparams : Dict[str, Any], default={}
            Dictionary of hyperparameters to pass to the chosen PyTorch learning rate scheduler.
        epochs : int, default=100
            Number of epochs to train for.
        verbose : bool, default False
            If True, will print training and validation loss at each epoch.
        clip_grad_max_norm : float, default=10.0
            Max norm of the gradients for gradient clipping for more information
            see: :obj:`torch.nn.utils.clip_grad_norm_` documentation.
        checkpoint_log_every : int, default=10
            Epoch interval to log a checkpoint file containing the model
            weights, optimizer, and scheduler parameters.
        plot_log_every : int, default=10
            Epoch interval to log a visualization plot of the latent space.
        plot_n_samples : int, default=10000
            Number of validation samples to use for plotting.
        plot_method : Optional[str], default="TSNE"
            The method for visualizing the latent space or if visualization
            should not be run, set :obj:`plot_method=None`. If using :obj:`"TSNE"`,
            it will attempt to use the RAPIDS.ai GPU implementation and
            will fallback to the sklearn CPU implementation if RAPIDS.ai
            is unavailable. A fast alternative is to plot the raw embeddings
            (or up to the first 3 dimensions if D > 3) using :obj:`"raw"`.
        train_subsample_pct : float, default=1.0
            Percentage of training data to use during hyperparameter sweeps.
        valid_subsample_pct : float, default=1.0
            Percentage of validation data to use during hyperparameter sweeps.
        use_wandb : bool, default=False
            If True, will log results to wandb. Metric keys include `"train_loss"`,
            `"train_recon_loss"`, `"train_kld_loss"`, `"valid_loss"`, `"valid_recon_loss"`
            and `"valid_kld_loss"`.

        Raises
        ------
        ValueError
            :obj:`split_pct` should be between 0 and 1.
        ValueError
            :obj:`train_subsample_pct` should be between 0 and 1.
        ValueError
            :obj:`valid_subsample_pct` should be between 0 and 1.
        ValueError
            Specified :obj:`device` as :obj:`cuda`, but it is unavailable.

        Examples
        --------
        For an accompanying example, see:
        https://github.com/ramanathanlab/mdlearn/tree/main/examples/symmetric_conv2d_vae/training.
        """
        super().__init__(
            seed,
            False,
            num_data_workers,
            prefetch_factor,
            split_pct,
            split_method,
            batch_size,
            shuffle,
            device,
            epochs,
            verbose,
            clip_grad_max_norm,
            checkpoint_log_every,
            plot_log_every,
            plot_n_samples,
            plot_method,
            train_subsample_pct,
            valid_subsample_pct,
            use_wandb,
        )

        self.input_shape = input_shape
        self.lambda_rec = lambda_rec
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_name = scheduler_name
        self.scheduler_hparams = scheduler_hparams

        from mdlearn.utils import get_torch_optimizer, get_torch_scheduler

        self.model = SymmetricConv2dVAE(
            input_shape=input_shape,
            filters=filters,
            kernels=kernels,
            strides=strides,
            affine_widths=affine_widths,
            affine_dropouts=affine_dropouts,
            latent_dim=latent_dim,
            activation=activation,
            output_activation=output_activation,
        ).to(self.device)

        if self.use_wandb:
            import wandb

            wandb.watch(self.model)

        # Setup optimizer
        self.optimizer = get_torch_optimizer(
            self.optimizer_name, self.optimizer_hparams, self.model.parameters()
        )

        # Setup learning rate scheduler
        self.scheduler = get_torch_scheduler(
            self.scheduler_name, self.scheduler_hparams, self.optimizer
        )

        # Log the train and validation loss each epoch
        self.loss_curve_ = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kld_loss": [],
            "valid_loss": [],
            "valid_recon_loss": [],
            "valid_kld_loss": [],
        }

    def fit(
        self,
        X: np.ndarray,
        scalars: Dict[str, np.ndarray] = {},
        output_path: PathLike = "./",
        checkpoint: Optional[PathLike] = None,
    ):
        r"""Trains the autoencoder on the input data :obj:`X`.

        Parameters
        ----------
        X : np.ndarray
            Input contact matrices in sparse COO format of shape (N,)
            where N is the number of data examples, and the empty dimension
            is ragged. The row and column index vectors should be contatenated
            and the values are assumed to be 1 and don't need to be explcitly
            passed.
        scalars : Dict[str, np.ndarray], default={}
            Dictionary of scalar arrays. For instance, the root mean squared
            deviation (RMSD) for each feature vector can be passed via
            :obj:`{"rmsd": np.array(...)}`. The dimension of each scalar array
            should match the number of input feature vectors N.
        output_path : PathLike, default="./"
            Path to write training results to. Makes an :obj:`output_path/checkpoints`
            folder to save model checkpoint files, and :obj:`output_path/plots` folder
            to store latent space visualizations.
        checkpoint : Optional[PathLike], default=None
            Path to a specific model checkpoint file to restore training.

        Raises
        ------
        TypeError
            If :obj:`scalars` is not type dict. A common error is to pass
            :obj:`output_path` as the second argument.
        NotImplementedError
            If using a learning rate scheduler other than :obj:`ReduceLROnPlateau`,
            a step function will need to be implemented.
        """
        if not isinstance(scalars, dict):
            raise TypeError(
                "scalars should be of type dict. A common error"
                " is to pass output_path as the second argument."
            )

        from mdlearn.data.datasets.contact_map import ContactMapDataset
        from mdlearn.data.utils import train_valid_split
        from mdlearn.utils import log_checkpoint
        from mdlearn.visualize import log_latent_visualization

        if self.use_wandb:
            import wandb

        exist_ok = (checkpoint is not None) or self.use_wandb
        output_path, checkpoint_path, plot_path = self._make_output_dir(
            output_path, exist_ok
        )

        # Set available number of cores
        self._set_num_threads()

        # Load training and validation data
        dataset = ContactMapDataset(X, self.input_shape, scalars)
        train_loader, valid_loader = train_valid_split(
            dataset,
            self.split_pct,
            self.split_method,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_data_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            pin_memory=not self.in_gpu_memory,
        )
        self.scalar_dset_names = list(scalars.keys())

        # Optionally resume training from a checkpoint
        start_epoch = self._resume_training(checkpoint)

        # Start training
        for epoch in range(start_epoch, self.epochs + 1):
            # Training
            self.model.train()
            avg_train_loss, avg_train_recon_loss, avg_train_kld_loss = self._train(
                train_loader
            )

            if self.verbose:
                print(
                    "====> Epoch: {} Train:\tAvg loss: {:.4f}\tAvg recon: {:.4f}\tAvg kld loss: {:.4f}".format(
                        epoch, avg_train_loss, avg_train_recon_loss, avg_train_kld_loss
                    )
                )

            # Validation
            self.model.eval()
            with torch.no_grad():
                (
                    avg_valid_loss,
                    avg_valid_recon_loss,
                    avg_valid_kld_loss,
                    z,
                    paints,
                ) = self._validate(valid_loader)

            if self.verbose:
                print(
                    "====> Epoch: {} Valid:\tAvg loss: {:.4f}\tAvg recon: {:.4f}\tAvg kld loss: {:.4f}".format(
                        epoch, avg_valid_loss, avg_valid_recon_loss, avg_valid_kld_loss
                    )
                )

            # Step the learning rate scheduler
            self.step_scheduler(epoch, avg_train_loss, avg_valid_loss)

            # Log a model checkpoint file
            if epoch % self.checkpoint_log_every == 0:
                log_checkpoint(
                    checkpoint_path / f"checkpoint-epoch-{epoch}.pt",
                    epoch,
                    self.model,
                    {"optimizer": self.optimizer},
                    self.scheduler,
                )

            if self.use_wandb:
                metrics = {
                    "train_loss": avg_train_loss,
                    "train_recon_loss": avg_train_recon_loss,
                    "train_kld_loss": avg_train_kld_loss,
                    "valid_loss": avg_valid_loss,
                    "valid_recon_loss": avg_valid_recon_loss,
                    "valid_kld_loss": avg_valid_kld_loss,
                }

            # Log a visualization of the latent space
            if (self.plot_method is not None) and (epoch % self.plot_log_every == 0):
                htmls = log_latent_visualization(
                    z,
                    paints,
                    plot_path,
                    epoch,
                    self.plot_n_samples,
                    self.plot_method,
                )
                if self.use_wandb:
                    # Optionally, log visualizations to wandb
                    for name, html in htmls.items():
                        metrics[name] = wandb.Html(html, inject=False)  # noqa

            if self.use_wandb:
                wandb.log(metrics)  # noqa

            # Save the losses
            self.loss_curve_["train_loss"].append(avg_train_loss)
            self.loss_curve_["train_recon_loss"].append(avg_train_recon_loss)
            self.loss_curve_["train_kld_loss"].append(avg_train_kld_loss)
            self.loss_curve_["valid_loss"].append(avg_valid_loss)
            self.loss_curve_["valid_recon_loss"].append(avg_valid_recon_loss)
            self.loss_curve_["valid_kld_loss"].append(avg_valid_kld_loss)

    def predict(
        self,
        X: np.ndarray,
        inference_batch_size: int = 128,
        checkpoint: Optional[PathLike] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""Predict using the LinearAE

        Parameters
        ----------
        X : np.ndarray
            The input data to predict on.
        inference_batch_size : int, default=512
            The batch size for inference.
        checkpoint : Optional[PathLike], default=None
            Path to a specific model checkpoint file.

        Returns
        -------
        Tuple[np.ndarray, float, float, float, float]
            The :obj:`z` latent vectors corresponding to the
            input data :obj:`X` and the average losses [total,
            reconstruction, KL-divergence]
        """
        from mdlearn.data.datasets.contact_map import ContactMapDataset

        dataset = ContactMapDataset(X, self.input_shape)
        data_loader = DataLoader(
            dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            pin_memory=not self.in_gpu_memory,
        )

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        # Make copy of class state incase of failure during inference
        tmp = self.scalar_dset_names.copy()
        self.model.eval()
        with torch.no_grad():
            try:
                # Set to empty list to avoid storage of paint scalars
                # that are not convenient to pass to the predict function.
                self.scalar_dset_names = []
                (
                    avg_valid_loss,
                    avg_valid_recon_loss,
                    avg_valid_kld_loss,
                    latent_vectors,
                    _,
                ) = self._validate(data_loader)
                # Restore class state
                self.scalar_dset_names = tmp
                return (
                    latent_vectors,
                    avg_valid_loss,
                    avg_valid_recon_loss,
                    avg_valid_kld_loss,
                )
            except Exception as e:
                # Restore class state incase of failure
                self.scalar_dset_names = tmp
                raise e

    def _train(self, train_loader) -> Tuple[float, float, float]:
        avg_loss, avg_recon_loss, avg_kld_loss = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_loader):

            if i / len(train_loader) > self.train_subsample_pct:
                break  # Early stop for sweeps

            x = batch["X"].to(self.device, non_blocking=True)

            # Forward pass
            _, recon_x = self.model(x)
            kld_loss = self.model.kld_loss()
            recon_loss = self.model.recon_loss(x, recon_x)
            loss = self.lambda_rec * recon_loss + kld_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_max_norm
            )
            self.optimizer.step()
            # Collect loss
            avg_loss += loss.item()
            avg_recon_loss += recon_loss.item()
            avg_kld_loss += kld_loss.item()

        avg_loss /= len(train_loader)
        avg_recon_loss /= len(train_loader)
        avg_kld_loss /= len(train_loader)

        return avg_loss, avg_recon_loss, avg_kld_loss

    def _validate(
        self, valid_loader
    ) -> Tuple[float, float, float, np.ndarray, Dict[str, np.ndarray]]:
        avg_loss, avg_recon_loss, avg_kld_loss = 0.0, 0.0, 0.0
        paints = defaultdict(list)
        latent_vectors = []
        for i, batch in enumerate(valid_loader):

            if i / len(valid_loader) > self.valid_subsample_pct:
                break  # Early stop for sweeps

            x = batch["X"].to(self.device, non_blocking=True)

            # Forward pass
            z, recon_x = self.model(x)
            kld_loss = self.model.kld_loss()
            recon_loss = self.model.recon_loss(x, recon_x)
            loss = self.lambda_rec * recon_loss + kld_loss

            # Collect loss
            avg_loss += loss.item()
            avg_recon_loss += recon_loss.item()
            avg_kld_loss += kld_loss.item()

            # Collect latent vectors for visualization
            latent_vectors.append(z.cpu().numpy())
            for name in self.scalar_dset_names:
                paints[name].append(batch[name].cpu().numpy())

        avg_loss /= len(valid_loader)
        avg_recon_loss /= len(valid_loader)
        avg_kld_loss /= len(valid_loader)
        # Group latent vectors and paints
        latent_vectors = np.concatenate(latent_vectors)
        paints = {name: np.concatenate(scalar) for name, scalar in paints.items()}

        return avg_loss, avg_recon_loss, avg_kld_loss, latent_vectors, paints
