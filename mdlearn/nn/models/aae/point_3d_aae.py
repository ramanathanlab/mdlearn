"""Adversarial Autoencoder for 3D point cloud data (3dAAE)"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader
from tqdm import tqdm

from mdlearn.data.datasets.point_cloud import PointCloudDatasetInMemory
from mdlearn.data.utils import train_valid_split
from mdlearn.nn.models.aae import AAE  # type: ignore[attr-defined]
from mdlearn.nn.models.aae import ChamferLoss  # type: ignore[attr-defined]
from mdlearn.nn.modules.conv1d_encoder import Conv1dEncoder
from mdlearn.nn.modules.linear_decoder import LinearDecoder
from mdlearn.nn.modules.linear_discriminator import LinearDiscriminator
from mdlearn.nn.utils import PathLike  # type: ignore[attr-defined]
from mdlearn.nn.utils import Trainer  # type: ignore[attr-defined]
from mdlearn.utils import get_torch_optimizer
from mdlearn.utils import log_checkpoint
from mdlearn.utils import resume_checkpoint
from mdlearn.visualize import log_latent_visualization


class AAE3d(AAE):
    def __init__(
        self,
        num_points: int,
        num_features: int = 0,
        latent_dim: int = 20,
        encoder_bias: bool = True,
        encoder_relu_slope: float = 0.0,
        encoder_filters: list[int] = [64, 128, 256, 256, 512],
        encoder_kernels: list[int] = [5, 5, 3, 1, 1],
        decoder_bias: bool = True,
        decoder_relu_slope: float = 0.0,
        decoder_affine_widths: list[int] = [64, 128, 512, 1024],
        discriminator_bias: bool = True,
        discriminator_relu_slope: float = 0.0,
        discriminator_affine_widths: list[int] = [512, 128, 64],
    ):
        """Adversarial Autoencoder module for point cloud data from
        the `"Adversarial Autoencoders for Compact Representations of 3D Point Clouds"
        <https://arxiv.org/abs/1811.07605>`_ paper and adapted to work on
        atomic coordinate data in the
        `"AI-Driven Multiscale Simulations Illuminate Mechanisms of SARS-CoV-2 Spike Dynamics"
        <https://www.biorxiv.org/content/10.1101/2020.11.19.390187v1.abstract>`_ paper.
        Inherits from :obj:`mdlearn.nn.models.aae.AAE`.

        Parameters
        ----------
        num_points : int
            Number of input points in point cloud.
        num_features : int, optional
            Number of scalar features per point in addition to 3D
            coordinates, by default 0
        latent_dim : int, optional
            Latent dimension of the encoder, by default 20
        encoder_bias : bool, optional
            Use a bias term in the encoder Conv1d layers, by default True.
        encoder_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the encoder
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        encoder_filters : List[int], optional
            Encoder Conv1d filter sizes, by default [64, 128, 256, 256, 512]
        encoder_kernels : List[int], optional
            Encoder Conv1d kernel sizes, by default [5, 5, 3, 1, 1]
        decoder_bias : bool, optional
            Use a bias term in the decoder Linear layers, by default True
        decoder_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the decoder
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        decoder_affine_widths : List[int], optional
            Decoder Linear layers :obj:`in_features`, by default [64, 128, 512, 1024]
        discriminator_bias : bool, optional
            Use a bias term in the discriminator Linear layers, by default True.
        discriminator_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the discriminator
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        discriminator_affine_widths : List[int], optional
            Discriminator Linear layers :obj:`in_features`, by default [512, 128, 64]
        """
        encoder = Conv1dEncoder(
            num_points,
            num_features,
            latent_dim,
            encoder_bias,
            encoder_relu_slope,
            encoder_filters,
            encoder_kernels,
        )

        decoder = LinearDecoder(
            num_points,
            num_features,
            latent_dim,
            decoder_bias,
            decoder_relu_slope,
            decoder_affine_widths,
        )

        discriminator = LinearDiscriminator(
            latent_dim,
            discriminator_bias,
            discriminator_relu_slope,
            discriminator_affine_widths,
        )

        super().__init__(encoder, decoder, discriminator)

        self._recon_loss = ChamferLoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input point cloud data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The :math:`z`-latent vector, and the :obj:`recon_x`
            reconstruction.
        """
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x

    def critic_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Classification loss (critic) function.

        Parameters
        ----------
        real_logits : torch.Tensor
            Discriminator output logits from prior distribution.
        fake_logits : torch.Tensor
            Discriminator output logits from encoded latent vectors.

        Returns
        -------
        torch.Tensor
            Classification loss i.e. the difference between logit means.
        """
        return torch.mean(fake_logits) - torch.mean(real_logits)

    def gp_loss(self, noise: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Gradient penalty loss function.

        Parameters
        ----------
        noise : [type]
            Random noise sampled from prior distribution.
        z : [type]
            Encoded latent vectors.

        Returns
        -------
        torch.Tensor
            The gradient penalty loss.
        """
        alpha = torch.rand(z.shape[0], 1).to(
            z.device,
        )  # z.shape[0] is batch_size
        interpolates = noise + alpha * (z - noise)
        disc_interpolates = self.discriminate(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(z.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        slopes = torch.sqrt(torch.sum(gradients**2, dim=1))
        gradient_penalty = ((slopes - 1) ** 2).mean()
        return gradient_penalty  # type: ignore[no-any-return]

    def recon_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction loss using ChamferLoss.

        Parameters
        ----------
        x : torch.Tensor
            The original input tensor.
        recon_x : torch.Tensor
            The reconstructed output tensor.

        Returns
        -------
        torch.Tensor
            Reconstruction loss measured by Chamfer distance.
        """
        # Here we need input shape (batch_size, num_points, points_dim)
        return torch.mean(
            self._recon_loss(recon_x.permute(0, 2, 1), x.permute(0, 2, 1)),
        )

    def decoder_loss(self, fake_logit: torch.Tensor) -> torch.Tensor:
        """Decoder/Generator loss.

        Parameters
        ----------
        fake_logit : torch.Tensor
            Output of discriminator.

        Returns
        -------
        torch.Tensor
            Negative mean of the fake logits.
        """
        return -torch.mean(fake_logit)


class AAE3dTrainer(Trainer):
    """Trainer class to fit a 3D Adversarial Autoencoder (3dAAE) model."""

    def __init__(
        self,
        num_points: int,
        num_features: int = 0,
        latent_dim: int = 20,
        encoder_bias: bool = True,
        encoder_relu_slope: float = 0.0,
        encoder_filters: list[int] = [64, 128, 256, 256, 512],
        encoder_kernels: list[int] = [5, 5, 3, 1, 1],
        decoder_bias: bool = True,
        decoder_relu_slope: float = 0.0,
        decoder_affine_widths: list[int] = [64, 128, 512, 1024],
        discriminator_bias: bool = True,
        discriminator_relu_slope: float = 0.0,
        discriminator_affine_widths: list[int] = [512, 128, 64],
        disc_optimizer_name: str = 'Adam',
        disc_optimizer_hparams: dict[str, Any] = {'lr': 0.0001},
        ae_optimizer_name: str = 'Adam',
        ae_optimizer_hparams: dict[str, Any] = {'lr': 0.0001},
        cms_transform: bool = False,
        noise_mu: float = 0.0,
        noise_std: float = 0.2,
        lambda_gp: float = 10.0,
        lambda_rec: float = 1.0,
        init_weights: PathLike | None = None,
        seed: int = np.random.default_rng().integers(2**31 - 1, dtype=int),
        num_data_workers: int = 0,
        prefetch_factor: int = 2,
        split_pct: float = 0.8,
        split_method: str = 'random',
        batch_size: int = 64,
        inference_batch_size: int = 64,
        shuffle: bool = True,
        device: str = 'cpu',
        epochs: int = 100,
        verbose: bool = False,
        clip_grad_max_norm: float = 10.0,
        checkpoint_log_every: int = 10,
        plot_log_every: int = 10,
        plot_n_samples: int = 10000,
        plot_method: str | None = None,
        train_subsample_pct: float = 1.0,
        valid_subsample_pct: float = 1.0,
        use_wandb: bool = False,
    ):
        r"""
        Parameters
        ----------
        num_points : int
            Number of input points in point cloud.
        num_features : int, optional
            Number of scalar features per point in addition to 3D
            coordinates, by default 0
        latent_dim : int, optional
            Latent dimension of the encoder, by default 20
        encoder_bias : bool, optional
            Use a bias term in the encoder Conv1d layers, by default True.
        encoder_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the encoder
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        encoder_filters : list[int], optional
            Encoder Conv1d filter sizes, by default [64, 128, 256, 256, 512]
        encoder_kernels : list[int], optional
            Encoder Conv1d kernel sizes, by default [5, 5, 3, 1, 1]
        decoder_bias : bool, optional
            Use a bias term in the decoder Linear layers, by default True
        decoder_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the decoder
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        decoder_affine_widths : list[int], optional
            Decoder Linear layers :obj:`in_features`, by default [64, 128, 512, 1024]
        discriminator_bias : bool, optional
            Use a bias term in the discriminator Linear layers, by default True.
        discriminator_relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation in the discriminator
            with :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0
        discriminator_affine_widths : list[int], optional
            Discriminator Linear layers :obj:`in_features`, by default [512, 128, 64]
        disc_optimizer_name : str, default="Adam"
            Name of the PyTorch optimizer to use for the discriminator.
        disc_optimizer_hparams : dict[str, Any], default={"lr": 0.0001}
            Dictionary of hyperparameters to pass to the chosen PyTorch optimizer.
        ae_optimizer_name : str, default="Adam"
            Name of the PyTorch optimizer to use for the autoencoder.
        ae_optimizer_hparams : dict[str, Any], default={"lr": 0.0001}
            Dictionary of hyperparameters to pass to the chosen PyTorch optimizer.
        cms_transform : bool, default=False
            If True, will subtract center of mass from batch and shift and scale
            batch by the full dataset statistics.
        noise_mu : float, default=0.0
            Mean of the prior distribution.
        noise_std : float, default=0.2
            Standard deviation of the prior distribution.
        lambda_gp : float, default=10.0
            Factor to scale gradient penalty loss by during training such that
            :obj:`loss = critic_loss + lambda_gp * gp_loss`.
        lambda_rec : float, default=1.0
            Factor to scale reconstruction loss by during training such that
            :obj:`loss = decoder_loss + lambda_rec * recon_loss`.
        init_weights : PathLike | None, default=None
            Path to a specific model checkpoint file to load model weights for
            initialization (does not load optimizer states).
        seed : int, default=np.random.default_rng().integers(2**31 - 1, dtype=int)
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
        batch_size : int, default=64
            Mini-batch size for training.
        inference_batch_size : int, default=64
            Mini-batch size for inference.
        shuffle : bool, default=True
            Whether to shuffle training data or not.
        device : str, default="cpu"
            Specify training hardware either :obj:`cpu` or :obj:`cuda` for GPU devices.
        epochs : int, default=100
            Number of epochs to train for.
        verbose : bool, default False
            If True, will print training and validation loss at each epoch.
        clip_grad_max_norm : float, default=10.0
            Max norm of the gradients for gradient clipping for more information
            see: :obj:`torch.nn.utils.clip_grad_norm_` documentation. Note that
            this parameter is ignored in the AAE3d model.
        checkpoint_log_every : int, default=10
            Epoch interval to log a checkpoint file containing the model
            weights, optimizer, and scheduler parameters.
        plot_log_every : int, default=10
            Epoch interval to log a visualization plot of the latent space.
        plot_n_samples : int, default=10000
            Number of validation samples to use for plotting.
        plot_method : str | None, default=None
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
            If True, will log results to wandb. Metric keys include `train_disc_loss`,
            `train_ae_loss`, and `valid_recon_loss`.

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
        """
        super().__init__(
            seed,
            False,
            num_data_workers,
            prefetch_factor,
            split_pct,
            split_method,
            batch_size,
            inference_batch_size,
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

        self.latent_dim = latent_dim
        self.cms_transform = cms_transform
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.lambda_gp = lambda_gp
        self.lambda_rec = lambda_rec

        self.model = AAE3d(
            num_points=num_points,
            num_features=num_features,
            latent_dim=latent_dim,
            encoder_bias=encoder_bias,
            encoder_relu_slope=encoder_relu_slope,
            encoder_filters=encoder_filters,
            encoder_kernels=encoder_kernels,
            decoder_bias=decoder_bias,
            decoder_relu_slope=decoder_relu_slope,
            decoder_affine_widths=decoder_affine_widths,
            discriminator_bias=discriminator_bias,
            discriminator_relu_slope=discriminator_relu_slope,
            discriminator_affine_widths=discriminator_affine_widths,
        )

        # Optionally initialize model with pre-trained weights
        if init_weights is not None:
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model from {init_weights}')

        # Set model device
        self.model.to(self.device)

        # Setup wandb
        if self.use_wandb:
            import wandb

            wandb.watch(self.model)

        # Setup optimizers
        self.disc_optimizer = get_torch_optimizer(
            disc_optimizer_name,
            disc_optimizer_hparams,
            self.model.discriminator.parameters(),
        )
        self.ae_optimizer = get_torch_optimizer(
            ae_optimizer_name,
            ae_optimizer_hparams,
            itertools.chain(
                self.model.encoder.parameters(),
                self.model.decoder.parameters(),
            ),
        )

        # Log the train and validation loss each epoch
        self.loss_curve_: dict[str, list[float]] = {
            'train_disc_loss': [],
            'train_ae_loss': [],
            'valid_recon_loss': [],
        }

    def _load_checkpoint(self, checkpoint: PathLike) -> int:
        """Load parameters from a checkpoint file.

        Parameters
        ----------
        checkpoint : PathLike
            PyTorch checkpoint file (.pt) to load model, optimizer
            and scheduler parameters from.

        Returns
        -------
        int
            Epoch where training left off.
        """
        # NOTE: This function needed to be overriden to handle the
        #       additional optimizer for the discriminator.
        return resume_checkpoint(
            checkpoint,
            self.model,
            {
                'disc_optimizer': self.disc_optimizer,
                'ae_optimizer': self.ae_optimizer,
            },
            self.scheduler,
        )

    def fit(
        self,
        X: ArrayLike,
        scalars: dict[str, ArrayLike] = {},
        output_path: PathLike = './',
        checkpoint: PathLike | None = None,
    ) -> None:
        r"""Trains the autoencoder on the input data :obj:`X`.

        Parameters
        ----------
        X : ArrayLike
            Input point cloud data of shape (N, 3, num_points) where N is the
            number of data examples, 3 is the x, y, z coordinates of each point,
            and num_points is the number of points in the point cloud (e.g. number
            of residues in a protein structure).
        scalars : Dict[str, ArrayLike], default={}
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
                'scalars should be of type dict. A common error'
                ' is to pass output_path as the second argument.',
            )

        if self.use_wandb:
            import wandb

        exist_ok = (checkpoint is not None) or self.use_wandb
        output_path, checkpoint_path, plot_path = self._make_output_dir(
            output_path,
            exist_ok,
        )

        # Set available number of cores
        self._set_num_threads()

        # Load training and validation data
        dataset = PointCloudDatasetInMemory(
            data=X,
            scalars=scalars,
            cms_transform=self.cms_transform,
        )
        train_loader, valid_loader = train_valid_split(
            dataset=dataset,
            split_pct=self.split_pct,
            method=self.split_method,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_data_workers,
            prefetch_factor=(
                None if self.num_data_workers == 0 else self.prefetch_factor
            ),
            persistent_workers=self.persistent_workers,
            drop_last=True,
            pin_memory=not self.in_gpu_memory,
        )
        self.scalar_dset_names = list(scalars.keys())

        # Optionally resume training from a checkpoint
        start_epoch = self._resume_training(checkpoint)

        # Start training
        for epoch in range(start_epoch, self.epochs):
            train_start = time.time()
            # Training
            self.model.train()
            avg_train_disc_loss, avg_train_ae_loss = self._train(train_loader)

            print(
                f'====> Epoch: {epoch} Train:\tAvg Disc loss: {avg_train_disc_loss:.4f}\tAvg AE loss: {avg_train_ae_loss:.4f}\tTime: {time.time() - train_start:.4f}',
            )

            valid_start = time.time()
            # Validation
            self.model.eval()
            with torch.no_grad():
                avg_valid_recon_loss, latent_vectors, scalars = self._validate(
                    valid_loader,
                )

            print(
                f'====> Epoch: {epoch} Valid:\tAvg recon loss: {avg_valid_recon_loss:.4f}\tTime: {time.time() - valid_start:.4f}\n',
            )

            print(f'Total time: {time.time() - train_start:.4f}')

            if self.use_wandb:
                metrics: dict[str, Any] = {
                    'train_disc_loss': avg_train_disc_loss,
                    'train_ae_loss': avg_train_ae_loss,
                    'valid_recon_loss': avg_valid_recon_loss,
                }

            # Visualize latent space
            if (
                self.plot_method is not None
                and epoch % self.plot_log_every == 0
            ):
                html_strings = log_latent_visualization(
                    latent_vectors,
                    scalars,
                    plot_path,
                    epoch,
                    self.plot_n_samples,
                    self.plot_method,
                )
                if self.use_wandb:
                    for name, html_string in html_strings.items():
                        metrics[name] = wandb.Html(html_string, inject=False)

            if self.use_wandb:
                wandb.log(metrics)

            if epoch % self.checkpoint_log_every == 0:
                log_checkpoint(
                    checkpoint_path / f'checkpoint-epoch-{epoch}.pt',
                    epoch,
                    self.model,
                    {
                        'disc_optimizer': self.disc_optimizer,
                        'ae_optimizer': self.ae_optimizer,
                    },
                )

            # Save the losses
            self.loss_curve_['train_disc_loss'].append(avg_train_disc_loss)
            self.loss_curve_['train_ae_loss'].append(avg_train_ae_loss)
            self.loss_curve_['valid_recon_loss'].append(avg_valid_recon_loss)

    def predict(
        self,
        X: ArrayLike,
        inference_batch_size: int | None = None,
        checkpoint: PathLike | None = None,
    ) -> tuple[ArrayLike, float]:
        r"""Predict using the LinearAE

        Parameters
        ----------
        X : ArrayLike
            Input point cloud data of shape (N, 3, num_points) where N is the
            number of data examples, 3 is the x, y, z coordinates of each point,
            and num_points is the number of points in the point cloud (e.g. number
            of residues in a protein structure).
        inference_batch_size : int, default=None
            The batch size for inference (if None uses the
            value specified during Trainer construction).
        checkpoint : PathLike | None, default=None
            Path to a specific model checkpoint file.

        Returns
        -------
        tuple[ArrayLike, float]
            The :obj:`z` latent vectors corresponding to the
            input data :obj:`X` and the average reconstruction loss.
        """
        # Fall back to default batch size
        if inference_batch_size is None:
            inference_batch_size = self.inference_batch_size

        # Setup the dataset and data loader
        dataset = PointCloudDatasetInMemory(
            data=X,
            cms_transform=self.cms_transform,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            prefetch_factor=(
                None if self.num_data_workers == 0 else self.prefetch_factor
            ),
            persistent_workers=self.persistent_workers,
            drop_last=False,
            pin_memory=not self.in_gpu_memory,
        )

        # Load model checkpoint
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
                avg_valid_recon_loss, latent_vectors, _ = self._validate(
                    data_loader,
                )
                # Restore class state
                self.scalar_dset_names = tmp
                return latent_vectors, avg_valid_recon_loss
            except Exception as e:
                # Restore class state incase of failure
                self.scalar_dset_names = tmp
                raise e

    def _train(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> tuple[float, float]:
        avg_disc_loss, avg_ae_loss = 0.0, 0.0
        # Create prior noise buffer array
        noise = torch.FloatTensor(self.batch_size, self.latent_dim).to(
            self.device,
        )
        for batch in tqdm(train_loader):
            x = batch['X'].to(self.device, non_blocking=True)

            # Encoder/Discriminator forward
            # Get latent vectors
            z = self.model.encode(x)
            # Get prior noise
            noise.normal_(mean=self.noise_mu, std=self.noise_std)
            # Get discriminator logits
            real_logits = self.model.discriminate(noise)
            fake_logits = self.model.discriminate(z)
            # Discriminator loss
            critic_loss = self.model.critic_loss(real_logits, fake_logits)
            gp_loss = self.model.gp_loss(noise, z)
            disc_loss = critic_loss + self.lambda_gp * gp_loss

            # Discriminator backward
            self.disc_optimizer.zero_grad()
            self.model.discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
            self.disc_optimizer.step()

            # Decoder forward
            recon_x = self.model.decode(z)
            recon_loss = self.model.recon_loss(x, recon_x)
            # Discriminator forward
            fake_logit = self.model.discriminate(z)
            decoder_loss = self.model.decoder_loss(fake_logit)
            ae_loss = decoder_loss + self.lambda_rec * recon_loss

            # AE backward
            self.ae_optimizer.zero_grad()
            self.model.decoder.zero_grad()
            self.model.encoder.zero_grad()
            ae_loss.backward()  # type: ignore[no-untyped-call]

            # Collect loss
            avg_disc_loss += disc_loss.item()
            avg_ae_loss += ae_loss.item()

        avg_disc_loss /= len(train_loader)
        avg_ae_loss /= len(train_loader)

        return avg_disc_loss, avg_ae_loss

    def _validate(
        self,
        valid_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> tuple[float, ArrayLike, dict[str, ArrayLike]]:
        scalars = defaultdict(list)
        latent_vectors = []
        avg_recon_loss = 0.0
        for i, batch in enumerate(valid_loader):
            if i / len(valid_loader) > self.valid_subsample_pct:
                break  # Early stop for sweeps

            x = batch['X'].to(self.device)
            z = self.model.encode(x)
            recon_x = self.model.decode(z)
            avg_recon_loss += self.model.recon_loss(x, recon_x).item()

            # Collect latent vectors for visualization
            latent_vectors.append(z.cpu().numpy())
            for name in self.scalar_dset_names:
                scalars[name].append(batch[name].cpu().numpy())

        avg_recon_loss /= len(valid_loader)
        latent_vectors = np.concatenate(latent_vectors)
        scalars = {
            name: np.concatenate(scalar) for name, scalar in scalars.items()
        }  # type: ignore[assignment]

        return avg_recon_loss, latent_vectors, scalars  # type: ignore[return-value]
