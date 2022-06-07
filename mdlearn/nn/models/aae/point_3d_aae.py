"""Adversarial Autoencoder for 3D point cloud data (3dAAE)"""
from typing import List, Tuple

import torch

from mdlearn.nn.models.aae import AAE, ChamferLoss
from mdlearn.nn.modules.conv1d_encoder import Conv1dEncoder
from mdlearn.nn.modules.linear_decoder import LinearDecoder
from mdlearn.nn.modules.linear_discriminator import LinearDiscriminator


class AAE3d(AAE):
    def __init__(
        self,
        num_points: int,
        num_features: int = 0,
        latent_dim: int = 20,
        encoder_bias: bool = True,
        encoder_relu_slope: float = 0.0,
        encoder_filters: List[int] = [64, 128, 256, 256, 512],
        encoder_kernels: List[int] = [5, 5, 3, 1, 1],
        decoder_bias: bool = True,
        decoder_relu_slope: float = 0.0,
        decoder_affine_widths: List[int] = [64, 128, 512, 1024],
        discriminator_bias: bool = True,
        discriminator_relu_slope: float = 0.0,
        discriminator_affine_widths: List[int] = [512, 128, 64],
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input point cloud data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The :math:`z`-latent vector, and the :obj:`recon_x`
            reconstruction.
        """
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x

    def critic_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
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
        alpha = torch.rand(z.shape[0], 1).to(z.device)  # z.shape[0] is batch_size
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
        return gradient_penalty

    def recon_loss(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
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
            self._recon_loss(recon_x.permute(0, 2, 1), x.permute(0, 2, 1))
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
