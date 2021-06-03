import torch
from typing import List
from mdlearn.nn.models.aae import AAE
from mdlearn.nn.modules.conv1d_encoder import Conv1dEncoder
from mdlearn.nn.modules.conv1d_decoder import Conv1dDecoder
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

        encoder = Conv1dEncoder(
            num_points,
            num_features,
            latent_dim,
            encoder_bias,
            encoder_relu_slope,
            encoder_filters,
            encoder_kernels,
        )

        decoder = Conv1dDecoder(
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

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x

    def discriminate(self, z: torch.Tensor):
        logit = self.discriminator(z)
        return logit
