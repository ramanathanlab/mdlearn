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
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = ((slopes - 1) ** 2).mean()
        return gradient_penalty
