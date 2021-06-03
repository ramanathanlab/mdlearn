import torch.nn as nn
from mdlearn.nn.utils import reset
from mdlearn.nn.models.vae import VAE


class AAE(VAE):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        # AE resets encoder and decoder on init
        reset(self.discriminator)

    def reset_parameters(self):
        """Reset encoder, decoder and discriminator parameters."""
        super().reset_parameters()
        reset(self.discriminator)

    def discriminate(self, *args, **kwargs):
        """Discriminator forward pass.

        Parameters
        ----------
        *args
            Variable length discriminator argument list.
        **kwargs
            Arbitrary discriminator keyword arguments.

        Returns
        -------
        torch.Tensor :
            The discriminator output.
        """
        return self.discriminator(*args, **kwargs)
