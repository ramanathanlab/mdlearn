import torch
from torch import nn
from torch.nn import functional as F

from mdlearn.nn.utils import reset


class AE(nn.Module):
    """Autoencoder base class module."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            The encoder module.
        decoder : torch.nn.Module
            The decoder module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        AE.reset_parameters(self)

    def reset_parameters(self):
        """Reset encoder and decoder parameters."""
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        """Encoder forward pass."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decoder forward pass."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
        r"""Compute the reconstruction loss between :obj:`x` and :obj:`recon_x`.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        recon_x : torch.Tensor
            The reconstruction of the input data :obj:`x`.

        Returns
        -------
        torch.Tensor
            The reconstruction loss between :obj:`x` and :obj:`recon_x`.
        """
        return F.binary_cross_entropy(recon_x, x, reduction="sum")
