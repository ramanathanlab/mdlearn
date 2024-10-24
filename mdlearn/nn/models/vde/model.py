""".. warning:: VDE models are still under development, use with caution!"""
from __future__ import annotations

import torch
from torch import nn

from mdlearn.nn.models.vae import VAE


class VDE(VAE):
    """Variational dynamics encoder base class module
    based off the `"Variational Encoding of Complex Dynamics"
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7398762/>`_ paper
    Inherits from :obj:`mdlearn.nn.models.vae.VAE`.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder, decoder)

    def ac_loss(
        self,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
    ) -> torch.Tensor:
        r"""Negative autocorrelation loss.

        Parameters
        ----------
        z_t : torch.Tensor
            :math:`z_t`-latent vector.
        z_t_tau : torch.Tensor
            :math:`z_{t+\tau}`-latent vector.

        Returns
        -------
        torch.Tensor
            Negative autocorrelation loss between :math:`z_t` and :math:`z_{t+\tau}`.
        """
        z_t_mean_diff = z_t - torch.mean(z_t)
        z_t_lag_mean_diff = z_t_tau - torch.mean(z_t_tau)

        # TODO: original code base divides by torch.norm(z, 2) instead of std
        autocorrelation = torch.mean(z_t_mean_diff * z_t_lag_mean_diff) / (
            torch.std(z_t) * torch.std(z_t_tau)
        )

        return -autocorrelation
