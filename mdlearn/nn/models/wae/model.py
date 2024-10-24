from __future__ import annotations

import math

import torch

from mdlearn.nn.models.vae import VAE


class WAE(VAE):
    """Wasserstein autoencoder base class module.
    Inherits from :obj:`mdlearn.nn.models.vae.VAE`.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.__rf__ = {}

    def mmdrf_loss(
        self,
        z: torch.Tensor,
        sigma: float,
        kernel: str,
        rf_dim: int,
        rf_resample: bool,
    ) -> torch.Tensor:
        r"""Computes the loss :math:`|\mu_{real} - \mu_{fake}|_H`

        Parameters
        ----------
        z : torch.Tensor
            The :math:`z`-latent vector.
        sigma : float
            TODO
        kernel : str
            The type of kernel function to use.
        rf_dim : int
            Random features kernel dimension.
        rf_resample : bool
            Whether ot not to resample the random features.

        Returns
        -------
        torch.Tensor
            MMD RF loss.
        """
        z_prior = torch.randn_like(z)  # shape and device
        mu_real = self._compute_mmd_mean_rf(
            z,
            sigma,
            kernel,
            rf_dim,
            rf_resample,
        )
        mu_fake = self._compute_mmd_mean_rf(
            z_prior,
            sigma,
            kernel,
            rf_dim,
            rf_resample,
        )
        # TODO: try l2norm of difference
        loss = ((mu_real - mu_fake) ** 2).sum()
        return loss

    def _random_feature_approx(
        self,
        z: torch.Tensor,
        kernel: str,
        rf_dim: int,
        rf_resample: bool,
    ):
        """Random features approximation of gaussian kernel."""
        if kernel not in self.__rf__ or rf_resample:
            # Sample rf if it's the first time or we want to resample every time
            rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
            rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)
            self.__rf__ = {'gaussian': (rf_w, rf_b)}  # Cache rf approx
        else:
            rf_w, rf_b = self.__rf__['gaussian']
            if rf_w.shape != (z.shape[1], rf_dim):
                raise ValueError('Not expecting z dim or rf_dim to change')

        return rf_w, rf_b

    def _compute_mmd_mean_rf(
        self,
        z: torch.Tensor,
        sigma: float,
        kernel: str,
        rf_dim: int,
        rf_resample: bool = False,
    ) -> torch.Tensor:
        if kernel == 'gaussian':
            rf_w, rf_b = self._random_feature_approx(
                z,
                kernel,
                rf_dim,
                rf_resample,
            )
            z_rf = self._compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
        else:
            raise ValueError(f'Invalid kernel {kernel} for rf regularization')

        mu_rf = z_rf.mean(0, keepdim=False)
        return mu_rf

    def _compute_gaussian_rf(
        self,
        z: torch.Tensor,
        rf_w: torch.Tensor,
        rf_b: torch.Tensor,
        sigma: float,
        rf_dim: int,
    ) -> torch.Tensor:
        z_emb = (z @ rf_w) / sigma + rf_b
        z_emb = torch.cos(z_emb) * (2.0 / rf_dim) ** 0.5
        return z_emb
