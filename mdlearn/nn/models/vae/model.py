from __future__ import annotations

from typing import Optional

import torch

from mdlearn.nn.models.ae import AE

MAX_LOGSTD = 10

# TODO: implement switch between logstd vs logvar


class VAE(AE):
    """Variational autoencoder base class module.
    Inherits from :obj:`mdlearn.nn.models.ae.AE`.
    """

    def __init__(self, encoder, decoder):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            The encoder module.
        decoder : torch.nn.Module
            The decoder module.
        """
        super().__init__(encoder, decoder)

    def reparametrize(
        self,
        mu: torch.Tensor,
        logstd: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for :obj:`mu` and :obj:`logstd`.

        Parameters
        ----------
        mu : torch.Tensor
            First encoder output.
        logstd : torch.Tensor
            Second encoder output.

        Returns
        -------
        torch.Tensor :
            If training, return the reparametrized output.
            Otherwise, return :obj:`mu`.
        """
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> torch.Tensor:
        """Encoder forward pass and reparameterization of mu and logstd.

        Parameters
        ----------
        *args
            Variable length encoder argument list.
        **kwargs
            Arbitrary encoder keyword arguments.

        Returns
        -------
        torch.Tensor :
            The encoded :math:`z`-latent batch tensor.

        Notes
        -----
        Clamps logstd using a max logstd of 10.
        """
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kld_loss(
        self,
        mu: Optional[torch.Tensor] = None,
        logstd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Computes the KLD loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Parameters
        ----------
        mu : torch.Tensor, optional
            The latent space for :math:`\mu`. If set to :obj:`None`,
            uses the last computation of :math:`\mu`.
        logstd : torch.Tensor, optional
            The latent space for :math:`\log\sigma`. If set to :obj:`None`,
            uses the last computation of :math:`\log\sigma^2`.

        Returns
        -------
        torch.Tensor :
            KL divergence loss given :obj:`mu` and :obj:`logstd`.

        Notes
        -----
        Clamps logstd using a max logstd of 10.
        """
        mu = self.__mu__ if mu is None else mu
        logstd = (
            self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        )
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1),
        )
