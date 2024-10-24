from __future__ import annotations

import torch
from torch import nn

from mdlearn.nn.models.vae import VAE  # type: ignore[attr-defined]
from mdlearn.nn.utils import reset


class ChamferLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        _, num_points_x, _ = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        dtype = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = (
            xx[:, diag_ind_x, diag_ind_x]
            .unsqueeze(1)
            .expand_as(zz.transpose(2, 1))
        )  # type: ignore[index]
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)  # type: ignore[index]
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


class AAE(VAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        discriminator: nn.Module,
    ) -> None:
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        # AE resets encoder and decoder on init
        reset(self.discriminator)

    def reset_parameters(self) -> None:
        """Reset encoder, decoder and discriminator parameters."""
        super().reset_parameters()
        reset(self.discriminator)

    def discriminate(self, *args, **kwargs) -> torch.Tensor:
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
