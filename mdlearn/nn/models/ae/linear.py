import torch
from typing import List, Tuple
from torch.nn import functional as F
from mdlearn.nn.models.ae import AE
from mdlearn.nn.modules.dense_net import DenseNet


class LinearAE(AE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        neurons: List[int] = [128],
        bias: bool = True,
        relu_slope: float = 0.0,
        inplace_activation: bool = False,
    ):
        """A symmetric autoencoder with all linear layers.

        Parameters
        ----------
        input_dim : int
            Dimension of input tensor (should be flattened).
        latent_dim: int
            Dimension of the latent space.
        neurons : List[int], optional
            Linear layers :obj:`in_features`, by default [128].
        bias : bool, optional
            Use a bias term in the Linear layers, by default True.
        relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0.
        inplace_activation : bool, optional
            Sets the inplace option for the activation function.
        """

        neurons = neurons.copy() + [latent_dim]
        encoder = DenseNet(input_dim, neurons, bias, relu_slope, inplace_activation)
        decoder_neurons = list(reversed(neurons))[1:] + [input_dim]
        decoder = DenseNet(
            neurons[-1], decoder_neurons, bias, relu_slope, inplace_activation
        )

        super().__init__(encoder, decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z = F.relu(z)
        recon_x = self.decode(z)
        return z, recon_x

    def recon_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        r"""Compute the MSE reconstruction loss between :obj:`x` and :obj:`recon_x`.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        recon_x : torch.Tensor
            The reconstruction of the input data :obj:`x`

        Returns
        -------
        torch.Tensor
            The reconstruction loss between :obj:`x` and :obj:`recon_x`.
        """
        return F.mse_loss(recon_x, x, reduction=reduction)
