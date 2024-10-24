from __future__ import annotations

import torch
from torch import nn

from mdlearn.nn.utils import get_activation


class LinearDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 20,
        bias: bool = True,
        relu_slope: float = 0.0,
        affine_widths: list[int] = [512, 128, 64],
    ):
        """LinearDiscriminator module.

        Parameters
        ----------
        latent_dim : int, optional
            Latent dimension of the decoder, by default 20.
        bias : bool, optional
            Use a bias term in the Linear layers, by default True.
        relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0.
        affine_widths : List[int], optional
            Linear layers :obj:`in_features`, by default [64, 128, 512, 1024].
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.bias = bias
        self.negative_slope = relu_slope
        self.affine_widths = affine_widths

        # Select activation
        self.activation_kwargs = {'inplace': True}
        if self.negative_slope > 0.0:
            self.activation = 'LeakyReLU'
            self.activation_kwargs['negative_slope'] = self.negative_slope
        else:
            self.activation = 'ReLU'

        self.model = nn.Sequential(*self._affine_layers())

    def forward(self, x: torch.Tensor):
        logit = self.model(x)
        return logit

    def _affine_layers(self) -> list[nn.Module]:
        """Compose affine layers.

        Returns
        -------
        list:
            Linear layers and activations.
        """
        layers = []

        in_features = self.latent_dim

        for width in self.affine_widths:
            layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=width,
                    bias=self.bias,
                ),
            )

            layers.append(
                get_activation(self.activation, **self.activation_kwargs),
            )

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add final logit prediction layer
        layers.append(
            nn.Linear(in_features=in_features, out_features=1, bias=self.bias),
        )

        return layers
