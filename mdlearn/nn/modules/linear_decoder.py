"""LinearDecoder module for point cloud data."""
from typing import List

import torch
import torch.nn as nn

from mdlearn.nn.utils import get_activation


class LinearDecoder(nn.Module):
    def __init__(
        self,
        num_points: int,
        num_features: int = 0,
        latent_dim: int = 20,
        bias: bool = True,
        relu_slope: float = 0.0,
        affine_widths: List[int] = [64, 128, 512, 1024],
    ):
        """LinearDecoder module for point cloud data.

        Parameters
        ----------
        num_points : int
            Number of input points in point cloud.
        num_features : int, optional
            Number of scalar features per point in addition to 3D
            coordinates, by default 0.
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

        self.num_points = num_points
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.bias = bias
        self.relu_slope = relu_slope
        self.affine_widths = affine_widths

        # Select activation
        self.activation_kwargs = {"inplace": True}
        if self.relu_slope > 0.0:
            self.activation = "LeakyReLU"
            self.activation_kwargs["negative_slope"] = self.relu_slope
        else:
            self.activation = "ReLU"

        self.model = nn.Sequential(*self._affine_layers())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output = self.model(z.squeeze())
        output = output.view(-1, (3 + self.num_features), self.num_points)
        return output

    def _affine_layers(self) -> List[nn.Module]:
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
            layers.append(get_activation(self.activation, **self.activation_kwargs))

            # in_features of next layer is out_features of current layer
            in_features = width

        layers.append(
            nn.Linear(
                in_features=self.affine_widths[-1],
                out_features=self.num_points * (3 + self.num_features),
                bias=self.bias,
            )
        )

        return layers
