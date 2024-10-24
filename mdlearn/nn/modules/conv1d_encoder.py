"""Conv1dEncoder module for point cloud data."""
from __future__ import annotations

import torch
from torch import nn

from mdlearn.nn.utils import get_activation


class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        num_points: int,
        num_features: int = 0,
        latent_dim: int = 20,
        bias: bool = True,
        relu_slope: float = 0.0,
        filters: list[int] = [64, 128, 256, 256, 512],
        kernels: list[int] = [5, 5, 3, 1, 1],
    ):
        """Conv1dEncoder module for point cloud data.

        Parameters
        ----------
        num_points : int
            Number of input points in point cloud.
        num_features : int, optional
            Number of scalar features per point in addition to 3D
            coordinates, by default 0.
        latent_dim : int, optional
            Latent dimension of the encoder, by default 20.
        bias : bool, optional
            Use a bias term in the Conv1d layers, by default True.
        relu_slope : float, optional
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`, by default 0.0.
        filters : List[int], optional
            Encoder Conv1d filter sizes, by default [64, 128, 256, 256, 512].
        kernels : List[int], optional
            Encoder Conv1d kernel sizes, by default [5, 5, 3, 1, 1].
        """
        super().__init__()

        self.num_points = num_points
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.bias = bias
        self.relu_slope = relu_slope
        self.filters = filters
        self.kernels = kernels

        # Select activation
        self.activation_kwargs = {'inplace': True}
        if self.relu_slope > 0.0:
            self.activation = 'LeakyReLU'
            self.activation_kwargs['negative_slope'] = self.relu_slope
        else:
            self.activation = 'ReLU'

        self.conv = nn.Sequential(*self._conv_layers())

        self.fc = nn.Sequential(
            nn.Linear(self.filters[-1], self.filters[-2]),
            get_activation(self.activation, **self.activation_kwargs),
        )

        self.mu = nn.Linear(filters[-2], self.latent_dim)
        self.logstd = nn.Linear(filters[-2], self.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        return self.mu(logit), self.logstd(logit)

    def _conv_layers(self) -> list[nn.Module]:
        layers = []
        # Three xyz atoms + other optional scalars
        in_channels = 3 + self.num_features
        for filter_, kernel in zip(self.filters, self.kernels):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=filter_,
                    kernel_size=kernel,
                    bias=self.bias,
                ),
            )
            layers.append(
                get_activation(self.activation, **self.activation_kwargs),
            )

            # in_channels of next layer is out_channels of current layer
            in_channels = filter_

        layers.pop()  # Remove extra activation on final layer

        return layers
