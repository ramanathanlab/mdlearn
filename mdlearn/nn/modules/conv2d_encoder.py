from math import isclose
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from mdlearn.nn.utils import (
    _init_weights,
    conv_output_shape,
    get_activation,
    same_padding,
)


class Conv2dEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        init_weights: Optional[str] = None,
        filters: List[int] = [64, 64, 64],
        kernels: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 1],
        affine_widths: List[int] = [128],
        affine_dropouts: List[float] = [0.0],
        latent_dim: int = 3,
        activation: str = "ReLU",
    ):
        super().__init__()

        # Stores (channels, height, width) of conv layers
        self.shapes = [input_shape]
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.affine_widths = affine_widths
        self.affine_dropouts = affine_dropouts
        self.latent_dim = latent_dim
        self.activation = activation

        self.encoder = nn.Sequential(
            *self._conv_layers(), nn.Flatten(), *self._affine_layers()
        )

        self.mu = self._latent_layer()
        self.logstd = self._latent_layer()

        self.init_weights(init_weights)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return self.mu(x), self.logstd(x)

    def init_weights(self, init_weights: Optional[str]):
        if init_weights is None:
            self.encoder.apply(_init_weights)
            _init_weights(self.mu)
            _init_weights(self.logstd)
        # Loading checkpoint weights
        elif init_weights.endswith(".pt"):
            checkpoint = torch.load(init_weights, map_location="cpu")
            self.load_state_dict(checkpoint["encoder_state_dict"])

    def _conv_layers(self):
        """Compose convolution layers.

        Returns
        -------
        list:
            Convolution layers and activations.
        """

        layers = []

        for filter_, kernel, stride in zip(self.filters, self.kernels, self.strides):

            padding = same_padding(self.shapes[-1][1:], kernel, stride)

            layers.append(
                nn.Conv2d(
                    in_channels=self.shapes[-1][0],
                    out_channels=filter_,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

            layers.append(get_activation(self.activation))

            # Output shape is (channels, height, width)
            self.shapes.append(
                conv_output_shape(self.shapes[-1][1:], kernel, stride, padding, filter_)
            )

        return layers

    def _affine_layers(self):
        """Compose affine layers.

        Returns
        -------
        list:
            Linear layers, dropout layers, and activations.
        """
        layers = []

        # First layer gets flattened convolutional output
        in_features = np.prod(self.shapes[-1])

        for width, dropout in zip(self.affine_widths, self.affine_dropouts):

            layers.append(nn.Linear(in_features=in_features, out_features=width))

            layers.append(get_activation(self.activation))

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return layers

    def _latent_layer(self):
        return nn.Linear(
            in_features=self.affine_widths[-1],
            out_features=self.latent_dim,
        )
