from math import isclose
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from mdlearn.nn.utils import _init_weights, get_activation, same_padding


class Conv2dDecoder(nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, ...],
        encoder_shapes: List[Tuple[int, ...]],
        init_weights: Optional[str] = None,
        filters: List[int] = [64, 64, 64],
        kernels: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 1],
        affine_widths: List[int] = [128],
        affine_dropouts: List[float] = [0.0],
        latent_dim: int = 3,
        activation: str = "ReLU",
        output_activation: str = "Sigmoid",
    ):
        super().__init__()

        self.output_shape = output_shape
        self.encoder_shapes = encoder_shapes
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.affine_widths = affine_widths
        self.affine_dropouts = affine_dropouts
        self.latent_dim = latent_dim
        self.activation = activation
        self.output_activation = output_activation

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers, self.conv_acts = self._conv_layers()
        self.conv_output_sizes = list(reversed(self.encoder_shapes[:-1]))
        # Reshape flattened x as a tensor (channels, output1, output2)
        self.reshape = (-1, *self.encoder_shapes[-1])

        self.init_weights(init_weights)

    def init_weights(self, init_weights: Optional[str]):
        if init_weights is None:
            self.affine_layers.apply(_init_weights)
            self.conv_layers.apply(_init_weights)
        # Loading checkpoint weights
        elif init_weights.endswith(".pt"):
            checkpoint = torch.load(init_weights, map_location="cpu")
            self.load_state_dict(checkpoint["decoder_state_dict"])

    def forward(self, x):
        x = self.affine_layers(x).view(self.reshape)
        batch_size = x.size()[0]
        for conv_t, act, output_size in zip(
            self.conv_layers, self.conv_acts, self.conv_output_sizes
        ):
            x = act(conv_t(x, output_size=(batch_size, *output_size)))
        return x

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        layers : list
            Convolution layers
        activations : list
            Activation functions
        """
        layers, activations = [], []

        # The first out_channels should be the second to last filter size
        tmp = self.filters.pop()

        # self.output_shape[0] Needs to be the last out_channels to match the input matrix
        for i, (filter_, kernel, stride) in enumerate(
            zip(
                (*self.filters, self.output_shape[0]),
                self.kernels,
                self.strides,
            )
        ):
            shape = self.encoder_shapes[-1 * i - 1]

            # TODO: this is a quick fix but might not generalize to some architectures
            if stride == 1:
                padding = same_padding(shape[1:], kernel, stride)
            else:
                padding = tuple(
                    int(dim % 2 == 0) for dim in self.encoder_shapes[-1 * i - 2][1:]
                )

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=shape[0],
                    out_channels=filter_,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

            # TODO: revist padding, output_padding, see github issue.
            #       This code may not generalize to other examples. Needs testing.
            #       this also needs to be addressed in conv_output_dim

            activations.append(get_activation(self.activation))

        # Overwrite output activation
        activations[-1] = get_activation(self.output_activation)

        # Restore invariant state
        self.filters.append(tmp)

        return nn.ModuleList(layers), activations

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        layers : list
            Linear layers
        """

        layers = []

        in_features = self.latent_dim

        for width, dropout in zip(self.affine_widths, self.affine_dropouts):

            layers.append(nn.Linear(in_features=in_features, out_features=width))

            layers.append(get_activation(self.activation))

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add last layer with dims to connect the last linear layer to
        # the first convolutional decoder layer
        layers.append(
            nn.Linear(
                in_features=self.affine_widths[-1],
                out_features=np.prod(self.encoder_shapes[-1]),
            )
        )
        layers.append(get_activation(self.activation))

        return layers
