"""DenseNet module."""
from typing import List

import torch
import torch.nn as nn

from mdlearn.nn.utils import get_activation


class DenseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        neurons: List[int] = [128],
        bias: bool = True,
        relu_slope: float = 0.0,
        inplace_activation: bool = False,
    ):
        """DenseNet module for easy feedforward network creation.
        Creates a neural network with Linear layers and ReLU (or
        LeakyReLU activation). The returned tensor from the forward
        function, does not pass through an activation function.

        Parameters
        ----------
        input_dim : int
            Dimension of input tensor (should be flattened).
        neurons : List[int], default=[128]
            Linear layers :obj:`in_features`.
        bias : bool, default=True
            Use a bias term in the Linear layers.
        relu_slope : float, default=0.0
            If greater than 0.0, will use LeakyReLU activiation with
            :obj:`negative_slope` set to :obj:`relu_slope`.
        inplace_activation : bool, default=False
            Sets the inplace option for the activation function.

        Raises
        ------
        ValueError
            :obj:`neurons` should specify atleast one layer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.neurons = neurons
        self.bias = bias
        self.relu_slope = relu_slope

        if not self.neurons:
            raise ValueError(
                "Model must have atleast one layer, received an empty list for `neurons`."
            )

        # Select activation
        self.activation_kwargs = {"inplace": inplace_activation}
        if self.relu_slope > 0.0:
            self.activation = "LeakyReLU"
            self.activation_kwargs["negative_slope"] = self.relu_slope
        else:
            self.activation = "ReLU"

        self.model = nn.Sequential(*self._affine_layers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dense network.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            The output of the neural network with dimension (batch size, last neuron size).
        """
        output = self.model(x.squeeze())
        return output

    def _affine_layers(self) -> List[nn.Module]:
        layers = []

        in_features = self.input_dim
        for out_features in self.neurons:
            layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=self.bias,
                ),
            )
            layers.append(get_activation(self.activation, **self.activation_kwargs))

            # in_features of next layer is out_features of current layer
            in_features = out_features

        # Remove last activation function
        layers.pop()

        return layers
