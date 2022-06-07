"""LSTMNet module."""
from typing import List

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        neurons: List[int] = [32],
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """LSTMNet module for easy StackedLSTM network creation.
        The returned tensor from the forward function, is the
        hidden state of the final LSTM layer.

        Parameters
        ----------
        input_dim : int
            Dimension D of input tensor (N, D) where N is the length
            of the sequence and D is the dimension of each example.
        neurons : List[int], default=[32]
            LSTM layers :obj:`hidden_size`.
        bias: bool, default=True
            If False, then each layer does not use bias weights b_ih and b_hh.
        dropout: float, default=0.0
            If non-zero, introduces a Dropout layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal
            to dropout.

        Note
        ----
        Bidirectional LSTMs are not currently supported in this module.

        Raises
        ------
        ValueError
            :obj:`neurons` should specify atleast one layer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.neurons = neurons
        self.bias = bias
        self.dropout = dropout

        if not self.neurons:
            raise ValueError(
                "Model must have atleast one layer, received an empty list for `neurons`."
            )

        self.stacked_lstm = nn.ModuleList(self._lstm_layers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, N, D) where B is the batch size,
            N is the length of the sequence, and D is the dimension.

        Returns
        -------
        torch.Tensor
            The output of the neural network with dimension (batch size, last neuron size).
        """
        output = x
        for lstm in self.stacked_lstm:
            output, (h_n, _) = lstm(output)

        # Handle bidirectional index, here LSTM.num_layers is 1.
        return h_n[0, ...]

    def _lstm_layers(self) -> List[nn.Module]:
        layers = []
        input_size = self.input_dim
        for hidden_size in self.neurons:
            layers.append(
                nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=self.dropout,
                )
            )

            # input_size of next layer is hidden_size of current layer
            input_size = hidden_size

        return layers
