from typing import List, Optional, Tuple

import torch

from mdlearn.nn.models.wae import WAE
from mdlearn.nn.modules.conv2d_decoder import Conv2dDecoder
from mdlearn.nn.modules.conv2d_encoder import Conv2dEncoder


class SymmetricConv2dWAE(WAE):
    """Convolutional variational autoencoder from the
    `"Deep clustering of protein folding simulations"
    <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2507-5>`_
    paper implemented with wasserstein regularization.
    Inherits from :obj:`mdlearn.nn.models.wae.WAE`."""

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
        output_activation: str = "Sigmoid",
    ):
        """
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            (height, width) input dimensions of input image.
        init_weights : Optional[str]
            .pt weights file to initial weights with.
        filters : List[int]
            Convolutional filter dimensions.
        kernels : List[int]
            Convolutional kernel dimensions (assumes square kernel).
        strides : List[int]
            Convolutional stride lengths (assumes square strides).
        affine_widths : List[int]
            Number of neurons in each linear layer.
        affine_dropouts : List[float]
            Dropout probability for each linear layer. Dropout value
            of 0.0 will skip adding the dropout layer.
        latent_dim : int
            Latent dimension for :math:`mu` and :math:`logstd` layers.
        activation : str
            Activation function to use between convultional and linear layers.
        output_activation : str
            Output activation function for last decoder layer.
        """
        self._check_hyperparameters(
            kernels, strides, filters, affine_widths, affine_dropouts
        )

        encoder = Conv2dEncoder(
            input_shape,
            init_weights,
            filters,
            kernels,
            strides,
            affine_widths,
            affine_dropouts,
            latent_dim,
            activation,
        )

        decoder = Conv2dDecoder(
            input_shape,
            encoder.shapes,
            init_weights,
            list(reversed(filters)),
            list(reversed(kernels)),
            list(reversed(strides)),
            list(reversed(affine_widths)),
            list(reversed(affine_dropouts)),
            latent_dim,
            activation,
            output_activation,
        )

        super().__init__(encoder, decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of variational autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input :obj:`x` data to encode and reconstruct.

        Returns
        -------
        torch.Tensor
            :math:`z`-latent space batch tensor.
        torch.Tensor
            :obj:`recon_x` reconstruction of :obj:`x`.
        """
        z = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x

    def _check_hyperparameters(
        self,
        kernels: List[int],
        strides: List[int],
        filters: List[int],
        affine_widths: List[int],
        affine_dropouts: List[float],
    ):
        """Check that hyperparameters are consistent and logical."""
        if not (len(kernels) == len(strides) == len(filters)):
            raise ValueError("Number of filters, kernels and strides must be equal.")

        if len(affine_dropouts) != len(affine_widths):
            raise ValueError(
                "Number of dropouts must equal the number of affine widths."
            )

        # Common convention: allows for filter center and for even padding
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Only odd valued kernel sizes allowed.")

        if any(p < 0 or p > 1 for p in affine_dropouts):
            raise ValueError("Dropout probabilities, p, must be 0 <= p <= 1.")
