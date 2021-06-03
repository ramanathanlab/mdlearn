from torch import nn


def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def conv_output_dim(input_dim, kernel_size, stride, padding, transpose=False):
    """
    Parameters
    ----------
    input_dim : int
        input size. may include padding
    kernel_size : int
        filter size
    stride : int
        stride length
    padding : int
        length of 0 pad
    """

    if transpose:
        # TODO: see symmetric decoder _conv_layers,
        #       may have bugs for transpose layers
        output_padding = 1 if stride > 1 else 0
        output_padding = 0
        return (input_dim - 1) * stride + kernel_size - 2 * padding + output_padding

    return (2 * padding + input_dim - kernel_size) // stride + 1


def conv_output_shape(
    input_dim, kernel_size, stride, padding, num_filters, transpose=False, dim=2
):
    """
    Parameters
    ----------
    input_dim : tuple
        (height, width) dimensions for convolution input
    kernel_size : int
        filter size
    stride : int
        stride length
    padding : tuple
        (height, width) length of 0 pad
    num_filters : int
        number of filters
    transpose : bool
        signifies whether Conv or ConvTranspose
    dim : int
        1 or 2, signifies Conv1d or Conv2d
    Returns
    -------
    (channels, height, width) tuple
    """
    if isinstance(input_dim, int):
        input_dim, padding = [input_dim], [padding]

    dims = [
        conv_output_dim(d, kernel_size, stride, p, transpose)
        for d, p in zip(input_dim, padding)
    ]

    if dim == 1:
        return num_filters, dims[0]
    if dim == 2:
        return num_filters, dims[0], dims[1]

    raise ValueError(f"Invalid dim: {dim}")


def _same_padding(input_dim, kernel_size, stride):
    """
    Implements Keras-like same padding.
    If the stride is one then use same_padding.
    Otherwise, select the smallest pad such that the
    kernel_size fits evenly within the input_dim.
    """
    if stride == 1:
        # In this case we want output_dim = input_dim
        # input_dim = output_dim = (2*pad + input_dim - kernel_size) // stride + 1
        return (input_dim * (stride - 1) - stride + kernel_size) // 2

    # Largest i such that: alpha = kernel_size + i*stride <= input_dim
    # Then input_dim - alpha is the pad
    # i <= (input_dim - kernel_size) // stride
    for i in reversed(range((input_dim - kernel_size) // stride + 1)):
        alpha = kernel_size + i * stride
        if alpha <= input_dim:
            # TODO: see symmetric decoder
            # adjustment = int(input_dim % 2 == 0)
            return input_dim - alpha  # + adjustment

    raise Exception("No padding found")


def same_padding(input_dim, kernel_size, stride):
    """
    Returns Keras-like same padding.
    Works for rectangular input_dim.
    Parameters
    ----------
    input_dim : tuple, int
        (height, width) dimensions for Conv2d input
        int for Conv1d input
    kernel_size : int
        filter size
    stride : int
        stride length
    """

    # Handle Conv1d case
    if isinstance(input_dim, int):
        return _same_padding(input_dim, kernel_size, stride)

    h_pad = _same_padding(input_dim[0], kernel_size, stride)
    # If square input, no need to compute width padding
    if input_dim[0] == input_dim[1]:
        return h_pad, h_pad
    w_pad = _same_padding(input_dim[1], kernel_size, stride)
    return h_pad, w_pad


def get_activation(activation, *args, **kwargs):
    """
    Parameters
    ----------
    activation : str
        type of activation e.g. 'ReLU', etc
    """
    if activation == "ReLU":
        return nn.ReLU(*args, **kwargs)
    if activation == "LeakyReLU":
        return nn.LeakyReLU(*args, **kwargs)
    if activation == "Sigmoid":
        return nn.Sigmoid(*args, **kwargs)
    if activation == "Tanh":
        return nn.Tanh(*args, **kwargs)
    if activation == "None":
        return nn.Identity(*args, **kwargs)
    raise ValueError(f"Invalid activation type: {activation}")


# TODO: generalize this more.
def _init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_uniform_(m.weight)
