"""Functions to visualize modeling results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from numpy.typing import ArrayLike
from plotly.io import to_html

PathLike = Union[str, Path]


def plot_scatter(
    data: ArrayLike,
    color_dict: dict[str, ArrayLike] = {},
    color: Optional[str] = None,
) -> plotly.graph_objects._figure.Figure:
    df_dict = color_dict.copy()

    dim = data.shape[1]
    assert dim in [2, 3]
    for i, name in zip(range(dim), ['x', 'y', 'z']):
        df_dict[name] = data[:, i]

    df = pd.DataFrame(df_dict)
    scatter_kwargs = dict(
        x='x',
        y='y',
        color=color,
        width=1000,
        height=1000,
        size_max=7,
        hover_data=list(df_dict.keys()),
    )
    if dim == 2:
        fig = px.scatter(df, **scatter_kwargs)
    else:  # dim == 3
        fig = px.scatter_3d(df, z='z', **scatter_kwargs)
    return fig


def log_latent_visualization(
    data: ArrayLike,
    colors: dict[str, ArrayLike],
    output_path: PathLike,
    epoch: int = 0,
    n_samples: Optional[int] = None,
    method: str = 'raw',
) -> dict[str, str]:
    """Make scatter plots of the latent space using the specified
    method of dimensionality reduction.

    Parameters
    ----------
    data : ArrayLike
        The latent embeddings to visualize of shape (N, D) where
        N  is the number of examples and D is the number of dimensions.
    colors : Dict[str, ArrayLike]
        Each item in the dictionary will generate a different plot labeled
        with the key name. Each inner array should be of size N.
    output_path : PathLike
        The output directory path to save plots to.
    epoch : int, default=0
        The current epoch of training to label plots with.
    n_samples : Optional[int], default=None
        Number of samples to plot, will take a random sample of the
        :obj:`data` if :obj:`n_samples < N`. Otherwise, if :obj:`n_samples`
        is None, use all the data.
    method : str, default="raw"
        Method of dimensionality reduction used to plot. Currently supports:
        "PCA", "TSNE", "LLE", or "raw" for plotting the raw embeddings (or
        up to the first 3 dimensions if D > 3). If "TSNE" is specified, then
        the GPU accelerated RAPIDS.ai implementation will be tried first and
        if it is unavailable then the sklearn version will be used instead.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping each key in color to a raw HTML string containing
        the scatter plot data. These can be saved directly for visualization
        and logged to wandb during training.

    Raises
    ------
    ValueError
        If dimensionality reduction :obj:`method` is not supported.
    """
    # Make temp variables to not mutate input data
    if n_samples is not None:
        inds = np.random.choice(len(data), n_samples)
        _data = data[inds]
        _colors = {name: color[inds] for name, color in colors.items()}
    else:
        _data = data
        _colors = colors

    if method == 'PCA':
        from sklearn.decomposition import PCA

        model = PCA(n_components=3)
        data_proj = model.fit_transform(_data)

    elif method == 'TSNE':
        try:
            # Attempt to use rapidsai
            from cuml.manifold import TSNE

            # rapidsai only supports 2 dimensions
            model = TSNE(n_components=2, method='barnes_hut')
        except ImportError:
            from sklearn.manifold import TSNE

            model = TSNE(n_components=3, n_jobs=1)

        data_proj = model.fit_transform(_data)

    elif method == 'LLE':
        from sklearn import manifold

        data_proj, _ = manifold.locally_linear_embedding(
            _data,
            n_neighbors=12,
            n_components=3,
        )
    elif method == 'raw':
        if _data.shape[1] <= 3:
            # If _data only has 2 or 3 dimensions, use it directly.
            data_proj = _data
        else:
            # Use the first 3 dimensions of the raw data.
            data_proj = _data[:, :3]
    else:
        raise ValueError(f'Invalid dimensionality reduction method {method}')

    html_strings = {}
    for color in _colors:
        fig = plot_scatter(data_proj, _colors, color)
        html_string = to_html(fig)
        html_strings[color] = html_string

        fname = (
            Path(output_path)
            / f'latent_space-{method}-{color}-epoch-{epoch}.html'
        )
        with open(fname, 'w') as f:
            f.write(html_string)

    return html_strings
