"""Utility functions for handling PyTorch data objects."""
from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Subset


def train_valid_split(
    dataset: Dataset,
    split_pct: float = 0.8,
    method: str = 'random',
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Creates training and validation DataLoaders from :obj:`dataset`.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset class derived from :obj:`torch.utils.data.Dataset`.
    split_pct : float
        Percentage of data to be used as training data after a split.
    method : str, default="random"
        Method to split the data. For random split use "random", for a simple
        partition, use "partition".
    **kwargs
        Keyword arguments to :obj:`torch.utils.data.DataLoader`. Includes,
        :obj:`batch_size`, :obj:`drop_last`, etc (see `PyTorch Docs
        <https://pytorch.org/docs/stable/data.html>`_).

    Raises
    ------
    ValueError
        If :obj:`method` is not "random" or "partition".
    """
    train_length = int(len(dataset) * split_pct)
    if method == 'random':
        lengths = [train_length, len(dataset) - train_length]
        train_dataset, valid_dataset = random_split(dataset, lengths)
    elif method == 'partition':
        indices = list(range(len(dataset)))
        train_dataset = Subset(dataset, indices[:train_length])
        valid_dataset = Subset(dataset, indices[train_length:])
    else:
        raise ValueError(f'Invalid method: {method}.')
    train_loader = DataLoader(train_dataset, **kwargs)
    valid_loader = DataLoader(valid_dataset, **kwargs)
    return train_loader, valid_loader
