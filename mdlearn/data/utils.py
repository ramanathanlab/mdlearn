"""Utility functions for handling PyTorch data objects."""

from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


def train_valid_split(
    dataset: Dataset, split_pct: float = 0.8, **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and validation DataLoaders from :obj:`dataset`.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset class derived from :obj:`torch.utils.data.Dataset`.
    split_pct : float
        Percentage of data to be used as training data after a random split.
    **kwargs
        Keyword arguments to :obj:`torch.utils.data.DataLoader`. Includes,
        :obj:`batch_size`, :obj:`drop_last`, etc (see `PyTorch Docs
        <https://pytorch.org/docs/stable/data.html>`_).
    """
    train_length = int(len(dataset) * split_pct)
    lengths = [train_length, len(dataset) - train_length]
    train_dataset, valid_dataset = random_split(dataset, lengths)
    train_loader = DataLoader(train_dataset, **kwargs)
    valid_loader = DataLoader(valid_dataset, **kwargs)
    return train_loader, valid_loader
