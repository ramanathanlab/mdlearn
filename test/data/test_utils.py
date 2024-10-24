from __future__ import annotations

import numpy as np
import torch


def test_train_valid_split():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset
    from mdlearn.data.utils import train_valid_split

    data = np.array([1, 2, 3, 4, 5, 6])
    # data = [1, 2, 3, 4, 5, 6]
    # window = 2
    # horizon = 1
    # dataset[0]: {"X": [1, 2], "y": 3}
    # dataset[1]: {"X": [2, 3], "y": 4}
    # dataset[2]: {"X": [3, 4], "y": 5}
    # dataset[2]: {"X": [4, 5], "y": 6}
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=1)
    assert len(dataset) == 4
    train_loader, valid_loader = train_valid_split(
        dataset,
        split_pct=0.8,
        method='partition',
    )
    train_dataset = [x for x in train_loader]
    valid_dataset = [x for x in valid_loader]
    assert len(train_dataset) == 3
    assert len(valid_dataset) == 1
    # Need to squeeze out batch size of 1
    assert torch.equal(
        train_dataset[0]['X'].squeeze(),
        torch.tensor([1, 2]).float(),
    )
    assert torch.equal(train_dataset[0]['y'], torch.tensor([3]).float())
    assert torch.equal(
        train_dataset[1]['X'].squeeze(),
        torch.tensor([2, 3]).float(),
    )
    assert torch.equal(train_dataset[1]['y'], torch.tensor([4]).float())
    assert torch.equal(
        train_dataset[2]['X'].squeeze(),
        torch.tensor([3, 4]).float(),
    )
    assert torch.equal(train_dataset[2]['y'], torch.tensor([5]).float())
    assert torch.equal(
        valid_dataset[0]['X'].squeeze(),
        torch.tensor([4, 5]).float(),
    )
    assert torch.equal(valid_dataset[0]['y'], torch.tensor([6]).float())
