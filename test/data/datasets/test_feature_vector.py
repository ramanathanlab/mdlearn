from __future__ import annotations

import numpy as np
import pytest
import torch


def test_dataset_length():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset

    data = np.array([1, 2, 3, 4])

    # data = [1, 2, 3, 4]
    # window = 2
    # horizon = 1
    #   [1,2] -> 3
    #   [2,3] -> 4
    # len: 4 - 2 -1 +1 = 2
    assert len(TimeFeatureVectorDataset(data, window_size=2, horizon=1)) == 2

    # data = [1, 2, 3, 4]
    # window = 1
    # horizon = 1
    # [1] -> 2
    # [2] -> 3
    # [3] -> 4
    # len: 4 - 1 -1 + 1 = 3
    assert len(TimeFeatureVectorDataset(data, window_size=1, horizon=1)) == 3

    # data = [1, 2, 3, 4]
    # window = 1
    # horizon = 2
    # [1] -> 3
    # [2] -> 4
    # len: 4 - 1 - 2 + 1  = 2
    assert len(TimeFeatureVectorDataset(data, window_size=1, horizon=2)) == 2

    # data = [1, 2, 3, 4]
    # window = 3
    # horizon = 1
    # [1,2,3] -> 4
    # len: 4 - 3 - 1 + 1  = 1
    assert len(TimeFeatureVectorDataset(data, window_size=3, horizon=1)) == 1

    with pytest.raises(ValueError):
        TimeFeatureVectorDataset(data, window_size=5, horizon=1)


def test_dataset_windows():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset

    data = np.array([1, 2, 3, 4])

    # data = [1, 2, 3, 4]
    # window = 2
    # horizon = 1
    # dataset[0]: {"X": [1,2], "y": 3}
    # dataset[1]: {"X": [2,3], "y": 4}
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=1)
    assert torch.equal(dataset[0]['X'], torch.tensor([1, 2]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor(3).float())
    assert torch.equal(dataset[1]['X'], torch.tensor([2, 3]).float())
    assert torch.equal(dataset[1]['y'], torch.tensor(4).float())

    # Same but multidimensional input array
    data = np.array([[1], [2], [3], [4]])
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=1)
    assert torch.equal(dataset[0]['X'], torch.tensor([[1], [2]]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor([3]).float())
    assert torch.equal(dataset[1]['X'], torch.tensor([[2], [3]]).float())
    assert torch.equal(dataset[1]['y'], torch.tensor([4]).float())


def test_dataset_horizon():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset

    data = np.array([1, 2, 3, 4])
    # data = [1, 2, 3, 4]
    # window = 1
    # horizon = 2
    # dataset[0]: {"X": [1], "y": 3}
    # dataset[1]: {"X": [2], "y": 4}
    dataset = TimeFeatureVectorDataset(data, window_size=1, horizon=2)
    assert torch.equal(dataset[0]['X'], torch.tensor([1]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor(3).float())
    assert torch.equal(dataset[1]['X'], torch.tensor([2]).float())
    assert torch.equal(dataset[1]['y'], torch.tensor(4).float())

    data = np.array([1, 2, 3, 4, 5])
    # data = [1, 2, 3, 4, 5]
    # window = 2
    # horizon = 2
    # dataset[0]: {"X": [1, 2], "y": 4}
    # dataset[1]: {"X": [2, 3], "y": 5}
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=2)
    assert len(dataset) == 2
    assert torch.equal(dataset[0]['X'], torch.tensor([1, 2]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor(4).float())
    assert torch.equal(dataset[1]['X'], torch.tensor([2, 3]).float())
    assert torch.equal(dataset[1]['y'], torch.tensor(5).float())

    data = np.array([1, 2, 3, 4])
    # data = [1, 2, 3, 4]
    # window = 2
    # horizon = 2
    # dataset[0]: {"X": [1, 2], "y": 4}
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=2)
    assert torch.equal(dataset[0]['X'], torch.tensor([1, 2]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor(4).float())

    # Same but multidimensional input array
    data = np.array([[1], [2], [3], [4]])
    dataset = TimeFeatureVectorDataset(data, window_size=2, horizon=2)
    assert torch.equal(dataset[0]['X'], torch.tensor([[1], [2]]).float())
    assert torch.equal(dataset[0]['y'], torch.tensor([4]).float())


def test_scalar_index():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset

    data = np.array([1, 2, 3, 4, 5])
    scalars = {'scalar': np.array([5, 6, 7, 8, 9])}
    # data = [1, 2, 3, 4, 5]
    # window = 2
    # horizon = 2
    # dataset[0]: {"X": [1, 2], "y": 4, "scalar": 8, "index": 3}
    # dataset[1]: {"X": [2, 3], "y": 5, "scalar": 9, "index": 4}
    dataset = TimeFeatureVectorDataset(data, scalars, window_size=2, horizon=2)
    assert len(dataset) == 2
    assert torch.equal(dataset[0]['scalar'], torch.tensor(8))
    assert torch.equal(dataset[1]['scalar'], torch.tensor(9))
    assert torch.equal(dataset[0]['index'], torch.tensor(3))
    assert torch.equal(dataset[1]['index'], torch.tensor(4))
