from __future__ import annotations

import numpy as np


def test_lstm_net():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset
    from mdlearn.data.utils import train_valid_split
    from mdlearn.nn.modules.lstm_net import LSTMNet

    N, D, window_size, batch_size = 100, 16, 10, 8
    neurons = [4, 2, 1]

    X = np.random.normal(size=(N, D))
    dataset = TimeFeatureVectorDataset(X, window_size=window_size)
    train_loader, _ = train_valid_split(
        dataset,
        split_pct=0.8,
        method='partition',
        batch_size=batch_size,
    )

    model = LSTMNet(input_dim=D, neurons=neurons)
    # print(model)

    for data in train_loader:
        X, y = data['X'], data['y']
        assert X.shape == (batch_size, window_size, D)
        assert y.shape == (batch_size, D)
        y_pred = model(X)
        assert y_pred.shape == (batch_size, neurons[-1])
        break
