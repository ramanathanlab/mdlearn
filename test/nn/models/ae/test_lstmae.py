from __future__ import annotations

import numpy as np


def test_lstmae_net():
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset
    from mdlearn.data.utils import train_valid_split
    from mdlearn.nn.models.ae.lstm import LSTMAE

    N, D, latent_dim, window_size, batch_size = 100, 16, 2, 10, 8
    hidden_neurons = [8, 4]

    X = np.random.normal(size=(N, D))
    dataset = TimeFeatureVectorDataset(X, window_size=window_size)
    train_loader, _ = train_valid_split(
        dataset,
        split_pct=0.8,
        method='partition',
        batch_size=batch_size,
    )

    model = LSTMAE(D, latent_dim, hidden_neurons)
    # print(model)

    for data in train_loader:
        X, y = data['X'], data['y']
        assert X.shape == (batch_size, window_size, D)
        assert y.shape == (batch_size, D)
        z, y_pred = model(X)
        assert y_pred.shape == (batch_size, D)
        assert z.shape == (batch_size, latent_dim)
        break
