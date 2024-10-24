from __future__ import annotations

import numpy as np


def check_lstm(
    N: int,
    D: int,
    batch_size: int,
    window_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
):
    """Helper function for testing the LSTM model.

    Parameters
    ----------
    N : int
        Number of points in the dataset.
    D : int
        Dimension of the time series.
    batch_size : int
        Batch size to use for forward pass.
    window_size : int
        Window size of time series for the model to predict with.
    num_layers : int, default=1
        Number of layers in the LSTM model.
    bidirectional : bool, default=False
        Whether or not to use a bidirectional LSTM model.
    """
    from mdlearn.data.datasets.feature_vector import TimeFeatureVectorDataset
    from mdlearn.data.utils import train_valid_split
    from mdlearn.nn.models.lstm import LSTM

    X = np.random.normal(size=(N, D))
    dataset = TimeFeatureVectorDataset(X, window_size=window_size)
    train_loader, _ = train_valid_split(
        dataset,
        split_pct=0.8,
        method='partition',
        batch_size=batch_size,
    )

    model = LSTM(
        input_size=D,
        hidden_size=D,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    for data in train_loader:
        X, y = data['X'], data['y']
        assert X.shape == (batch_size, window_size, D)
        assert y.shape == (batch_size, D)
        y_pred = model(X)
        assert y_pred.shape == y.shape
        _ = model.mse_loss(y, y_pred)
        break


def test_lstm_model_scalar_time_series():
    # Here, D=1
    check_lstm(100, 1, 8, 4)


def test_lstm_model_multidim_time_series():
    # Here, D=16
    check_lstm(100, 16, 8, 4)


def test_lstm_model_window_size_1():
    # Here, window_size=1
    check_lstm(100, 1, 8, 1)  # D=1
    check_lstm(100, 16, 8, 1)  # D=16


def test_lstm_model_num_layers_2():
    # Here, num_layers=2
    check_lstm(100, 1, 8, 4, 2)  # D=1
    check_lstm(100, 16, 8, 4, 2)  # D=16


def test_lstm_model_biderectional():
    # Here, biderectional=True
    check_lstm(100, 1, 8, 4, 1, True)  # D=1
    check_lstm(100, 1, 8, 4, 2, True)  # D=1, num_layers=2
    check_lstm(100, 16, 8, 4, 2, True)  # D=16
