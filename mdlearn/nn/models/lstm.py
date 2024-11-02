""".. warning:: LSTM models are still under development, use with caution!"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mdlearn.nn.utils import Trainer
from mdlearn.utils import PathLike


class LSTM(nn.Module):
    """LSTM model to predict the dynamics for a
    time series of feature vectors.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Parameters
        ----------
        input_size: int
            The number of expected features in the input :obj:`x`.
        hidden_size: Optional[int], default=None
            The number of features in the hidden state h. By default, the
            :obj:`hidden_size` will be equal to the :obj:`input_size` in
            order to propogate the dynamics.
        num_layers: int, default=1
            Number of recurrent layers. E.g., setting num_layers=2 would mean
            stacking two LSTMs together to form a stacked LSTM, with the second
            LSTM taking in outputs of the first LSTM and computing the final
            results.
        bias: bool, default=True
            If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        dropout: float, default=0.0
            If non-zero, introduces a Dropout layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal
            to dropout.
        bidirectional: bool, default=False
            If True, becomes a bidirectional LSTM.
        """
        super().__init__()

        self.num_layers = num_layers
        if hidden_size is None:
            hidden_size = input_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Linear prediction head to map LSTM activation
        # function outputs to the correct output range
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape BxNxD for B batches of N examples
            by D feature dimensions.

        Returns
        -------
        torch.Tensor
            The predicted tensor of size (B, N, hidden_size).
        """
        _, (h_n, _) = self.lstm(x)  # output, (h_n, c_n)

        # Handle bidirectional and num_layers
        pred = h_n[self.num_layers - 1, ...]

        pred = self.head(pred)
        return pred

    def mse_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the MSE loss between :obj:`y_true` and :obj:`y_pred`.

        Parameters
        ----------
        y_true : torch.Tensor
            The true data.
        y_pred : torch.Tensor
            The prediction.
        reduction : str, default="mean"
            The reduction strategy for the F.mse_loss function.

        Returns
        -------
        torch.Tensor
            The MSE loss between :obj:`y_true` and :obj:`y_pred`.
        """
        return F.mse_loss(y_true, y_pred, reduction=reduction)


class LSTMTrainer(Trainer):
    """Trainer class to fit an LSTM model to a time series of feature vectors."""

    # TODO: Add example usage in documentation.

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        window_size: int = 10,
        horizon: int = 1,
        seed: int = np.random.default_rng().integers(2**31 - 1, dtype=int),
        in_gpu_memory: bool = False,
        num_data_workers: int = 0,
        prefetch_factor: int = 2,
        split_pct: float = 0.8,
        split_method: str = 'partition',
        batch_size: int = 128,
        inference_batch_size: int = 128,
        shuffle: bool = True,
        device: str = 'cpu',
        optimizer_name: str = 'RMSprop',
        optimizer_hparams: dict[str, Any] = {
            'lr': 0.001,
            'weight_decay': 0.00001,
        },
        scheduler_name: Optional[str] = None,
        scheduler_hparams: dict[str, Any] = {},
        epochs: int = 100,
        verbose: bool = False,
        clip_grad_max_norm: float = 10.0,
        checkpoint_log_every: int = 10,
        plot_log_every: int = 10,
        plot_n_samples: int = 10000,
        plot_method: Optional[str] = 'TSNE',
        train_subsample_pct: float = 1.0,
        valid_subsample_pct: float = 1.0,
        use_wandb: bool = False,
    ):
        """
        Parameters
        ----------
        input_size: int
            The number of expected features in the input x.
        hidden_size: Optional[int], default=None
            The number of features in the hidden state h. By default, the
            :obj:`hidden_size` will be equal to the :obj:`input_size` in
            order to propogate the dynamics.
        num_layers: int, default=1
            Number of recurrent layers. E.g., setting num_layers=2 would mean
            stacking two LSTMs together to form a stacked LSTM, with the second
            LSTM taking in outputs of the first LSTM and computing the final
            results.
        bias: bool, default=True
            If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        dropout: float, default=0.0
            If non-zero, introduces a Dropout layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal
            to dropout.
        bidirectional: bool, default=False
            If True, becomes a bidirectional LSTM.
        window_size : int, default=10
            Number of timesteps considered for prediction.
        horizon : int, default=1
            How many time steps to predict ahead.
        seed : int, default=np.random.default_rng().integers(2**31 - 1, dtype=int)
            Random seed for torch, numpy, and random module.
        in_gpu_memory : bool, default=False
            If True, will pre-load the entire :obj:`data` array to GPU memory.
        num_data_workers : int, default=0
            How many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
        prefetch_factor : int, by default=2
            Number of samples loaded in advance by each worker. 2 means there will be a
            total of 2 * num_workers samples prefetched across all workers.
        split_pct : float, default=0.8
            Proportion of data set to use for training. The rest goes to validation.
        split_method : str, default="random"
            Method to split the data. For random split use "random", for a simple
            partition, use "partition".
        batch_size : int, default=128
            Mini-batch size for training.
        inference_batch_size : int, default=128
            Mini-batch size for inference.
        shuffle : bool, default=True
            Whether to shuffle training data or not.
        device : str, default="cpu"
            Specify training hardware either :obj:`cpu` or :obj:`cuda` for GPU devices.
        optimizer_name : str, default="RMSprop"
            Name of the PyTorch optimizer to use. Matches PyTorch optimizer class name.
        optimizer_hparams : Dict[str, Any], default={"lr": 0.001, "weight_decay": 0.00001}
            Dictionary of hyperparameters to pass to the chosen PyTorch optimizer.
        scheduler_name : Optional[str], default=None
            Name of the PyTorch learning rate scheduler to use.
            Matches PyTorch optimizer class name.
        scheduler_hparams : Dict[str, Any], default={}
            Dictionary of hyperparameters to pass to the chosen PyTorch learning rate scheduler.
        epochs : int, default=100
            Number of epochs to train for.
        verbose : bool, default=False
            If True, will print training and validation loss at each epoch.
        clip_grad_max_norm : float, default=10.0
            Max norm of the gradients for gradient clipping for more information
            see: :obj:`torch.nn.utils.clip_grad_norm_` documentation.
        checkpoint_log_every : int, default=10
            Epoch interval to log a checkpoint file containing the model
            weights, optimizer, and scheduler parameters.
        plot_log_every : int, default=10
            Epoch interval to log a visualization plot of the latent space.
        plot_n_samples : int, default=10000
            Number of validation samples to use for plotting.
        plot_method : Optional[str], default="TSNE"
            The method for visualizing the latent space or if visualization
            should not be run, set :obj:`plot_method=None`. If using :obj:`"TSNE"`,
            it will attempt to use the RAPIDS.ai GPU implementation and
            will fallback to the sklearn CPU implementation if RAPIDS.ai
            is unavailable.
        train_subsample_pct : float, default=1.0
            Percentage of training data to use during hyperparameter sweeps.
        valid_subsample_pct : float, default=1.0
            Percentage of validation data to use during hyperparameter sweeps.
        use_wandb : bool, default=False
            If True, will log results to wandb.

        Raises
        ------
        ValueError
            :obj:`split_pct` should be between 0 and 1.
        ValueError
            :obj:`train_subsample_pct` should be between 0 and 1.
        ValueError
            :obj:`valid_subsample_pct` should be between 0 and 1.
        ValueError
            Specified :obj:`device` as :obj:`cuda`, but it is unavailable.
        """
        super().__init__(
            seed,
            in_gpu_memory,
            num_data_workers,
            prefetch_factor,
            split_pct,
            split_method,
            batch_size,
            inference_batch_size,
            shuffle,
            device,
            epochs,
            verbose,
            clip_grad_max_norm,
            checkpoint_log_every,
            plot_log_every,
            plot_n_samples,
            plot_method,
            train_subsample_pct,
            valid_subsample_pct,
            use_wandb,
        )

        self.window_size = window_size
        self.horizon = horizon
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_name = scheduler_name
        self.scheduler_hparams = scheduler_hparams

        from mdlearn.utils import get_torch_optimizer
        from mdlearn.utils import get_torch_scheduler

        # Set random seeds
        self._set_seed()

        self.model = LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            dropout,
            bidirectional,
        ).to(self.device)

        if self.use_wandb:
            import wandb

            wandb.watch(self.model)

        # Setup optimizer
        self.optimizer = get_torch_optimizer(
            self.optimizer_name,
            self.optimizer_hparams,
            self.model.parameters(),
        )

        # Setup learning rate scheduler
        self.scheduler = get_torch_scheduler(
            self.scheduler_name,
            self.scheduler_hparams,
            self.optimizer,
        )

        # Log the train and validation loss each epoch
        self.loss_curve_ = {'train': [], 'validation': []}

    def fit(
        self,
        X: np.ndarray,
        scalars: dict[str, np.ndarray] = {},
        output_path: PathLike = './',
        checkpoint: Optional[PathLike] = None,
    ):
        """Trains the LSTM on the input data :obj:`X`.

        Parameters
        ----------
        X : np.ndarray
            Input features vectors of shape (N, D) where N is the number
            of data examples, and D is the dimension of the feature vector.
        scalars : Dict[str, np.ndarray], default={}
            Dictionary of scalar arrays. For instance, the root mean squared
            deviation (RMSD) for each feature vector can be passed via
            :obj:`{"rmsd": np.array(...)}`. The dimension of each scalar array
            should match the number of input feature vectors N.
        output_path : PathLike, default="./"
            Path to write training results to. Makes an :obj:`output_path/checkpoints`
            folder to save model checkpoint files, and :obj:`output_path/plots` folder
            to store latent space visualizations.
        checkpoint : Optional[PathLike], default=None
            Path to a specific model checkpoint file to restore training.

        Raises
        ------
        ValueError
            If :obj:`X` does not have two dimensions. For scalar time series, please
            reshape to (N, 1).
        TypeError
            If :obj:`scalars` is not type dict. A common error is to pass
            :obj:`output_path` as the second argument.
        NotImplementedError
            If using a learning rate scheduler other than :obj:`ReduceLROnPlateau`,
            a step function will need to be implemented.
        """
        if len(X.shape) != 2:
            raise ValueError(
                f'X should be of dimension (N, D), got {X.shape}.',
            )
        if not isinstance(scalars, dict):
            raise TypeError(
                'scalars should be of type dict. A common error'
                ' is to pass output_path as the second argument.',
            )

        from mdlearn.data.datasets.feature_vector import (
            TimeFeatureVectorDataset,
        )
        from mdlearn.data.utils import train_valid_split
        from mdlearn.utils import log_checkpoint
        from mdlearn.visualize import log_latent_visualization

        if self.use_wandb:
            import wandb

        exist_ok = (checkpoint is not None) or self.use_wandb
        output_path, checkpoint_path, plot_path = self._make_output_dir(
            output_path,
            exist_ok,
        )

        # Set available number of cores
        self._set_num_threads()

        # Load training and validation data
        dataset = TimeFeatureVectorDataset(
            X,
            scalars,
            in_gpu_memory=self.in_gpu_memory,
            window_size=self.window_size,
            horizon=self.horizon,
        )
        train_loader, valid_loader = train_valid_split(
            dataset,
            self.split_pct,
            self.split_method,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_data_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            pin_memory=not self.in_gpu_memory,
        )
        self.scalar_dset_names = list(scalars.keys())

        # Optionally resume training from a checkpoint
        start_epoch = self._resume_training(checkpoint)

        # Start training
        for epoch in range(start_epoch, self.epochs + 1):
            # Training
            self.model.train()
            avg_train_loss = self._train(train_loader)

            if self.verbose:
                print(
                    f'====> Epoch: {epoch} Train:\tAvg loss: {avg_train_loss:.4f}',
                )

            # Validation
            self.model.eval()
            with torch.no_grad():
                avg_valid_loss, z, paints = self._validate(valid_loader)

            if self.verbose:
                print(
                    f'====> Epoch: {epoch} Valid:\tAvg loss: {avg_valid_loss:.4f}\n',
                )

            # Step the learning rate scheduler
            self.step_scheduler(epoch, avg_train_loss, avg_valid_loss)

            # Log a model checkpoint file
            if epoch % self.checkpoint_log_every == 0:
                log_checkpoint(
                    checkpoint_path / f'checkpoint-epoch-{epoch}.pt',
                    epoch,
                    self.model,
                    {'optimizer': self.optimizer},
                    self.scheduler,
                )

            if self.use_wandb:
                metrics = {
                    'train_loss': avg_train_loss,
                    'valid_loss': avg_valid_loss,
                }

            # Log a visualization of the latent space
            if (self.plot_method is not None) and (
                epoch % self.plot_log_every == 0
            ):
                htmls = log_latent_visualization(
                    z,
                    paints,
                    plot_path,
                    epoch,
                    self.plot_n_samples,
                    self.plot_method,
                )
                if self.use_wandb:
                    # Optionally, log visualizations to wandb
                    for name, html in htmls.items():
                        metrics[name] = wandb.Html(html, inject=False)

            if self.use_wandb:
                wandb.log(metrics)

            # Save the losses
            self.loss_curve_['train'].append(avg_train_loss)
            self.loss_curve_['validation'].append(avg_valid_loss)

    def predict(
        self,
        X: np.ndarray,
        inference_batch_size: int | None = None,
        checkpoint: Optional[PathLike] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict using the LSTM.

        Parameters
        ----------
        X : np.ndarray
            The input data to predict on.
        inference_batch_size : int, default=None
            The batch size for inference (if None uses the
            value specified during Trainer construction).
        checkpoint : Optional[PathLike], default=None
            Path to a specific model checkpoint file.

        Returns
        -------
        Tuple[np.ndarray, float]
            The predictions and the average MSE loss.
        """
        from mdlearn.data.datasets.feature_vector import (
            TimeFeatureVectorDataset,
        )

        # Fall back to default batch size
        if inference_batch_size is None:
            inference_batch_size = self.inference_batch_size

        dataset = TimeFeatureVectorDataset(
            X,
            in_gpu_memory=self.in_gpu_memory,
            window_size=self.window_size,
            horizon=self.horizon,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            pin_memory=not self.in_gpu_memory,
        )

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        # Make copy of class state incase of failure during inference
        tmp = self.scalar_dset_names.copy()
        self.model.eval()
        with torch.no_grad():
            try:
                # Set to empty list to avoid storage of paint scalars
                # that are not convenient to pass to the predict function.
                self.scalar_dset_names = []
                avg_loss, preds, _ = self._validate(data_loader)
                # Restore class state
                self.scalar_dset_names = tmp
                return preds, avg_loss
            except Exception as e:
                # Restore class state incase of failure
                self.scalar_dset_names = tmp
                raise e

    def _train(self, train_loader) -> float:
        avg_loss = 0.0
        for i, batch in enumerate(train_loader):
            if i / len(train_loader) > self.train_subsample_pct:
                break  # Early stop for sweeps

            x = batch['X'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)

            # Forward pass
            y_pred = self.model(x)
            loss = self.model.mse_loss(y, y_pred)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad_max_norm,
            )
            self.optimizer.step()

            # Collect loss
            avg_loss += loss.item()

        avg_loss /= len(train_loader)

        return avg_loss

    def _validate(
        self,
        valid_loader,
    ) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
        paints = defaultdict(list)
        preds = []
        avg_loss = 0.0
        for i, batch in enumerate(valid_loader):
            if i / len(valid_loader) > self.valid_subsample_pct:
                break  # Early stop for sweeps

            x = batch['X'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)

            # Forward pass
            y_pred = self.model(x)
            loss = self.model.mse_loss(y, y_pred)

            # Collect loss
            avg_loss += loss.item()

            # Collect latent vectors for visualization
            preds.append(y_pred.cpu().numpy())
            for name in self.scalar_dset_names:
                paints[name].append(batch[name].cpu().numpy())

        avg_loss /= len(valid_loader)
        # Group latent vectors and paints
        preds = np.concatenate(preds)
        paints = {
            name: np.concatenate(scalar) for name, scalar in paints.items()
        }

        return avg_loss, preds, paints
