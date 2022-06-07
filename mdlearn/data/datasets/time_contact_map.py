"""ContactMapTimeSeriesDataset Dataset."""
from typing import List, Optional, Tuple

import torch

from mdlearn.data.datasets.contact_map import ContactMapDataset
from mdlearn.utils import PathLike


class ContactMapTimeSeriesDataset(ContactMapDataset):
    """PyTorch Dataset class to load contact matrix data in the format of a time series."""

    def __init__(
        self,
        path: PathLike,
        shape: Tuple[int, ...],
        lag_time: int = 1,
        dataset_name: str = "contact_map",
        scalar_dset_names: List[str] = [],
        values_dset_name: Optional[str] = None,
        scalar_requires_grad: bool = False,
        in_memory: bool = True,
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to HDF5 file containing contact matrices.
        shape : Tuple[int, ...]
            Shape of contact matrices required by the model (H, W), may be (1, H, W).
        lag_time: int
            Delay time forward or backward in the input data. The time-lagged
            correlations is computed between :obj:`X[t]` and :obj:`X[t+lag_time]`.
        dataset_name : str
            Name of contact map dataset in HDF5 file.
        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file to be passed to training logs.
        values_dset_name: str, optional
            Name of HDF5 dataset field containing optional values of the entries
            the distance/contact matrix. By default, values are all assumed to be 1
            corresponding to a binary contact map and created on the fly.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            `scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        in_memory: bool
            If True, pull data stored in HDF5 from disk to numpy arrays. Otherwise,
            read each batch from HDF5 on the fly.

        Examples
        --------
        >>> dataset = ContactMapTimeSeriesDataset("contact_maps.h5", (28, 28))
        >>> dataset[0]
        {'X_t': torch.Tensor(..., dtype=float32), 'X_t_tau': torch.Tensor(..., dtype=float32), 'index': tensor(0)}
        >>> dataset[0]["X_t"].shape
        (28, 28)
        >>> dataset[0]["X_t_tau"].shape
        (28, 28)
        """
        super().__init__(
            path,
            shape,
            dataset_name,
            scalar_dset_names,
            values_dset_name,
            scalar_requires_grad,
            in_memory,
        )
        self.lag_time = lag_time

    def __len__(self):
        return self.len - self.lag_time

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._init_dataset()

        sample = {
            "X_t": self._get_data(idx),
            "X_t_tau": self._get_data(idx + self.lag_time),
        }
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample
