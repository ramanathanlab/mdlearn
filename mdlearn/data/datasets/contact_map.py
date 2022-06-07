"""ContactMap Dataset."""
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mdlearn.utils import PathLike


class ContactMapDataset(Dataset):
    """PyTorch Dataset class to load contact matrix data."""

    def __init__(
        self,
        path: PathLike,
        shape: Tuple[int, ...],
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
        >>> dataset = ContactMapDataset("contact_maps.h5", (28, 28))
        >>> dataset[0]
        {'X': torch.Tensor(..., dtype=float32), 'index': tensor(0)}
        >>> dataset[0]["X"].shape
        (28, 28)

        >>> dataset = ContactMapDataset("contact_maps.h5", (28, 28), scalar_dset_names=["rmsd"])
        >>> dataset[0]
        {'X': torch.Tensor(..., dtype=float32), 'index': tensor(0), 'rmsd': tensor(8.7578, dtype=torch.float16)}
        """
        self.file_path = str(path)
        self.dataset_name = dataset_name
        self.scalar_dset_names = scalar_dset_names
        self.shape = shape
        self._scalar_requires_grad = scalar_requires_grad
        self._values_dset_name = values_dset_name
        self.in_memory = in_memory

        # Check file for data length
        with self._open_h5_file() as f:
            self.len = len(f[self.dataset_name])

        # Only call _init_dataset once
        self._initialized = False

    def _open_h5_file(self):
        return h5py.File(self.file_path, "r", libver="latest", swmr=False)

    def _init_dataset(self):
        # Create h5py datasets
        self._h5_file = self._open_h5_file()
        self.dset = self._h5_file[self.dataset_name]
        if self._values_dset_name is not None:
            self.val_dset = self._h5_file[self._values_dset_name]
        # Load scalar dsets
        self.scalar_dsets = {
            name: self._h5_file[name] for name in self.scalar_dset_names
        }

        # Pull data into main memory (numpy)
        if self.in_memory:
            # self.dset and self.val_dset is a ragged array, requires storage as an object
            self.dset = np.array(self.dset, dtype=object)
            if self._values_dset_name is not None:
                self.val_dset = np.array(self.val_dset, dtype=object)
            self.scalar_dsets = {
                name: np.array(dset) for name, dset in self.scalar_dsets.items()
            }
            self._h5_file.close()

        self._initialized = True

    def _get_data(self, idx) -> torch.Tensor:
        # Data is stored as np.concatenate((row_inds, col_inds))
        ind = self.dset[idx] if self.in_memory else self.dset[idx, ...]
        indices = torch.from_numpy(ind.reshape(2, -1)).to(torch.long)

        # Create array of 1s, all values in the contact map are 1. Or load values.
        if self._values_dset_name is not None:
            values = torch.from_numpy(self.val_dset[idx, ...]).to(torch.float32)
        else:
            values = torch.ones(indices.shape[1], dtype=torch.float32)
        # Set shape to the last 2 elements of self.shape. Handles (1, W, H) and (W, H)
        data = torch.sparse.FloatTensor(indices, values, self.shape[-2:]).to_dense()
        data = data.view(self.shape)
        return data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._init_dataset()

        sample = {"X": self._get_data(idx)}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample
