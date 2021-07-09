import h5py
import torch
import numpy as np
from typing import List, Dict
from mdlearn.utils import PathLike
from torch.utils.data import Dataset


class FeatureVectorDataset(Dataset):
    """PyTorch Dataset class to load vector or scalar data directly
    from a np.ndarray."""

    def __init__(
        self,
        data: np.ndarray,
        scalars: Dict[str, np.ndarray] = {},
        scalar_requires_grad: bool = False,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Input features vectors of shape (N, D) where N is the number
            of data examples, and D is the dimension of the feature vector.
        scalars : Dict[str, np.ndarray]
            Dictionary of scalar arrays. For instance, the root mean squared
            deviation (RMSD) for each feature vector can be passed via
            :obj:`{"rmsd": np.array(...)}`. The dimension of each scalar array
            should match the number of input feature vectors N.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            :obj:`scalars`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        """

        if not all(len(scalars[key]) == len(data) for key in scalars):
            raise ValueError(
                "Dimension of scalar arrays should match "
                "the number of input feature vectors."
            )

        self.data = data
        self.scalars = scalars
        self._scalar_requires_grad = scalar_requires_grad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = {"X": self.data[idx]}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalars.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample


class FeatureVectorHDF5Dataset(Dataset):
    """PyTorch Dataset class to load vector or scalar data from an HDF5 file."""

    def __init__(
        self,
        path: PathLike,
        dataset_name: str,
        scalar_dset_names: List[str] = [],
        scalar_requires_grad: bool = False,
        in_memory: bool = True,
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to h5 file containing contact matrices.
        dataset_name : str
            Path to contact maps in HDF5 file.
        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file to be passed
            to training logs.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            `scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        in_memory: bool
            If True, pull data stored in HDF5 from disk to numpy arrays. Otherwise,
            read each batch from HDF5 on the fly.
        """
        self._file_path = str(path)
        self._dataset_name = dataset_name
        self._scalar_dset_names = scalar_dset_names
        self._scalar_requires_grad = scalar_requires_grad
        self.in_memory = in_memory

        # get lengths and paths
        with self._open_h5_file() as f:
            self.len = len(f[self._dataset_name])

        # Only call _init_dataset once
        self._initialized = False

    def _open_h5_file(self):
        return h5py.File(self._file_path, "r", libver="latest", swmr=False)

    def __len__(self):
        return self.len

    def _init_dataset(self):
        # Create h5py datasets
        self._h5_file = self._open_h5_file()
        self._dset = self._h5_file[self._dataset_name]
        # Load scalar dsets
        self._scalar_dsets = {
            name: self._h5_file[name] for name in self._scalar_dset_names
        }

        # Pull data into main memory (numpy) and close h5 file
        if self.in_memory:
            self.dset = np.array(self.dset)
            self.scalar_dsets = {
                name: np.array(dset) for name, dset in self.scalar_dsets.items()
            }
            self._h5_file.close()

        self._initialized = True

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._init_dataset()

        sample = {"X": self.dset[idx] if self.in_memory else self.dset[idx, ...]}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample
