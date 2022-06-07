"""PointCloud Dataset."""
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mdlearn.utils import PathLike


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset class to load point cloud data. Optionally, uses HDF5
    files to only read into memory what is necessary for one batch.
    """

    def __init__(
        self,
        path: PathLike,
        num_points: int,
        num_features: int = 0,
        dataset_name: str = "point_cloud",
        scalar_dset_names: List[str] = [],
        seed: int = 333,
        cms_transform: bool = False,
        scalar_requires_grad: bool = False,
        in_memory: bool = True,
    ):
        """
        Parameters
        ----------
        path : Union[str, Path]
            Path to HDF5 file containing data set.
        dataset_name : str
            Name of the point cloud data in the HDF5 file.
        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file to
            be passed to training logs.
        num_points : int
            Number of points per sample. Should be smaller or equal
            than the total number of points.
        num_features : int
            Number of additional per-point features in addition to xyz coords.
        seed : int
            Seed for the RNG for the splitting. Make sure it is the
            same for all workers reading from the same file.
        cms_transform: bool
            If True, subtract center of mass from batch and shift and scale
            batch by the full dataset statistics.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            :obj:`scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        in_memory: bool
            If True, pull data stored in HDF5 from disk to numpy arrays. Otherwise,
            read each batch from HDF5 on the fly.

        Examples
        --------
        >>> dataset = PointCloudDataset("point_clouds.h5", 28)
        >>> dataset[0]
        {'X': torch.Tensor(..., dtype=float32), 'index': tensor(0)}
        >>> dataset[0]["X"].shape
        torch.Size([3, 28])

        >>> dataset = PointCloudDataset("point_clouds.h5", 28, 1)
        >>> dataset[0]["X"].shape
        torch.Size([4, 28])

        >>> dataset = PointCloudDataset("point_clouds.h5", 28, scalar_dset_names=["rmsd"])
        >>> dataset[0]
        {'X': torch.Tensor(..., dtype=float32), 'index': tensor(0), 'rmsd': tensor(8.7578, dtype=torch.float16)}
        """
        self.file_path = str(path)
        self.dataset_name = dataset_name
        self.scalar_dset_names = scalar_dset_names
        self.num_points = num_points
        self.num_features = num_features
        self.seed = seed
        self.cms_transform = cms_transform
        self.scalar_requires_grad = scalar_requires_grad
        self.in_memory = in_memory

        with self._open_h5_file() as f:
            self.dset = f[self.dataset_name]
            self.len = self.dset.shape[0]
            self.num_features_total = self.dset.shape[1]
            self.num_points_total = self.dset.shape[2]

            # Sanity checks
            assert self.num_points_total >= self.num_points
            assert self.num_features_total == (3 + self.num_features)

        # Only call _init_dataset once
        self.not_initialized = True

    @property
    def point_cloud_size(self) -> Tuple[int, int]:
        return 3 + self.num_features, self.num_points

    def _open_h5_file(self):
        return h5py.File(self.file_path, "r", libver="latest", swmr=False)

    def _init_dataset(self):
        self._h5_file = self._open_h5_file()
        self.dset = self._h5_file[self.dataset_name]
        # Load scalar dsets
        self.scalar_dsets = {
            name: self._h5_file[name] for name in self.scalar_dset_names
        }
        # RNG for random point selection augmentation,
        # active if self.num_points_total > self.num_points
        if self.num_points < self.num_points_total:
            self.rng = np.random.default_rng(self.seed)

        # CMS transform if requested
        if self.cms_transform:
            # Center of mass over points
            cms = np.mean(
                self.dset[:, 0:3, :].astype(np.float64), axis=2, keepdims=True
            ).astype(np.float32)

            # Normalize input
            self.bias = np.zeros(self.point_cloud_size, dtype=np.float32)
            self.scale = np.ones(self.point_cloud_size, dtype=np.float32)
            self.bias[0:3, :] = (self.dset[:, 0:3, :] - cms).min()
            self.scale[0:3, :] = 1.0 / ((self.dset[:, 0:3, :] - cms).max() - self.bias)

        # Pull data into main memory (numpy)
        if self.in_memory:
            self.dset = np.array(self.dset[:, 0 : (3 + self.num_features), :])
            self.scalar_dsets = {
                name: np.array(dset) for name, dset in self.scalar_dsets.items()
            }
            self._h5_file.close()

        # Create temp buffer for IO
        self.buffer = np.zeros((1, *self.point_cloud_size), dtype=np.float32)

        self.not_initialized = False

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Only happens once. Need to open h5 file in current process
        if self.not_initialized:
            self._init_dataset()

        if self.num_points < self.num_points_total:
            # Select random points to read
            point_indices = self.rng.choice(
                self.num_points_total,
                size=self.num_points,
                replace=False,
                shuffle=False,
            )

            # Read from numpy array or HDF5
            self.buffer[0, ...] = self.dset[
                idx, 0 : (3 + self.num_features), point_indices
            ]
        elif self.in_memory:
            # Read from numpy array
            self.buffer[0, ...] = self.dset[
                idx, 0 : (3 + self.num_features), 0 : self.num_points
            ]
        else:
            # Read direcly from HDF5, since data is contiguous and not random idxs
            self.dset.read_direct(
                self.buffer,
                np.s_[idx : idx + 1, 0 : (3 + self.num_features), 0 : self.num_points],
                np.s_[0:1, 0 : (3 + self.num_features), 0 : self.num_points],
            )

        # CMS subtract
        if self.cms_transform:
            self.buffer[0, 0:3, :] -= np.mean(
                self.buffer[0, 0:3, :], axis=-1, keepdims=True
            )

            if np.any(np.isnan(self.buffer)):
                raise ValueError("NaN encountered in input.")

            # Normalize
            self.buffer = (self.buffer[0, ...] - self.bias) * self.scale

        sample = {"X": torch.from_numpy(self.buffer.squeeze())}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self.scalar_requires_grad
            )
        return sample


class CenterOfMassTransform:
    def __init__(self, data: np.ndarray) -> None:
        """Computes center of mass transformation

        Parameters
        ----------
        data : np.ndarray
            Dataset of positions with shape (num_examples, 3, num_points).
        """

        # Center of mass over points
        cms = np.mean(data.astype(np.float64), axis=2, keepdims=True).astype(np.float32)
        # Scalar bias and scale normalization factors
        self.bias: float = (data - cms).min()
        self.scale: float = 1.0 / ((data - cms).max() - self.bias)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize example by bias and scale factors

        Parameters
        ----------
        x : np.ndarray
            Data to transform shape (3, num_points). Modifies :obj:`x`.

        Returns
        -------
        np.ndarray
            The transformed data

        Raises
        ------
        ValueError
            If NaN encountered in input
        """
        x -= np.mean(x, axis=1, keepdims=True)

        if np.any(np.isnan(x)):
            raise ValueError("NaN encountered in input.")

        # Normalize
        x = (x - self.bias) * self.scale
        return x


class PointCloudDatasetInMemory(Dataset):
    """
    PyTorch Dataset class to load point cloud data. Optionally, uses HDF5
    files to only read into memory what is necessary for one batch.
    """

    def __init__(
        self,
        data: np.ndarray,
        scalars: Dict[str, np.ndarray] = {},
        cms_transform: bool = False,
        scalar_requires_grad: bool = False,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Dataset of positions with shape (num_examples, 3, num_points)
        scalars : Dict[str, np.ndarray], default={}
            Dictionary of scalar arrays. For instance, the root mean squared
            deviation (RMSD) for each feature vector can be passed via
            :obj:`{"rmsd": np.array(...)}`. The dimension of each scalar array
            should match the number of input feature vectors N.
        cms_transform: bool
            If True, subtract center of mass from batch and shift and scale
            batch by the full dataset statistics.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            :obj:`scalar_dset_names`. Set to True, to use scalars for learning.
            If scalars are only required for plotting, then set it as False.
        """
        self.data = data
        self.scalars = scalars
        self.cms_transform = cms_transform
        self.scalar_requires_grad = scalar_requires_grad

        if self.cms_transform:
            self.transform = CenterOfMassTransform(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        data = self.data[idx].copy()  # shape (3, num_points)

        # CMS subtract
        if self.cms_transform:
            data = self.transform.transform(data)

        sample = {"X": torch.from_numpy(data)}
        # Add scalars
        for name, dset in self.scalars.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self.scalar_requires_grad
            )
        return sample
