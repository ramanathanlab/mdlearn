"""Spatial decorrelation functions."""
import numpy as np
from typing import Optional

# TODO: get rid of np.matrix


def sd2(data: np.ndarray, m: Optional[int] = None, verbose: bool = False):
    r"""Perform spatial decorrelation of 2nd order of real signals.

    Parameters
    ==========
    data :  np.ndarray
        data array of shape (T, 3N) where T is the number of frames in the MD
        trajectory, N is the number of atoms in the system and 3 is due to the
        x,y,z coordinates for each atom.

    m : Optional[int], default=None
        Dimensionality of the subspace we are interested in. Default value is None,
        in which case m=n. If m is omitted, U is a square 3n x 3n matrix (as many
        sources as sensors).

    verbose : bool, default=False
        Print information on progress.

    Returns
    =======
    Y : np.ndarray
        A 3n x m matrix U (NumPy matrix type), such that :math:`Y = U x data` is a
        2nd order spatially whitened source extracted from the 3n x T data matrix
        :obj:`data` by performing PCA on :obj:`m` components of the real data.
        :obj:`Y` is a matrix of spatially uncorrelated components.
    S : np.ndarray
        Eigen values of the :obj:`data` covariance matrix.
    B : np.ndarray
        Eigen vectors of the :obj:`data` covariance matrix. The eigen vectors are
        orthogonal.
    U : np.ndarray
        The sphering matrix used to transform :obj:`data` by :math:`Y = U x data`.
    """
    if not isinstance(verbose, bool):
        raise TypeError("verbose parameter should be either True or False")

    if not isinstance(data, np.ndarray):
        raise ValueError(
            f"data (input data matrix) is of the wrong type ({type(data)})"
        )

    # Need to make a copy of the input array and use double precision (float64) .
    data_ = data.astype(np.float64)

    if len(data_.shape) != 2:
        raise ValueError(f"X has {len(data_.shape)} dimensions, should be 2")

    # T is number of samples, n is number of input signals
    T, n = data_.shape

    # Number of sources defaults to # of sensors
    if m is None:
        m = n

    if m > n:
        raise ValueError(
            f"SD2 -> Do not ask more sources ({m}) than sensors ({n})here!!!"
        )

    if verbose:
        print(f"2nd order Spatial Decorrelation -> Looking for {m} sources")
        print("2nd order Spatial Decorrelation -> Removing the mean value")

    data_ = data_.T

    # Remove the mean from data
    data_ -= data_.mean(1)

    # Whitening & projection onto signal subspace
    if verbose:
        print("2nd order Spatial Decorrelation -> Whitening the data")
    # An eigen basis for the sample covariance matrix
    [D, U] = np.linalg.eigh((data_ * data_.T) / float(T))
    # Sort by increasing variances
    k = D.argsort()
    Ds = D[k]
    # The m most significant princip. comp. by decreasing variance
    PCs = np.arange(n - 1, n - m - 1, -1)
    S = Ds[PCs]
    # At this stage, B does the PCA on m components
    B = U[:, k[PCs]].T

    # The scales of the principal components
    scales = np.sqrt(S)
    # Now, B does PCA followed by a rescaling = sphering
    U = np.diag(1.0 / scales) * B
    # Sphering
    Y = U * data_
    # B is a whitening matrix and Y is white.
    return Y, S, B.T, U
