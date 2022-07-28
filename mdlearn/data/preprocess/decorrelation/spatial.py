"""Spatial decorrelation functions."""
import warnings
from typing import Optional

import numpy as np


def SD2(data: np.ndarray, m: Optional[int] = None, verbose: bool = False):
    r"""Perform spatial decorrelation of 2nd order of real signals.

    Parameters
    ----------
    data :  np.ndarray
        data array of shape (T, 3N) where T is the number of frames in the MD
        trajectory, N is the number of atoms in the system and 3 is due to the
        x,y,z coordinates for each atom.

    m : Optional[int], default=None
        Dimensionality of the subspace we are interested in. Default value is None,
        in which case m=n. If m is omitted, U is a square 3n x 3n matrix (as many
        sources as sensors).

    verbose : bool, default=False
        Print progress.

    Returns
    -------
    Y : np.ndarray
        A 3n x m matrix U (NumPy matrix type), such that :math:`Y = U \times`:obj:`data` is a
        2nd order spatially whitened source extracted from the 3n x T data matrix
        :obj:`data` by performing PCA on :obj:`m` components of the real data.
        :obj:`Y` is a matrix of spatially uncorrelated components.
    S : np.ndarray
        Eigen values of the :obj:`data` covariance matrix.
    B : np.ndarray
        Eigen vectors of the :obj:`data` covariance matrix. The eigen vectors are
        orthogonal.
    U : np.ndarray
        The sphering matrix used to transform :obj:`data` by :math:`Y = U \times`:obj:`data`.

    Raises
    ------
    TypeError
        If :obj:`verbose` is not of type bool.
    TypeError
        If :obj:`data` is not of type np.ndarray.
    ValueError
        If :obj:`data` does not have 2 dimensions.
    ValueError
        If :obj:`m` is greater than 3N, the second dimension of :obj:`data`.
    """
    if not isinstance(verbose, bool):
        raise TypeError("verbose parameter should be either True or False")

    if not isinstance(data, np.ndarray):
        raise TypeError(f"data (input data matrix) is of the wrong type ({type(data)})")

    # Need to make a copy of the input array and use double precision (float64) .
    data = data.astype(np.float64)

    if len(data.shape) != 2:
        raise ValueError(f"data has {len(data.shape)} dimensions, should be 2")

    # T is number of samples, n is number of input signals
    T, n = data.shape

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

    data = data.T

    # Remove the mean from data
    data -= data.mean(1).reshape(-1, 1)

    # Whitening & projection onto signal subspace
    if verbose:
        print("2nd order Spatial Decorrelation -> Whitening the data")
    # An eigen basis for the sample covariance matrix
    [D, U] = np.linalg.eigh((np.dot(data, data.T)) / float(T))
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
    U = np.dot(np.diag(1.0 / scales), B)
    # Sphering
    Y = np.dot(U, data)
    # B is a whitening matrix and Y is white.
    return Y, S, B.T, U


def SD4(  # noqa: C901
    Y: np.ndarray,
    m: Optional[int] = None,
    U: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> np.ndarray:
    """SD4 - Spatial Decorrelation of 4th order of real signals.

    SD4 does joint diagonalization of cumulant matrices of order 4 to
    decorrelate the signals in spatial domain. It allows us to extract
    signals which are as independent as possible and which were not
    obtained while performing SD2. Here we consider signals which are
    spatially decorrelated of order 2, meaning that SD2 should be run first.

    Parameters
    ----------
    Y : np.ndarray
        An :obj:`n x T` spatially whitened matrix (:obj:`n` subspaces,
        :obj:`T` samples). May be a numpy array or matrix where :obj:`n`
        is the number of subspaces we are interested in and :obj:`T` is the
        number of frames in the MD trajectory.

    m : Optional[int], default=None
        The number of subspaces we are interested in. Defaults to None, in
        which case m=k.

    U : Optional[np.ndarray], default=None
        Whitening matrix obtained after doing the PCA analysis on :obj:`n`
        components of real data.

    verbose : bool, default=False
        Print progress.

    Returns
    -------
    W : np.ndarray
        Separating matrix which is spatially decorrelated of 4th order.

    Raises
    ------
    ValueError
        If :obj:`m` is greater than :obj:`n`, the first dimension of :obj:`Y`.
    """
    warnings.simplefilter("ignore", np.ComplexWarning)

    # TODO: update all computations to np.ndarray
    Y = np.matrix(Y)
    U = np.matrix(U)

    # n is number of input signals, T is number of samples
    [n, T] = Y.shape

    # Number of sources defaults to # of sensors
    if m is None:
        m = n

    if m > n:
        raise ValueError(
            f"SD4 -> Do not ask more sources ({m}) than sensors ({n})here!!!"
        )

    if verbose:
        print("4th order Spatial Decorrelation -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit
    Y = Y.T

    # Dimension of the space of real symm matrices
    dimsymm = ((m) * ((m) + 1)) / 2
    nbcm = dimsymm  # number of cumulant matrices
    # Storage for cumulant matrices
    CM = np.matrix(np.zeros([(m), int((m) * nbcm)], dtype=np.float64))
    R = np.matrix(np.eye(int(m), dtype=np.float64))
    Qij = np.matrix(np.zeros([m, m], dtype=np.float64))  # Temp for a cumulant matrix
    Xim = np.zeros(m, dtype=np.float64)  # Temp
    Xijm = np.zeros(m, dtype=np.float64)  # Temp
    # Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # We don't translate that one because numpy doesn't need Tony's rule

    # Use a symmetry trick to save storage.
    # Will index the columns of CM where to store the cumulant mats.
    Range = np.arange(m)

    # Removing 4th order spatial decorrelations
    for im in range(m):

        Xim = Y[:, im]
        Xijm = np.multiply(Xim, Xim)

        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion

        Qij = (
            np.dot(np.multiply(Xijm, Y).T, Y) / float(T)
            - R
            - 2 * np.dot(R[:, im], R[:, im].T)
        )

        # To ensure symmetricity of the covariance matrix a mathematical computation is done
        Qij = (Qij + Qij.T) / 2
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = np.multiply(Xim, Y[:, jm])
            Qij = (
                np.sqrt(2) * np.dot(np.multiply(Xijm, Y).T, Y) / float(T)
                - R[:, im] * R[:, jm].T
                - R[:, jm] * R[:, im].T
            )

            # To ensure symmetricity of the covariance matrix a mathematical computation is done
            Qij = (Qij + Qij.T) / 2
            CM[:, Range] = Qij
            Range = Range + m

    nbcm = int(nbcm)
    # Now we have nbcm = m(m+1) / 2 cumulants matrices stored in a big m x m*nbcm array.
    V = np.matrix(np.eye(m, dtype=np.float64))

    Diag = np.zeros(m, dtype=np.float64)
    On = 0.0
    Range = np.arange(m)
    for im in range(nbcm):
        Diag = np.diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (np.multiply(CM, CM).sum(axis=0)).sum(axis=0) - On

    # A statistically scaled threshold on `small" angles
    seuil = 1.0e-6 / (np.sqrt(T))
    encore = True
    sweep = 0  # Sweep number
    updates = 0  # Total number of rotations
    upds = 0  # Number of rotations in a given s
    g = np.zeros([2, nbcm], dtype=np.float64)
    gg = np.zeros([2, 2], dtype=np.float64)
    G = np.zeros([2, 2], dtype=np.float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper

    if verbose:
        print("SD4 -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose:
            print(f"SD4 -> Sweep #{sweep}")
        sweep = sweep + 1
        upds = 0

        for p in range(m - 1):
            for q in range(p + 1, m):

                Ip = np.arange(p, m * nbcm, m)
                Iq = np.arange(q, m * nbcm, m)

                # computation of Givens angle
                g = np.concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                Gain = (np.sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if np.abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = np.cos(theta)
                    s = np.sin(theta)
                    G = np.matrix([[c, -s], [s, c]])
                    pair = np.array([p, q])
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, np.concatenate([Ip, Iq])] = np.append(
                        c * CM[:, Ip] + s * CM[:, Iq],
                        -s * CM[:, Ip] + c * CM[:, Iq],
                        axis=1,
                    )
                    On = On + Gain
                    Off = Off - Gain

        if verbose:
            print(f"completed in {upds} rotations")
        updates = updates + upds
    if verbose:
        print(f"SD4 -> Total of {updates} Givens rotations")

    # A separating matrix
    W = V.T * U

    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance. Therefore, the sort is
    # according to the norm of the columns of A = pinv(W)

    if verbose:
        print("SD4 -> Sorting the components")

    A = np.linalg.pinv(W)
    keys = np.array(np.argsort(np.multiply(A, A).sum(axis=0)[0]))[0]
    W = W[keys, :]
    W = W[::-1, :]  # Is this smart ?

    if verbose:
        print("SD4 -> Fixing the signs")
    b = W[:, 0]
    signs = np.array(np.sign(np.sign(b) + 0.1).T)[0]  # just a trick to deal with sign=0
    W = np.diag(signs) * W
    # TODO: update W to be np.ndarray
    W = np.asarray(W)
    return W
