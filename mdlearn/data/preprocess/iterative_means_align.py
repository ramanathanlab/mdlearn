import itertools
import numpy as np
from mdlearn.data.preprocess.kabsch_align import kabsch


def iterative_means(
    coords: np.ndarray,
    eps: float = 0.001,
    max_iter: int = 10,
):
    r"""Run iterative means alignment which aligns :obj:`coords`
    to the mean coordinate structure using the kabsch alignment
    algorithm implemented here: :obj:`mdlearn.data.preprocess.kabsch_align.kabsch`.
    Algorithm converges if either the difference of means coordinates
    computed from consecutive iterations is less than :obj:`eps` or
    if :obj:`max_iter` iterations have finished.

    Parameters
    ----------
    coords : np.ndarray
        Array of atomic coordinates with dimension (n_frames, 3, n_atoms)
    eps : float, default=0.001
        Error tolerance of the difference between mean coordinates computed
        from consecutive iterations, used to define convergence.
    max_iter : int, default=10
        Number of iterations before convergence.

    Returns
    -------
    [type]
        [description]
    """
    n_frames, _, n_atoms = coords.shape

    print("Shape of coords array in iterative_means:", coords.shape)

    avg_coords = []  # track average coordinates

    e_rmsd = []

    for itr in itertools.count(1):
        tmp_rmsd = []
        mean_coord = np.mean(coords, 0)
        avg_coords.append(mean_coord)
        for i in range(n_frames):
            from_xyz = coords[i]
            R, T, x_rmsd, err = kabsch(mean_coord, from_xyz)
            tmp_rmsd.append(x_rmsd)
            tmp = np.tile(T.flatten(), (n_atoms, 1)).T
            pxyz = np.dot(R, from_xyz) + tmp
            coords[i, :, :] = pxyz
        e_rmsd.append(np.array(tmp_rmsd).T)
        new_mean_coord = np.mean(coords, 0)
        err = np.sqrt(np.sum((mean_coord.flatten() - new_mean_coord.flatten()) ** 2))
        print(f"Iteration #{itr} with an error of {err}")
        if err <= eps or itr == max_iter:
            break  # Algorithm has converged

    return itr, avg_coords, e_rmsd, coords
