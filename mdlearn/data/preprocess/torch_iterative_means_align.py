import math
import torch
import itertools
from mdlearn.data.preprocess.torch_kabsch_align import torch_kabsch


def torch_iterative_means(
    coords: torch.Tensor,
    eps: float = 0.001,
    max_iter: int = 10,
):
    """Run iterative means alignment with kabsch alignment

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of coordinates of (N_frames, 3, N_atoms)
    eps : float, optional
        Desired error tolerance for convergence, by default 0.001
    max_iter : int, optional
        Number of iterations for convergence, by default 10

    Returns
    -------
    [type]
        [description]
    """

    # all coordinates are expected to be passed as a (Ns x 3 x Na)  array
    # where Na = number of atoms; Ns = number of snapshots

    # This file has been edited to produce identical results as the original matlab implementation.

    n_frames, _, n_atoms = coords.shape

    print("Shape of array in IterativeMeans: {0}".format(coords.shape))

    avgCoords = []  # track average coordinates

    eRMSD = []

    for itr in itertools.count(1):
        tmpRMSD = []
        mnC = torch.mean(coords, dim=0)
        avgCoords.append(mnC)
        for i in range(n_frames):
            fromXYZ = coords[i]
            R, T, xRMSD, err = torch_kabsch(mnC, fromXYZ)
            tmpRMSD.append(xRMSD)
            tmp = torch.tile(T.flatten(), (n_atoms, 1)).T
            pxyz = torch.matmul(R, fromXYZ) + tmp
            coords[i, :, :] = pxyz
        eRMSD.append(torch.Tensor(tmpRMSD).T)
        newMnC = torch.mean(coords, dim=0)
        err = math.sqrt(sum((mnC.flatten() - newMnC.flatten()) ** 2))
        print(f"Iteration #{itr} with an error of {err}")
        if err <= eps or itr == max_iter:
            break  # Algorithm has converged

    return [itr, avgCoords, eRMSD, coords]
