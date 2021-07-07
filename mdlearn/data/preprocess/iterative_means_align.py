import math
import itertools
import numpy as np
from typing import Tuple
from mdlearn.data.preprocess.kabsch_align import kabsch


def iterative_means(
    coords: np.ndarray,
    eps: float = 0.001,
    max_iter: int = 10,
    mapped: bool = False,
    fname: str = "na",
    shape: Tuple[int, int, int] = (0, 0, 0),
):

    if mapped:
        coords = np.memmap(fname, dtype="float64", mode="r+").reshape(shape)
    # all coordinates are expected to be passed as a (Ns x 3 x Na)  array
    # where Na = number of atoms; Ns = number of snapshots

    # This file has been edited to produce identical results as the original matlab implementation.

    Ns = np.shape(coords)[0]
    # dim = np.shape(coords)[1] # dim = 3
    Na = np.shape(coords)[2]

    print("Shape of array in IterativeMeans: {0}".format(np.shape(coords)))

    avgCoords = []  # track average coordinates

    eRMSD = []

    for itr in itertools.count(1):
        tmpRMSD = []
        mnC = np.mean(coords, 0)
        avgCoords.append(mnC)
        for i in range(Ns):
            fromXYZ = coords[i]
            R, T, xRMSD, err = kabsch(mnC, fromXYZ)
            tmpRMSD.append(xRMSD)
            tmp = np.tile(T.flatten(), (Na, 1)).T
            pxyz = np.dot(R, fromXYZ) + tmp
            coords[i, :, :] = pxyz
        eRMSD.append(np.array(tmpRMSD).T)
        newMnC = np.mean(coords, 0)
        err = math.sqrt(sum((mnC.flatten() - newMnC.flatten()) ** 2))
        print(f"Iteration #{itr} with an error of {err}")
        if err <= eps or itr == max_iter:
            break  # Algorithm has converged
    if mapped:
        del coords
        coords = 0
    return [itr, avgCoords, eRMSD, coords]
