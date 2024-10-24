from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mdlearn.data.preprocess.align.kabsch_align import kabsch


def _chunk_data(data: np.ndarray, partitions: int) -> list[np.ndarray]:
    chunk_size = len(data) // partitions
    chunks = [
        data[chunk_size * i : chunk_size * (1 + i)] for i in range(partitions)
    ]
    # Handle remainder
    chunks[-1] = np.concatenate(
        (chunks[-1], data[chunk_size * (partitions) :]),
    )
    return chunks


def _process_chunk(
    chunk: np.ndarray,
    mean_coord: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rmsds = np.zeros(len(chunk))
    for i in range(len(chunk)):
        rmsds[i], chunk[i], _ = kabsch(mean_coord, chunk[i], return_err=False)
    return rmsds, chunk


def iterative_means_align(
    coords: np.ndarray,
    eps: float = 0.001,
    max_iter: int = 10,
    inplace: bool = False,
    verbose: bool = False,
    num_workers: int = 1,
) -> tuple[int, list[np.ndarray], list[np.ndarray], np.ndarray]:
    r"""Run iterative means alignment.

    Run iterative means alignment which aligns :obj:`coords`
    to the mean coordinate structure using the kabsch alignment
    algorithm implemented here: :obj:`mdlearn.data.preprocess.align.kabsch_align`.
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
    inplace : bool, default=False
        If True, modifies :obj:`coords` inplace. Inplace operations may
        offer the ability to fit larger systems into memory.
    verbose : bool, default=False
        If True, prints verbose output
    num_workers : int, 1
        Number of workers for parallel processing of the trajectory, each
        worker will take a single core.

    Returns
    -------
    itr : int
        The iteration reached before convergence.
    avg_coords : List[np.ndarray]
        The average taken over all coords each iteration of the alignment.
    e_rmsd : List[np.ndarray]
        The root mean squared deviation (RMSD) of each structure with respect
        to the :obj:`avg_coords` for each iteration of the alignment.
    coords_ : np.ndarray
        The newly aligned trajectory of coordinates with the same shape as
        the input :obj:`coords` array.
    """
    coords_ = coords if inplace else coords.copy()

    if verbose:
        print('Shape of coords array in iterative_means:', coords_.shape)

    # Track average coordinates and RMSD
    avg_coords, e_rmsd = [], []

    # Precompute the first mean coordinate
    new_mean_coord = np.mean(coords_, axis=0)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for itr in itertools.count(1):
            # Reuse the previous mean calculation if possible
            mean_coord = new_mean_coord
            avg_coords.append(mean_coord)

            # Process trajectory in parallel chunks
            start_ind = 0
            itr_rmsds = np.zeros(len(coords_))
            chunks = _chunk_data(coords_, num_workers)
            for rmsds, chunk in executor.map(
                _process_chunk,
                chunks,
                [mean_coord] * len(chunks),
            ):
                # Collect RMSDs and aligned coordinates
                itr_rmsds[start_ind : start_ind + len(chunk)] = rmsds
                coords_[start_ind : start_ind + len(chunk)] = chunk
                start_ind += len(chunk)

            e_rmsd.append(itr_rmsds)
            new_mean_coord = np.mean(coords_, axis=0)
            err = np.linalg.norm(mean_coord - new_mean_coord)
            if verbose:
                print(f'Iteration #{itr} with an error of {err}')
            if err <= eps or itr == max_iter:
                break  # Algorithm has converged

    return itr, avg_coords, e_rmsd, coords_
