import numpy as np


def kabsch(to_xyz: np.ndarray, from_xyz: np.ndarray):
    r"""Aligns a single frame :obj:`fromXYZ` to another frame :obj:`toXYZ`
    using the kabsch method.

    Parameters
    ----------
    to_xyz : np.ndarray
        3 x N array of coordinates to align to.
    from_xyz : np.ndarray
        A 3 x N array of coordinates to align.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        If the arrays differ in the number of coordinates N.
    """

    if from_xyz.shape != to_xyz.shape:
        raise ValueError(
            f"KABSCH: unequal array sizes: {to_xyz.shape} mismatch {from_xyz.shape}"
        )

    dim, n_atoms = from_xyz.shape

    m1 = np.mean(from_xyz, 1).reshape((dim, 1))
    m2 = np.mean(to_xyz, 1).reshape((dim, 1))
    mean = np.tile(m1, n_atoms)

    t1 = from_xyz - mean
    t2 = to_xyz - mean

    u, _, wh = np.linalg.svd(np.dot(t2, t1.T))

    R = np.dot(
        np.dot(u, [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(u, wh))]]),
        wh,
    )
    T = m2 - np.dot(R, m1)

    tmp = np.reshape(np.tile(T, (n_atoms)), (dim, n_atoms))
    rotation = np.dot(R, from_xyz)
    err = to_xyz - rotation - tmp
    eRMSD = np.sqrt(np.sum(err ** 2) / n_atoms)
    new_xyz = rotation + tmp
    return eRMSD, err.T, new_xyz
