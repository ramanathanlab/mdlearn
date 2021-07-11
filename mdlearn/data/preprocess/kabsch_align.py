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
    shape1, shape2 = from_xyz.shape, to_xyz.shape

    if shape1[1] != shape2[1]:
        raise ValueError("KABSCH: unequal array sizes")

    m1 = np.mean(from_xyz, 1).reshape((shape1[0], 1))
    m2 = np.mean(to_xyz, 1).reshape((shape2[0], 1))
    mean = np.tile(m1, shape1[1])

    t1 = from_xyz - mean
    t2 = to_xyz - mean

    u, _, wh = np.linalg.svd(np.dot(t2, t1.T))

    R = np.dot(
        np.dot(u, [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(u, wh))]]),
        wh,
    )
    T = m2 - np.dot(R, m1)

    tmp3 = np.reshape(np.tile(T, (shape2[1])), shape1)
    err = to_xyz - np.dot(R, from_xyz) - tmp3

    eRMSD = np.sqrt(np.sum(err ** 2) / shape2[1])
    return R, T, eRMSD, err.T
