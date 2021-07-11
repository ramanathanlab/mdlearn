import numpy as np


def kabsch(toXYZ: np.ndarray, fromXYZ: np.ndarray):
    r"""Aligns a single frame :obj:`fromXYZ` to another frame :obj:`toXYZ`
    using the kabsch method.

    Parameters
    ----------
    toXYZ : np.ndarray
        3 x N array of coordinates to align to.
    fromXYZ : np.ndarray
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
    shape1, shape2 = fromXYZ.shape, toXYZ.shape

    if shape1[1] != shape2[1]:
        raise ValueError("KABSCH: unequal array sizes")

    m1 = np.mean(fromXYZ, 1).reshape((shape1[0], 1))
    m2 = np.mean(toXYZ, 1).reshape((shape2[0], 1))
    tmp1 = np.tile(m1, shape1[1])
    tmp2 = np.tile(m1, shape2[1])

    assert np.allclose(tmp1, tmp2)
    assert tmp1.shape == fromXYZ.shape
    assert tmp2.shape == toXYZ.shape
    t1 = fromXYZ - tmp1
    t2 = toXYZ - tmp2

    u, _, wh = np.linalg.svd(np.dot(t2, t1.T))

    R = np.dot(
        np.dot(u, [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(u, wh))]]),
        wh,
    )
    T = m2 - np.dot(R, m1)

    tmp3 = np.reshape(np.tile(T, (shape2[1])), shape1)
    err = toXYZ - np.dot(R, fromXYZ) - tmp3

    eRMSD = np.sqrt(np.sum(err ** 2) / shape2[1])
    return R, T, eRMSD, err.T
