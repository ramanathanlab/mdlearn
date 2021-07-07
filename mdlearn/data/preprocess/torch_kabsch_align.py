import torch


def torch_kabsch(toXYZ: torch.Tensor, fromXYZ: torch.Tensor):
    """
    Input is a 3 x N array of coordinates.
    """
    # This file has been edited to produce identical results as the original matlab implementation.
    shape1 = fromXYZ.shape
    shape2 = toXYZ.shape

    if not (shape1[1] == shape2[1]):
        raise ValueError("KABSCH: unequal array sizes")

    m1 = torch.mean(fromXYZ, dim=1).view((shape1[0], 1))
    m2 = torch.mean(toXYZ, dim=1).view((shape2[0], 1))
    tmp1 = torch.tile(m1, (shape1[1],))
    tmp2 = torch.tile(m1, (shape2[1],))

    # assert np.allclose(tmp1, tmp2)
    assert tmp1.shape == fromXYZ.shape
    assert tmp2.shape == toXYZ.shape
    t1 = fromXYZ - tmp1
    t2 = toXYZ - tmp2

    u, _, vh = torch.linalg.svd(torch.matmul(t2, t1.T))

    R = torch.matmul(
        torch.matmul(
            u,
            torch.Tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, torch.linalg.det(torch.matmul(u, vh))]]
            ),
        ),
        vh,
    )
    T = m2 - torch.matmul(R, m1)

    tmp3 = torch.tile(T, (shape2[1],)).view(shape1)
    err = toXYZ - torch.matmul(R, fromXYZ) - tmp3

    eRMSD = torch.sqrt(torch.sum(err ** 2) / shape2[1])
    return R, T, eRMSD, err.T