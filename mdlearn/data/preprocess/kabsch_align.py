import math
import numpy as np


def kabsch(toXYZ: np.ndarray, fromXYZ: np.ndarray):
    """
    Input is a 3 x N array of coordinates.
    """
    # This file has been edited to produce identical results as the original matlab implementation.
    len1 = np.shape(fromXYZ)
    len2 = np.shape(toXYZ)

    if not (len1[1] == len2[1]):
        print("KABSCH: unequal array sizes")
        return

    m1 = np.mean(fromXYZ, 1).reshape((len1[0], 1))
    # print np.shape(m1);
    m2 = np.mean(toXYZ, 1).reshape((len2[0], 1))
    tmp1 = np.tile(m1, len1[1])
    tmp2 = np.tile(m1, len2[1])

    assert np.allclose(tmp1, tmp2)
    assert tmp1.shape == fromXYZ.shape
    assert tmp2.shape == toXYZ.shape
    t1 = fromXYZ - tmp1
    t2 = toXYZ - tmp2

    [u, s, wh] = np.linalg.svd(np.dot(t2, t1.T))
    w = wh.T

    R = np.dot(
        np.dot(u, [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(u, w.T))]]),
        w.T,
    )
    T = m2 - np.dot(R, m1)

    tmp3 = np.reshape(np.tile(T, (len2[1])), (len1[0], len1[1]))
    err = toXYZ - np.dot(R, fromXYZ) - tmp3

    # eRMSD = math.sqrt(sum(sum((np.dot(err,err.T))))/len2[1]);
    eRMSD = math.sqrt(sum(sum(err ** 2)) / len2[1])
    return R, T, eRMSD, err.T


def wKabsch(toXYZ, fromXYZ, weights):
    len1 = np.shape(fromXYZ)
    # print 'len1: ', len1;
    len2 = np.shape(toXYZ)
    # print 'len2: ', len2;

    if not (len1[1] == len2[1]):
        print("wKABSCH: unequal array sizes")
        return

    dw = np.tile(weights, (3, 1))
    # print 'dw shape:', np.shape(dw);
    wFromXYZ = dw * fromXYZ
    # print 'wFromXYZ shape: ', np.shape(wFromXYZ);
    wToXYZ = dw * toXYZ
    # print 'wToXYZ shape: ', np.shape(wToXYZ);

    m1 = np.sum(wFromXYZ, 1) / np.sum(weights)
    # print np.shape(m1);
    m2 = np.sum(wToXYZ, 1) / np.sum(weights)
    # print np.shape(m2);

    tmp1 = np.reshape(np.tile(m1, (len1[1])), (len1[0], len1[1]))
    tmp2 = np.reshape(np.tile(m2, (len2[1])), (len2[0], len2[1]))
    t1 = np.reshape(fromXYZ - tmp1, (len1[0], len1[1]))
    # print 't1 shape: ', np.shape(t1);
    t2 = np.reshape(toXYZ - tmp2, (len2[0], len2[1]))

    aa = np.zeros((3, 3))
    for i in range(0, np.shape(t1)[1]):
        tmp = np.outer(t2[:, i], t1[:, i])
        # print 'tmp shape: ', np.shape(tmp);
        aa = aa + np.multiply(weights[i], tmp)
        aa = aa / np.sum(weights)

    [u, s, wh] = np.linalg.svd(aa)
    w = wh.T

    R = np.dot(
        np.dot(u, [[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(np.dot(u, w.T))]]),
        w.T,
    )
    T = m2 - np.dot(R, m1)

    tmp3 = np.reshape(np.tile(T, (len2[1])), (len1[0], len1[1]))
    err = toXYZ - np.dot(R, fromXYZ) - tmp3
    # eRMSD = math.sqrt(sum(sum((np.dot(err,err.T))))/len2[1]);
    eRMSD = math.sqrt(sum(sum(err ** 2)) / len2[1])
    return (R, T, eRMSD, err.T)


def wKabschDriver(toXYZ, fromXYZ, sMed=1.5, maxIter=20):
    scaleMed = sMed
    weights = np.ones(np.shape(toXYZ)[1])
    # print 'weights: ', np.shape(weights);
    Rc = []
    Tc = []
    sigc = []
    for _ in range(maxIter):
        R, T, eRMSD, err = wKabsch(toXYZ, fromXYZ, weights)
    Rc.append(R)
    Tc.append(T)
    tmp1 = np.reshape(
        np.tile(T, (np.shape(toXYZ[1]))),
        (np.shape(toXYZ)[0], np.shape(toXYZ)[1]),
    )
    deltaR = np.array(np.dot(R, fromXYZ) + tmp1 - toXYZ)
    # print 'deltaR shape: ', np.shape(deltaR);
    # print deltaR;
    # np.save('deltaR.npy', deltaR);
    nDeltaR = np.sqrt(np.sum(deltaR ** 2, axis=0))
    # print 'nDeltaR shape:', np.shape(nDeltaR);
    sig = scaleMed * np.median(nDeltaR)
    sigc.append(sig)
    weights = (sig ** 2) / ((sig ** 2 + nDeltaR ** 2) ** 2)
    # print np.shape(weights);
    return R, T, eRMSD, err
