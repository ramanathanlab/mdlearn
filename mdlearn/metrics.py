from __future__ import annotations

import numpy as np


def metric_cluster_quality(
    data: np.ndarray,
    metric: np.ndarray,
    n_samples: int = 10000,
    n_neighbors: int = 10,
) -> float:
    sample_inds = np.random.choice(len(data), n_samples)
    from sklearn.neighbors import NearestNeighbors

    clf = NearestNeighbors(n_neighbors=n_neighbors)
    clf.fit(data[sample_inds])
    dists, inds = clf.kneighbors(data)

    metric_stdevs, dist_means = [], []
    for dist, ind in zip(dists, inds):
        # dist_means.append(np.mean(dist))
        metric_stdevs.append(np.std(metric[ind]))

    metric_stdevs = np.array(metric_stdevs)
    # dist_means = np.array(dist_means)
    return np.mean(metric_stdevs)

    return dist_means.dot(metric_stdevs) / len(data)
