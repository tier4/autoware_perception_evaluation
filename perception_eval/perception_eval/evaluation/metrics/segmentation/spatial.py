# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/spatial.py) at fcf86419.

"""Fixed-radius neighbourhood helpers for the point-level segmentation metrics.

The neighbourhood-tolerant error rate and the error-cluster count need spatial neighbour queries
on a frame's points. These are thin, tested wrappers over ``scipy.spatial.cKDTree`` so the metric
components stay focused on their own math.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def tolerant_error_mask(coord: NDArray, pred: NDArray, target: NDArray, radius: float) -> NDArray[np.bool_]:
    """Misclassified points that have no correct-class neighbour within ``radius``.

    A point ``p`` counts as an error only when it is wrong (``pred != target``) *and* no point
    ``q`` within ``radius`` was predicted as ``p``'s true class. At ``radius <= 0`` it reduces to
    the plain misclassification mask. Empty input returns an empty mask.
    """
    coord = np.asarray(coord, dtype=np.float64)
    pred = np.asarray(pred)
    target = np.asarray(target)
    n = coord.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    wrong = pred != target
    if not wrong.any() or radius <= 0.0:
        return wrong.copy()
    mask = wrong.copy()
    for true_class in np.unique(target[wrong]):
        candidates = pred == true_class
        if not candidates.any():
            continue  # nothing predicted this class anywhere, no rescue possible
        queries = np.flatnonzero(wrong & (target == true_class))
        tree = cKDTree(coord[candidates])
        counts = tree.query_ball_point(coord[queries], r=radius, return_length=True)
        mask[queries[np.asarray(counts) > 0]] = False
    return mask


def cluster_sizes(coord: NDArray, radius: float) -> NDArray[np.int64]:
    """Sizes of the connected components of the radius-neighbour graph.

    Two points are linked when they lie within ``radius`` and a cluster is a connected component.
    Returns one entry per cluster. Empty input returns an empty array.
    """
    coord = np.asarray(coord, dtype=np.float64)
    n = coord.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if radius <= 0.0:
        return np.ones((n,), dtype=np.int64)
    tree = cKDTree(coord)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.ones((n,), dtype=np.int64)  # every point is its own cluster
    data = np.ones(pairs.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.bincount(labels).astype(np.int64)


def count_clusters(coord: NDArray, radius: float, min_cluster_points: int = 1) -> int:
    """Number of radius-neighbour clusters with at least ``min_cluster_points`` points."""
    sizes = cluster_sizes(coord, radius)
    return int(np.sum(sizes >= min_cluster_points))
