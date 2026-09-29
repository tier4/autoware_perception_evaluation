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

import unittest

import numpy as np
from perception_eval.evaluation.metrics.segmentation.spatial import cluster_sizes
from perception_eval.evaluation.metrics.segmentation.spatial import count_clusters
from perception_eval.evaluation.metrics.segmentation.spatial import tolerant_error_mask
from scipy.spatial.distance import cdist


class TestSpatial(unittest.TestCase):
    def test_tolerant_error_mask_equals_bruteforce_reference(self):
        rng = np.random.default_rng(3)
        coord = rng.uniform(-20, 20, (2000, 3))
        target = rng.integers(0, 3, 2000)
        pred = target.copy()
        flip = rng.random(2000) < 0.15
        pred[flip] = rng.integers(0, 3, int(flip.sum()))

        def reference(radius: float) -> np.ndarray:
            wrong = pred != target
            mask = np.zeros(coord.shape[0], dtype=bool)
            distance = cdist(coord, coord)
            for index in np.flatnonzero(wrong):
                rescued = np.any((distance[index] <= radius) & (pred == target[index]))
                mask[index] = not rescued
            return mask

        for radius in (0.0, 0.8, 3.0):
            np.testing.assert_array_equal(tolerant_error_mask(coord, pred, target, radius), reference(radius))
        self.assertEqual(tolerant_error_mask(np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), 1.0).shape, (0,))

    def test_cluster_sizes_match_union_find(self):
        rng = np.random.default_rng(9)
        coord = rng.uniform(0, 10, (300, 3))
        radius = 0.9
        sizes = cluster_sizes(coord, radius)
        # Union-find reference.
        parent = list(range(300))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        distance = cdist(coord, coord)
        for i in range(300):
            for j in range(i + 1, 300):
                if distance[i, j] <= radius:
                    parent[find(i)] = find(j)
        roots = [find(i) for i in range(300)]
        expected = sorted(np.bincount(np.unique(roots, return_inverse=True)[1]).tolist())
        self.assertEqual(sorted(sizes.tolist()), expected)
        self.assertEqual(int(sizes.sum()), 300)

    def test_cluster_edge_cases(self):
        self.assertEqual(cluster_sizes(np.zeros((0, 3)), 1.0).shape, (0,))
        self.assertEqual(cluster_sizes(np.array([[0.0, 0, 0], [10.0, 0, 0]]), 1.0).tolist(), [1, 1])
        self.assertEqual(cluster_sizes(np.array([[0.0, 0, 0], [0.5, 0, 0]]), 0.0).tolist(), [1, 1])
        self.assertEqual(
            count_clusters(np.array([[0.0, 0, 0], [0.5, 0, 0], [10.0, 0, 0]]), 1.0, min_cluster_points=2), 1
        )


if __name__ == "__main__":
    unittest.main()
