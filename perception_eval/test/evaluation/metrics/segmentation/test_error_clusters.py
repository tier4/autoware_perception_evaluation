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

import math
from test.evaluation.metrics.segmentation.helpers import make_frame
from test.evaluation.metrics.segmentation.helpers import make_view
import unittest

from perception_eval.evaluation.metrics.segmentation.error_clusters import ErrorClusters


class TestErrorClusters(unittest.TestCase):
    def test_merges_nearby_errors_into_one(self):
        target = [0] * 10
        pred = [1, 1, 1] + [0] * 7
        metric = ErrorClusters(cluster_radius=1.5)
        metric.update(make_view(make_frame(range(10), target, pred)))
        out = metric.compute()
        self.assertAlmostEqual(out["error_rate"], 0.3)
        self.assertAlmostEqual(out["error_cluster_count"], 1.0)
        self.assertAlmostEqual(out["error_clusters_per_frame"], 1.0)

    def test_counts_singleton_error(self):
        target = [0] * 10
        pred = [1] + [0] * 9
        metric = ErrorClusters(cluster_radius=0.5)
        metric.update(make_view(make_frame(range(10), target, pred)))
        self.assertAlmostEqual(metric.compute()["error_cluster_count"], 1.0)

    def test_clean_prediction(self):
        target = [0] * 10
        metric = ErrorClusters()
        metric.update(make_view(make_frame(range(10), target, target)))
        out = metric.compute()
        self.assertAlmostEqual(out["error_rate"], 0.0)
        self.assertAlmostEqual(out["error_cluster_count"], 0.0)

    def test_emits_global_and_per_class(self):
        # Road: 2 adjacent wrong points -> one cluster, obstacle points all correct.
        metric = ErrorClusters(cluster_radius=1.5)
        metric.update(make_view(make_frame([0, 1, 20, 30], target=[0, 0, 1, 1], pred=[1, 1, 1, 1])))
        out = metric.compute()
        self.assertAlmostEqual(out["error_rate"], 0.5)
        self.assertAlmostEqual(out["error_rate_road"], 1.0)
        self.assertAlmostEqual(out["error_cluster_count_road"], 1.0)
        self.assertAlmostEqual(out["error_rate_obstacle"], 0.0)
        self.assertAlmostEqual(out["error_cluster_count_obstacle"], 0.0)

    def test_min_cluster_points_and_per_frame(self):
        metric = ErrorClusters(cluster_radius=1.5, min_cluster_points=2)
        metric.update(make_view(make_frame([0, 1, 20], target=[0, 0, 0], pred=[1, 1, 1])))
        metric.update(make_view(make_frame([0, 1], target=[0, 0], pred=[0, 0])))
        metric.update(make_view(make_frame([0, 1], target=[0, 0], pred=[1, 1])))
        out = metric.compute()
        self.assertAlmostEqual(out["error_cluster_count"], 2.0)  # singleton at x=20 dropped
        self.assertAlmostEqual(out["error_clusters_per_frame"], 2.0 / 3.0)

    def test_empty_and_validation(self):
        metric = ErrorClusters()
        out = metric.compute()
        self.assertTrue(math.isnan(out["error_rate"]))
        self.assertTrue(math.isnan(out["error_clusters_per_frame"]))
        with self.assertRaises(ValueError):
            ErrorClusters(min_cluster_points=0)
        with self.assertRaises(ValueError):
            ErrorClusters(cluster_radius=-1.0)


if __name__ == "__main__":
    unittest.main()
