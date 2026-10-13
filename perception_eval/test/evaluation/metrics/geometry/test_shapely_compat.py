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

import os
import re
import unittest

import numpy as np
from perception_eval.evaluation.metrics.geometry.shapely_compat import get_parts
from perception_eval.evaluation.metrics.geometry.shapely_compat import Prepared
from perception_eval.evaluation.metrics.geometry.shapely_compat import SpatialIndex
from perception_eval.evaluation.metrics.geometry.shapely_compat import union_all
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


class TestPrepared(unittest.TestCase):
    def setUp(self):
        self.square = Prepared(box(0.0, 0.0, 10.0, 10.0))

    def test_contains_points(self):
        xy = np.array([[5.0, 5.0], [10.5, 5.0], [0.0, 0.0], [9.999, 9.999]])
        np.testing.assert_array_equal(self.square.contains_points(xy), [True, False, False, True])
        self.assertEqual(self.square.contains_points(np.zeros((0, 2))).shape, (0,))
        self.assertEqual(Prepared(Polygon()).contains_points(xy).tolist(), [False] * 4)

    def test_intersects_and_covers(self):
        footprints = [box(9.0, 9.0, 12.0, 12.0), box(20.0, 20.0, 21.0, 21.0)]
        np.testing.assert_array_equal(self.square.intersects_many(footprints), [True, False])
        self.assertTrue(self.square.intersects(Point(1.0, 1.0)))
        self.assertFalse(self.square.intersects(Point(11.0, 1.0)))
        self.assertTrue(self.square.covers(LineString([(1.0, 1.0), (9.0, 9.0)])))
        self.assertFalse(self.square.covers(LineString([(1.0, 1.0), (11.0, 11.0)])))
        self.assertAlmostEqual(self.square.distance(Point(13.0, 5.0)), 3.0)
        self.assertFalse(Prepared(Polygon()).covers(Point(0.0, 0.0)))


class TestHelpers(unittest.TestCase):
    def test_union_all_and_get_parts(self):
        union = union_all([box(0, 0, 1, 1), box(0.5, 0, 1.5, 1)])
        self.assertAlmostEqual(union.area, 1.5)
        parts = get_parts(union_all([box(0, 0, 1, 1), box(5, 5, 6, 6)]))
        self.assertEqual(len(parts), 2)
        self.assertEqual(get_parts(Polygon()), [])
        self.assertEqual(len(get_parts(box(0, 0, 1, 1))), 1)
        self.assertTrue(union_all([]).is_empty)

    def test_spatial_index_matches_brute_force(self):
        rng = np.random.default_rng(0)
        geometries = [box(x, y, x + 2.0, y + 2.0) for x, y in rng.uniform(0, 50, (40, 2))]
        index = SpatialIndex(geometries)
        self.assertEqual(len(index), 40)
        for _ in range(10):
            x, y = rng.uniform(0, 50, 2)
            probe = box(x, y, x + 5.0, y + 5.0)
            expected = sorted(i for i, g in enumerate(geometries) if g.envelope.intersects(probe.envelope))
            self.assertEqual(index.query(probe).tolist(), expected)
        self.assertEqual(SpatialIndex([]).query(Point(0, 0)).shape, (0,))


class TestNoDirectShapelyImports(unittest.TestCase):
    def test_metric_modules_use_compat_layer(self):
        import perception_eval.evaluation.metrics as metrics

        root = os.path.dirname(metrics.__file__)
        offenders = []
        for dirpath, _, filenames in os.walk(root):
            if os.path.basename(dirpath) == "geometry":
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                with open(os.path.join(dirpath, filename)) as f:
                    for line in f:
                        if (
                            re.match(r"^(import shapely|from shapely)", line.strip())
                            and "shapely.geometry import" not in line
                        ):
                            offenders.append(os.path.relpath(os.path.join(dirpath, filename), root))
                            break
        self.assertEqual(offenders, [], f"use geometry/shapely_compat.py instead of shapely in {offenders}")


if __name__ == "__main__":
    unittest.main()
