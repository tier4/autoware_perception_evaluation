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
import pickle
import unittest

import numpy as np
from perception_eval.evaluation.metrics.component import compose_key
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.component import MetricReport
from perception_eval.evaluation.metrics.component import range_suffix
from perception_eval.evaluation.metrics.component import validate_ranges
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.naming import metric_token
from perception_eval.evaluation.metrics.naming import number_token
from perception_eval.evaluation.metrics.naming import threshold_token


class TestNaming(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(metric_token("Corner Error/car"), "corner_error_car")
        self.assertEqual(number_token(99.5), "99p5")
        self.assertEqual(number_token(-0.5), "minus0p5")
        self.assertEqual(threshold_token(0.5), "0p5m")
        self.assertEqual(threshold_token(2.0), "2m")
        self.assertEqual(label_metric_name(1, ("car", "Truck")), "truck")
        self.assertEqual(label_metric_name(3, None), "class_3")
        with self.assertRaises(ValueError):
            label_metric_name(2, ("car", "truck"))


class TestMetricRange(unittest.TestCase):
    def test_suffix(self):
        self.assertEqual(range_suffix(MetricRange("0-50m", 0.0, 50.0)), "0m_50m")
        self.assertEqual(MetricRange("far", 90.0, None).suffix, "90m_inf")
        self.assertEqual(MetricRange("half", 0.5, 1.5).suffix, "0p5m_1p5m")

    def test_contains(self):
        metric_range = MetricRange("0-50m", 0.0, 50.0)
        np.testing.assert_array_equal(metric_range.contains([0.0, 49.999, 50.0, 60.0]), [True, True, False, False])
        np.testing.assert_array_equal(MetricRange("far", 50.0, None).contains([49.0, 50.0]), [False, True])

    def test_validation(self):
        with self.assertRaises(ValueError):
            MetricRange("bad", 10.0, 5.0)
        with self.assertRaises(ValueError):
            MetricRange("bad", -1.0, 5.0)
        with self.assertRaises(ValueError):
            validate_ranges([MetricRange("a", 0.0, 50.0), MetricRange("b", 0.0, 50.0)])
        with self.assertRaises(ValueError):
            validate_ranges([MetricRange("a", 0.0, 50.0), MetricRange("a", 50.0, 90.0)])
        self.assertEqual(len(validate_ranges([MetricRange("a", 0.0, 50.0), MetricRange("b", 50.0, None)])), 2)


class TestComposeKey(unittest.TestCase):
    def test_levels_are_skipped_when_empty(self):
        self.assertEqual(compose_key("detection", "corner_mean_car"), "detection/corner_mean_car")
        self.assertEqual(
            compose_key(
                "detection", "corner_p95_car", taxonomy=None, filter_name="region_road", range_suffix_="0m_30m"
            ),
            "detection/region_road/0m_30m/corner_p95_car",
        )
        self.assertEqual(
            compose_key("segmentation", "error_clusters_per_frame", taxonomy="grouped", filter_name="region_road"),
            "segmentation/grouped/region_road/error_clusters_per_frame",
        )

    def test_rejects_slashes_and_empty_metric_key(self):
        with self.assertRaises(ValueError):
            compose_key("detection", "a/b")
        with self.assertRaises(ValueError):
            compose_key("detection", "")


class TestMetricReport(unittest.TestCase):
    def test_add_and_duplicate(self):
        report = MetricReport()
        report.add("detection/mAP", 0.5)
        with self.assertRaises(ValueError):
            report.add("detection/mAP", 0.6)
        report.update({"detection/road/0m_30m/corner_mean_car": 0.1})
        self.assertEqual(report.get("detection/mAP"), 0.5)
        self.assertEqual(report.select("detection/road"), {"detection/road/0m_30m/corner_mean_car": 0.1})

    def test_flat_keys(self):
        report = MetricReport(values={"detection/road/0m_30m/corner_mean_car": 0.1})
        self.assertEqual(report.to_flat_keys(), {"detection_road_0m_30m_corner_mean_car": 0.1})
        colliding = MetricReport(values={"a/b": 1.0, "a_b": 2.0})
        with self.assertRaises(ValueError):
            colliding.to_flat_keys()

    def test_summary_hides_confusion_cells_and_lists_coverage(self):
        report = MetricReport(
            values={"detection/confusion_car__truck": 1.0, "detection/ece": float("nan")},
            coverage={"region_road": (3, 5)},
        )
        report.warn("partial coverage")
        report.warn("partial coverage")
        text = report.summary()
        self.assertNotIn("confusion_car__truck", text)
        self.assertIn("detection/ece: nan", text)
        self.assertIn("region_road: 3/5", text)
        self.assertEqual(len(report.warnings), 1)

    def test_round_trips(self):
        report = MetricReport(values={"a/b": float("nan"), "a/c": 1.0}, coverage={"r": (1, 2)}, warnings=["w"])
        restored = MetricReport.deserialization(report.serialization())
        self.assertTrue(math.isnan(restored.values["a/b"]))
        self.assertEqual(restored.coverage, {"r": (1, 2)})
        self.assertEqual(restored.warnings, ["w"])
        pickled = pickle.loads(pickle.dumps(report))
        self.assertEqual(pickled.values["a/c"], 1.0)


class TestMeanValid(unittest.TestCase):
    def test_mean_valid(self):
        self.assertAlmostEqual(mean_valid([1.0, float("nan"), 3.0]), 2.0)
        self.assertTrue(math.isnan(mean_valid([float("nan")])))
        self.assertTrue(math.isnan(mean_valid([])))


if __name__ == "__main__":
    unittest.main()
