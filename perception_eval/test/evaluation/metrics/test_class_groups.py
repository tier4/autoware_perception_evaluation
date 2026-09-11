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
from perception_eval.evaluation.metrics.class_groups import fold_confusion
from perception_eval.evaluation.metrics.class_groups import fold_labels
from perception_eval.evaluation.metrics.class_groups import fold_matrix
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.class_groups import validate_name_tokens

NAMES = ("car", "truck", "road", "sidewalk")
GROUPS = {"grouped_vehicle": ["car", "truck"], "grouped_flat": ["road", "sidewalk"]}


class TestClassGroups(unittest.TestCase):
    def test_full_taxonomy(self):
        lut, names = resolve_class_groups(NAMES, GROUPS)
        self.assertEqual(names, ("grouped_vehicle", "grouped_flat"))
        self.assertEqual(lut.tolist(), [0, 0, 1, 1])

    def test_singleton_and_forward_compat_member(self):
        lut, names = resolve_class_groups(
            NAMES,
            {
                "grouped_vehicle": ["car", "truck"],
                "grouped_road": ["road"],
                "grouped_walk": ["sidewalk", "ghost_point"],
            },
        )
        self.assertEqual(names, ("grouped_vehicle", "grouped_road", "grouped_walk"))
        self.assertEqual(lut.tolist(), [0, 0, 1, 2])

    def test_requires_full_partition(self):
        with self.assertRaisesRegex(ValueError, "cover every trained class"):
            resolve_class_groups(NAMES, {"grouped_vehicle": ["car", "truck"]})

    def test_rejects_absent_only_group(self):
        with self.assertRaisesRegex(ValueError, "no trained member"):
            resolve_class_groups(
                NAMES,
                {
                    "grouped_vehicle": ["car", "truck"],
                    "grouped_flat": ["road", "sidewalk"],
                    "grouped_ghost": ["ghost_point", "phantom"],
                },
            )

    def test_rejects_duplicate_membership_and_name_collision(self):
        with self.assertRaisesRegex(ValueError, "appears in both"):
            resolve_class_groups(NAMES, {"a": ["car", "truck"], "b": ["car", "road", "sidewalk"]})
        with self.assertRaisesRegex(ValueError, "collides with a trained class"):
            resolve_class_groups(NAMES, {"car": ["car", "truck"], "b": ["road", "sidewalk"]})
        with self.assertRaises(ValueError):
            resolve_class_groups(None, GROUPS)
        with self.assertRaises(ValueError):
            resolve_class_groups(NAMES, {})

    def test_fold_labels_leaves_ignore_untouched(self):
        lut, _ = resolve_class_groups(NAMES, GROUPS)
        self.assertEqual(fold_labels(np.array([0, 1, 2, 3, -1]), lut).tolist(), [0, 0, 1, 1, -1])

    def test_fold_confusion_and_matrix(self):
        lut, names = resolve_class_groups(NAMES, GROUPS)
        confusion = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int64)
        folded = fold_confusion(confusion, lut, len(names))
        self.assertEqual(folded.tolist(), [[2, 0], [0, 2]])
        fold = fold_matrix(lut, len(names))
        probabilities = np.array([[0.4, 0.0, 0.35, 0.25]])
        np.testing.assert_allclose(probabilities @ fold, [[0.4, 0.6]])

    def test_validate_name_tokens(self):
        with self.assertRaises(ValueError):
            validate_name_tokens(("Car", "car"))
        validate_name_tokens(("car", "truck"))
        validate_name_tokens(None)


if __name__ == "__main__":
    unittest.main()
