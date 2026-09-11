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
from perception_eval.evaluation.metrics.confusion_report import confusion_cells
from perception_eval.evaluation.metrics.confusion_report import confusion_class_token


class TestConfusionReport(unittest.TestCase):
    def test_cells_flatten_true_by_pred(self):
        matrix = np.array([[5, 1], [2, 7]], dtype=np.int64)
        cells = confusion_cells(matrix, ("road", "obstacle"))
        self.assertEqual(cells["confusion_road__road"], 5.0)
        self.assertEqual(cells["confusion_road__obstacle"], 1.0)
        self.assertEqual(cells["confusion_obstacle__road"], 2.0)
        self.assertEqual(cells["confusion_obstacle__obstacle"], 7.0)
        self.assertEqual(len(cells), 4)

    def test_tokens_and_validation(self):
        self.assertEqual(confusion_class_token(0, None), "class_0")
        self.assertEqual(confusion_class_token(1, ("Car", "Big Truck")), "big_truck")
        with self.assertRaises(ValueError):
            confusion_class_token(2, ("car", "truck"))
        with self.assertRaises(ValueError):
            confusion_cells(np.zeros((2, 3)), None)
        with self.assertRaises(ValueError):
            confusion_cells(np.zeros((2, 2)), ("only",))


if __name__ == "__main__":
    unittest.main()
