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

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.evaluation_task import set_task
from perception_eval.common.schema import FrameID


class TestEvaluationTask(unittest.TestCase):
    def test_segmentation_is_a_3d_task(self):
        task = EvaluationTask.SEGMENTATION
        self.assertTrue(task.is_3d())
        self.assertFalse(task.is_2d())
        self.assertFalse(task.is_fp_validation())
        self.assertEqual(EvaluationTask.from_value("segmentation"), task)
        self.assertEqual(set_task("segmentation"), task)
        self.assertEqual(FrameID.from_task("segmentation"), FrameID.BASE_LINK)
        self.assertEqual(FrameID.from_task(task), FrameID.BASE_LINK)

    def test_2d_task_set_unchanged(self):
        two_d = {task for task in EvaluationTask if task.is_2d()}
        self.assertEqual(
            two_d,
            {
                EvaluationTask.DETECTION2D,
                EvaluationTask.TRACKING2D,
                EvaluationTask.CLASSIFICATION2D,
                EvaluationTask.FP_VALIDATION2D,
            },
        )
        self.assertEqual(FrameID.from_task(EvaluationTask.DETECTION), FrameID.BASE_LINK)
        self.assertEqual(FrameID.from_task(EvaluationTask.TRACKING), FrameID.MAP)
        with self.assertRaises(ValueError):
            FrameID.from_task(EvaluationTask.DETECTION2D)


if __name__ == "__main__":
    unittest.main()
