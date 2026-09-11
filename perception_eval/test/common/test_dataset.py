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

from copy import deepcopy
import pickle
from test.util.dummy_object import make_dummy_data
import unittest

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from pyquaternion import Quaternion


class TestFrameGroundTruthSceneId(unittest.TestCase):
    def _frame(self, scene_id=None) -> FrameGroundTruth:
        _, ground_truth = make_dummy_data()
        ego2map = HomogeneousMatrix((1.0, 2.0, 0.0), Quaternion(), src=FrameID.BASE_LINK, dst=FrameID.MAP)
        return FrameGroundTruth(
            unix_time=100, frame_name="0", objects=ground_truth, transforms=[ego2map], scene_id=scene_id
        )

    def test_default_is_none(self):
        self.assertIsNone(self._frame().scene_id)
        # Five-argument construction (pre-scene_id call sites) still works.
        _, ground_truth = make_dummy_data()
        self.assertIsNone(FrameGroundTruth(100, "0", ground_truth, None, None).scene_id)

    def test_pickle_and_deepcopy_preserve_scene_id(self):
        frame = self._frame("/data/scene_a")
        self.assertEqual(pickle.loads(pickle.dumps(frame)).scene_id, "/data/scene_a")
        self.assertEqual(deepcopy(frame).scene_id, "/data/scene_a")
        self.assertEqual(len(deepcopy(frame).objects), 4)

    def test_serialization_round_trip_and_missing_key(self):
        # Object serialization is exercised elsewhere; an empty frame isolates the scene_id handling.
        frame = FrameGroundTruth(unix_time=100, frame_name="0", objects=[], scene_id="scene_b")
        data = frame.serialization()
        self.assertEqual(data["scene_id"], "scene_b")
        self.assertIsNone(data["raw_data"])
        # Old serialized dicts have no scene_id key.
        data.pop("scene_id")
        data["transform_matrices_type"] = "list"
        restored = FrameGroundTruth.deserialization(data)
        self.assertIsNone(restored.scene_id)
        self.assertEqual(restored.frame_name, "0")


if __name__ == "__main__":
    unittest.main()
