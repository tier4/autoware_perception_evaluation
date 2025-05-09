# Copyright 2025 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.object import DynamicObject


# TODO(vividf): This class currently only supports 3D objects. Consider extending it to support 2D evaluation as well.
class PerceptionFrame:
    """The frame that included estimated objects and ground truth objects

    Args:
        unix_time (float): The unix time of the frame
        estimated_objects (List[DynamicObject]): The list of object result.
        frame_ground_truth (FrameGroundTruth): FrameGroundTruth instance.
    """

    def __init__(
        self,
        estimated_objects: List[DynamicObject],
        ground_truth_objects: FrameGroundTruth,
        unix_time: float,
    ) -> None:
        self.unix_time: float = unix_time

        self.estimated_objects: List[DynamicObject] = estimated_objects
        self.ground_truth_objects: FrameGroundTruth = ground_truth_objects

    def __reduce__(self) -> Tuple[PerceptionFrame, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (
            self.__class__,
            (
                self.estimated_objects,
                self.ground_truth_objects,
                self.unix_time,
            ),
        )

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "estimated_objects": [estimated_object.serialization() for estimated_object in self.estimated_objects],
            "ground_truth_objects": self.ground_truth_objects.serialization(),
            "unix_time": self.unix_time,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> PerceptionFrame:
        """Deserialize the data to PerceptionFrame."""

        return cls(
            estimated_objects=[DynamicObject.deserialization(obj) for obj in data["estimated_objects"]],
            ground_truth_objects=FrameGroundTruth.deserialization(data["ground_truth_objects"]),
            unix_time=data["unix_time"],
        )
