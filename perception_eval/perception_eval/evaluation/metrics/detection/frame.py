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

"""Immutable per-frame input of the advanced detection metrics.

``PerceptionEvaluationManager.add_frame_result()`` snapshots the filtered estimated and ground
truth objects (before any matching) into a :class:`DetectionFrame` so the scene-level suite can
re-match class-agnostically and apply spatial filters with frame boundaries intact.

This module deliberately imports only from ``perception_eval.common`` so that
``evaluation/result`` and ``manager`` can depend on it without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from numpy.typing import NDArray
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import TransformDict


@dataclass(frozen=True)
class DetectionFrame:
    """Filtered objects of one frame, retained for the advanced detection metrics.

    Attributes:
        frame_name (str): Frame name (sample index string in the dataset loader).
        unix_time (int): Frame timestamp.
        scene_id (str | None): Scene identifier used to resolve a lanelet map, or ``None``.
        estimated_objects (tuple[DynamicObject, ...]): Estimations after the manager-level filter.
        ground_truth_objects (tuple[DynamicObject, ...]): Ground truths after the manager-level filter.
        transforms (TransformDict | None): Frame transforms; ``(BASE_LINK, MAP)`` gives the ego pose.
    """

    frame_name: str
    unix_time: int
    scene_id: Optional[str]
    estimated_objects: Tuple[DynamicObject, ...]
    ground_truth_objects: Tuple[DynamicObject, ...]
    transforms: Optional[TransformDict] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimated_objects", tuple(self.estimated_objects))
        object.__setattr__(self, "ground_truth_objects", tuple(self.ground_truth_objects))

    @property
    def ego2map(self) -> Optional[NDArray]:
        """4x4 base_link -> map matrix, or ``None`` when the frame carries no ego pose."""
        if self.transforms is None:
            return None
        matrix = self.transforms.get((FrameID.BASE_LINK, FrameID.MAP))
        if matrix is None:
            inverse = self.transforms.get((FrameID.MAP, FrameID.BASE_LINK))
            if inverse is None:
                return None
            matrix = inverse.inv()
        return matrix.matrix

    @property
    def num_estimated(self) -> int:
        return len(self.estimated_objects)

    @property
    def num_ground_truth(self) -> int:
        return len(self.ground_truth_objects)

    def serialization(self) -> Dict[str, Any]:
        """Serialize into a dict (objects via their own ``serialization``)."""
        return {
            "frame_name": self.frame_name,
            "unix_time": self.unix_time,
            "scene_id": self.scene_id,
            "estimated_objects": [obj.serialization() for obj in self.estimated_objects],
            "ground_truth_objects": [obj.serialization() for obj in self.ground_truth_objects],
            "transforms": self.transforms.serialization() if self.transforms is not None else None,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> DetectionFrame:
        """Inverse of :meth:`serialization`."""
        transforms = data.get("transforms")
        return cls(
            frame_name=data["frame_name"],
            unix_time=data["unix_time"],
            scene_id=data.get("scene_id"),
            estimated_objects=tuple(DynamicObject.deserialization(obj) for obj in data["estimated_objects"]),
            ground_truth_objects=tuple(DynamicObject.deserialization(obj) for obj in data["ground_truth_objects"]),
            transforms=TransformDict.deserialization(transforms) if transforms is not None else None,
        )
