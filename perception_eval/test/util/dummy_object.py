# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from secrets import token_hex
from typing import List
from typing import Tuple

from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.object import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from pyquaternion.quaternion import Quaternion


def make_dummy_data(
    frame_id: FrameID = FrameID.BASE_LINK,
    use_unique_id: bool = True,
) -> Tuple[List[DynamicObject], List[DynamicObject]]:
    """[summary]
    Make dummy predicted objects and ground truth objects.

    Args:
        use_unique_id (bool): Whether use unique ID between different labels for estimated objects. Defaults to True.
            If False, it may have same ID in spite of different labels.
            For example,
                labels = [car, car, pedestrian]
                IDs = [0, 1, 0]

    Returns:
        List[DynamicObject], List[DynamicObject]: dummy_estimated_objects and
        dummy_ground_truth_objects
    """
    dummy_estimated_objects: List[DynamicObject] = [
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.5, 1.5, 1.5)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "0",
        ),
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(0.5, 0.5, 0.5)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.BICYCLE, "bicycle", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "0",
        ),
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 1.0, 1.0)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "1",
        ),
    ]
    dummy_ground_truth_objects: List[DynamicObject] = [
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 1.0, 1.0)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 1.0, 1.0)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.BICYCLE, "bicycle", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 1.0, 1.0)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.PEDESTRIAN, "pedestrian", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            frame_id=frame_id,
            position=(-1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 1.0, 1.0)),
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.MOTORBIKE, "motorbike", []),
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
    ]
    return dummy_estimated_objects, dummy_ground_truth_objects


def make_dummy_data2d(use_roi: bool = True) -> Tuple[List[DynamicObject2D], List[DynamicObject2D]]:
    """[summary]
    Make 2D dummy predicted objects and ground truth objects.

    Args:
        use_roi (bool): If False, roi is None and uuid will be specified. Defaults to True.

    Returns:
        List[DynamicObject2D], List[DynamicObject2D]: dummy_estimated_objects and dummy_ground_truth_objects.
    """
    frame_id = FrameID.CAM_FRONT
    dummy_estimated_objects: List[DynamicObject2D] = [
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            roi=(100, 100, 200, 100) if use_roi else None,
            uuid=token_hex(16) if use_roi else "0",
        ),
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.BICYCLE, "bicycle", []),
            roi=(0, 0, 50, 50) if use_roi else None,
            uuid=token_hex(16) if use_roi else "1",
        ),
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            roi=(200, 200, 200, 100) if use_roi else None,
            uuid=token_hex(16) if use_roi else "2",
        ),
    ]
    dummy_ground_truth_objects: List[DynamicObject2D] = [
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=1.0,
            semantic_label=Label(AutowareLabel.CAR, "car", []),
            roi=(100, 100, 200, 100) if use_roi else None,
            uuid=token_hex(16) if use_roi else "0",
        ),
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=1.0,
            semantic_label=Label(AutowareLabel.BICYCLE, "bicycle", []),
            roi=(0, 0, 50, 50) if use_roi else None,
            uuid=token_hex(16) if use_roi else "1",
        ),
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=1.0,
            semantic_label=Label(AutowareLabel.PEDESTRIAN, "pedestrian", []),
            roi=(200, 200, 200, 100) if use_roi else None,
            uuid=token_hex(16) if use_roi else "2",
        ),
        DynamicObject2D(
            unix_time=100,
            frame_id=frame_id,
            semantic_score=1.0,
            semantic_label=Label(AutowareLabel.MOTORBIKE, "motorbike", []),
            roi=(300, 100, 50, 50) if use_roi else None,
            uuid=token_hex(16) if use_roi else "3",
        ),
    ]
    return dummy_estimated_objects, dummy_ground_truth_objects
