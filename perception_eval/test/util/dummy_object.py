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
from perception_eval.common.object import DynamicObject
from pyquaternion.quaternion import Quaternion


def make_dummy_data(use_unique_id: bool = True) -> Tuple[List[DynamicObject], List[DynamicObject]]:
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
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.5, 1.5, 1.5),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "0",
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(0.5, 0.5, 0.5),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "0",
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16) if use_unique_id else "1",
        ),
    ]
    dummy_ground_truth_objects: List[DynamicObject] = [
        DynamicObject(
            unix_time=100,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.MOTORBIKE,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
            pointcloud_num=10,
        ),
    ]
    return dummy_estimated_objects, dummy_ground_truth_objects
