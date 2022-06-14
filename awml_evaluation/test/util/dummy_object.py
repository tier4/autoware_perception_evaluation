from secrets import token_hex
from typing import List
from typing import Tuple

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from pyquaternion.quaternion import Quaternion


def make_dummy_data(use_unique_id: bool = True) -> Tuple[List[DynamicObject], List[DynamicObject]]:
    """[summary]
    Make dummy predicted objects and ground truth objects.

    Args:
        use_unique_id (bool): Whether use unique ID between different labels for estimated objects. Defaults to True.
            If False, it may have same ID inspite of different labels.
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
        ),
    ]
    return dummy_estimated_objects, dummy_ground_truth_objects
