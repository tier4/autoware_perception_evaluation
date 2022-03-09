from typing import List
from typing import Tuple

from pyquaternion.quaternion import Quaternion

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject


def make_dummy_data() -> Tuple[List[DynamicObject], List[DynamicObject]]:
    """[summary]
    Make dummy predicted objects and ground truth objects.

    Returns:
        List[DynamicObject], List[DynamicObject]: dummy_predicted_objects and
        dummy_ground_truth_objects
    """
    dummy_predicted_objects: List[DynamicObject] = [
        DynamicObject(
            unix_time=100,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.5, 1.5, 1.5),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(0.5, 0.5, 0.5),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
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
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, -1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.MOTORBIKE,
            velocity=(1.0, 1.0, 1.0),
        ),
    ]
    return dummy_predicted_objects, dummy_ground_truth_objects
