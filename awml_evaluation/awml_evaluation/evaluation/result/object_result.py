from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_objects
from awml_evaluation.common.object import distance_objects_bev
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.object_matching import get_iou_3d
from awml_evaluation.evaluation.matching.object_matching import get_uc_plane_distance

logger = getLogger(__name__)


class DynamicObjectWithResult:
    """[summary]
    Evaluation result for a predicted object

    Attributes:
        self.predicted_object (DynamicObject):
                The predicted object by inference like CenterPoint.
        self.ground_truth_object (Optional[DynamicObject]):
                Ground truth object corresponding to predicted object.
        self.is_label_correct (bool):
                Whether the label of predicted_object is same as the label of ground truth object
        self.center_distance (Optional[float]):
                The center distance between predicted object and ground truth object
        self.uc_plane_distance (Optional[float]):
                The plane distance for use case evaluation
        self.iou_3d (float): The 3d IoU between predicted object and ground truth object
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluation result for an object predicted object

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of Ground truth objects
        """
        self.predicted_object: DynamicObject = predicted_object

        self.ground_truth_object: Optional[DynamicObject] = None
        self.center_distance: Optional[float] = None
        (
            self.ground_truth_object,
            self.center_distance,
        ) = DynamicObjectWithResult._get_correspond_ground_truth_object(
            predicted_object,
            ground_truth_objects,
        )
        self.is_label_correct: bool = self._is_label_correct()

        # detection
        self.iou_3d: float = get_iou_3d(
            self.predicted_object,
            self.ground_truth_object,
        )
        self.uc_plane_distance: Optional[float] = get_uc_plane_distance(
            self.predicted_object,
            self.ground_truth_object,
        )

    def _is_label_correct(self) -> bool:
        """[summary]
        Get whether label is correct.

        Returns:
            bool: Whether label is correct
        """
        if self.ground_truth_object:
            return self.predicted_object.semantic_label == self.ground_truth_object.semantic_label
        else:
            return False

    def matching_threshold(
        self,
        matching_mode: Optional[MatchingMode],
    ) -> float:
        if matching_mode == MatchingMode.CENTERDISTANCE:
            return self.center_distance
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            return self.uc_plane_distance
        elif matching_mode == MatchingMode.IOU3d:
            return self.iou_3d
        else:
            raise NotImplementedError

    def is_matching(
        self,
        matching_mode: Optional[MatchingMode],
        matching_threshold: Optional[float],
    ) -> bool:
        is_matching_ = True
        if not matching_mode:
            is_matching_ = False
        elif not matching_threshold:
            is_matching_ = False
        elif matching_mode == MatchingMode.CENTERDISTANCE:
            is_matching_ = is_matching_ and self.center_distance < matching_threshold
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            is_matching_ = is_matching_ and self.uc_plane_distance < matching_threshold
        elif matching_mode == MatchingMode.IOU3d:
            is_matching_ = is_matching_ and self.iou_3d > matching_threshold
        else:
            raise NotImplementedError
        return is_matching_

    @property
    def distance_error_bev(self) -> float:
        """[summary]
        Get error center distance between ground truth and predicted object.

        Returns:
            float: error center distance between ground truth and predicted object.
        """
        return distance_objects_bev(self.predicted_object, self.ground_truth_object)

    @staticmethod
    def _get_correspond_ground_truth_object(
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> Tuple[Optional[DynamicObject], Optional[float]]:
        """[summary]
        Search correspond ground truth by minimum center distance

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of ground truth objects

        Returns:
            Optional[DynamicObject]: correspond ground truth
            Optional[float]: center distance between predicted object and ground truth object
        """
        if not ground_truth_objects:
            return None, None

        min_distance_ground_truth_object = ground_truth_objects[0]
        min_distance = distance_objects(predicted_object, min_distance_ground_truth_object)

        # object which is min distance from the center of object
        for ground_truth_object in ground_truth_objects:
            center_distance = distance_objects(predicted_object, ground_truth_object)
            if center_distance < min_distance:
                min_distance = center_distance
                min_distance_ground_truth_object = ground_truth_object

        return min_distance_ground_truth_object, min_distance
