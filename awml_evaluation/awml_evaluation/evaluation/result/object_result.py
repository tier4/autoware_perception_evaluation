from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_objects_bev
from awml_evaluation.evaluation.matching.object_matching import CenterDistanceMatching
from awml_evaluation.evaluation.matching.object_matching import IOU3dMatching
from awml_evaluation.evaluation.matching.object_matching import IOUBEVMatching
from awml_evaluation.evaluation.matching.object_matching import Matching
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.object_matching import PlaneDistanceMatching

logger = getLogger(__name__)


class DynamicObjectWithPerceptionResult:
    """[summary]
    Evaluation result for a predicted object

    Attributes:
        self.predicted_object (DynamicObject):
                The predicted object by inference like CenterPoint.
        self.ground_truth_object (Optional[DynamicObject]):
                Ground truth object corresponding to predicted object.
        self.is_label_correct (bool):
                Whether the label of predicted_object is same as the label of ground truth object
        self.center_distance (CenterDistanceMatching):
                The center distance between predicted object and ground truth object
        self.plane_distance (PlaneDistanceMatching):
                The plane distance for use case evaluation
        self.iou_bev (IOUBEVMatching):
                The bev IoU between predicted object and ground truth object
        self.iou_3d (IOU3dMatching):
                The 3d IoU between predicted object and ground truth object
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluation result for an object predicted object.

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of Ground truth objects
        """
        self.predicted_object: DynamicObject = predicted_object
        (
            self.ground_truth_object,
            self.ground_truth_object_index,
        ) = self._get_correspond_ground_truth_object(
            predicted_object,
            ground_truth_objects,
        )
        self.is_label_correct: bool = self._is_label_correct()

        # detection
        self.center_distance: CenterDistanceMatching = CenterDistanceMatching(
            self.predicted_object,
            self.ground_truth_object,
        )
        self.iou_bev: IOUBEVMatching = IOUBEVMatching(
            self.predicted_object,
            self.ground_truth_object,
        )
        self.iou_3d: IOU3dMatching = IOU3dMatching(
            self.predicted_object,
            self.ground_truth_object,
        )
        self.plane_distance: PlaneDistanceMatching = PlaneDistanceMatching(
            self.predicted_object,
            self.ground_truth_object,
        )

    def is_result_correct(
        self,
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> bool:
        """[summary]
        The function judging whether the result is target or not.

        Args:
            matching_mode (MatchingMode):
                    The matching mode to evaluate.
            matching_threshold (float):
                    The matching threshold to evaluate.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.

        Returns:
            bool: If label is correct and satisfy matching threshold, return True
        """
        # Whether is matching to ground truth
        matching: Matching = self.get_matching(matching_mode)
        is_matching_: bool = matching.is_better_than(matching_threshold)
        # Whether both label is true and matching is true
        is_correct: bool = self.is_label_correct and is_matching_
        return is_correct

    def get_matching(
        self,
        matching_mode: MatchingMode,
    ) -> Matching:
        """[summary]
        Get matching class

        Args:
            matching_mode (MatchingMode):
                    The matching mode to evaluate. Defaults to None.

        Raises:
            NotImplementedError: Not implemented matching class

        Returns:
            Matching: Matching class
        """
        if matching_mode == MatchingMode.CENTERDISTANCE:
            return self.center_distance
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            return self.plane_distance
        elif matching_mode == MatchingMode.IOUBEV:
            return self.iou_bev
        elif matching_mode == MatchingMode.IOU3D:
            return self.iou_3d
        else:
            raise NotImplementedError

    def get_distance_error_bev(self) -> float:
        """[summary]
        Get error center distance between ground truth and predicted object.

        Returns:
            float: error center distance between ground truth and predicted object.
        """
        return distance_objects_bev(self.predicted_object, self.ground_truth_object)

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

    def _get_correspond_ground_truth_object(
        self,
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> Optional[Tuple[DynamicObject, int]]:
        """[summary]
        Search correspond ground truth by minimum center distance

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of ground truth objects

        Returns:
            Optional[Tuple[DynamicObject, int]]: Correspond ground truth, index
        """
        if not ground_truth_objects:
            return (None, None)

        correspond_ground_truth_object: DynamicObject = ground_truth_objects[0]
        correspond_ground_truth_object_index: int = 0
        best_matching_distance: CenterDistanceMatching = CenterDistanceMatching(
            predicted_object=predicted_object,
            ground_truth_object=correspond_ground_truth_object,
        )

        # object which is min distance from the center of object
        for index, ground_truth_object in enumerate(ground_truth_objects):
            matching_distance: Matching = CenterDistanceMatching(
                predicted_object=predicted_object,
                ground_truth_object=ground_truth_object,
            )
            if best_matching_distance.value is not None:
                if matching_distance.is_better_than(best_matching_distance.value):
                    best_matching_distance = matching_distance
                    correspond_ground_truth_object = ground_truth_object
                    correspond_ground_truth_object_index = index
        return (correspond_ground_truth_object, correspond_ground_truth_object_index)
