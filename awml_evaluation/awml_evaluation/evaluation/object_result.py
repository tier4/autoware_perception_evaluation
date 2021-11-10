from typing import List
from typing import Tuple

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_between


class DynamicObjectWithResult:
    """[summary]
    Evaluation result class with each predicted object

    Attributes:
        self.predicted_object (DynamicObject): The predicted object by inference like CenterPoint
        self.ground_truth_object (DynamicObject): Ground truth object corresponding to
                                                  predicted object
        self.is_label_correct (bool): Whether the label of predicted_object is same as the label
                                      of ground truth object
        self.center_distance (float): The center distance between predicted object and ground
                                      truth object
        self.iou_3d (float): The 3d IoU between predicted object and ground truth object
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of Ground truth objects
        """
        self.predicted_object: DynamicObject = predicted_object

        # iou and is_label_correct
        self.ground_truth_object: DynamicObject
        self.is_label_correct: bool = False
        self.center_distance: float = 0.0
        self.iou_3d: float = 0.0

        (
            self.ground_truth_object,
            self.center_distance,
            self.iou_3d,
            self.is_label_correct,
        ) = self._get_evaluation_result(predicted_object, ground_truth_objects)

        # case a (distance result)
        self.distance_case_a: float = self._get_distance_case_a(
            predicted_object, ground_truth_objects
        )

        # case b (True or False result)
        self.is_passed_to_case_b: bool = self._is_passed_to_case_b(
            predicted_object, ground_truth_objects
        )

    @staticmethod
    def _get_evaluation_result(
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> Tuple[DynamicObject, float, float, bool]:
        """[summary]
        Get evaluation result

        Args:
            predicted_object (DynamicObject): The predicted object by inference like CenterPoint
            ground_truth_objects (List[DynamicObject]): The list of Ground truth objects

        Returns:
            Tuple[DynamicObject, float, float, bool]:
                min_distance_ground_truth_object (DynamicObject)
                min_distance (float)
                max_iou (float)
                is_label_correct (bool)
        """
        if not ground_truth_objects:
            return None, float("inf"), 0.0, False

        min_distance_ground_truth_object = ground_truth_objects[0]
        min_distance = distance_between(min_distance_ground_truth_object, predicted_object)

        # object which is min distance from the center of object
        for ground_truth_object in ground_truth_objects:
            center_distance = distance_between(ground_truth_object, predicted_object)
            if center_distance < min_distance:
                min_distance = center_distance
                min_distance_ground_truth_object = ground_truth_object

        max_iou = 0.0
        # for ground_truth_object in ground_truth_objects:
        #     iou = _get_iou_3d(predicted_object, ground_truths_object)
        #     if iou > max_iou:
        #         max_iou = iou
        #         max_iou_ground_truth_object = ground_truth_object

        # caliculate label
        if min_distance < 1000:
            if min_distance_ground_truth_object.semantic_label == predicted_object.semantic_label:
                is_label_correct = True
            else:
                is_label_correct = False
        else:
            is_label_correct = False
        return min_distance_ground_truth_object, min_distance, max_iou, is_label_correct

    @staticmethod
    def _get_distance_case_a(
        predicted_object: DynamicObject,
        ground_truth_objects: List[DynamicObject],
    ) -> float:
        # TODO impl
        pass

    @staticmethod
    def _is_passed_to_case_b(
        predicted_object: DynamicObject, ground_truth_objects: List[DynamicObject]
    ) -> bool:
        # TODO impl
        pass

    @staticmethod
    def _caliculate_iou_3d(
        predicted_object: DynamicObject, ground_truth_object: DynamicObject
    ) -> float:
        # TODO impl
        return 0.0
