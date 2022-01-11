from logging import getLogger
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.threshold import get_label_threshold
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import filter_ground_truth_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_object_results
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult

logger = getLogger(__name__)


class Ap:
    """[summary]
    AP class

    Attributes:
        self.ap (float):
                AP (Average Precision) result
        self.matching_average (Optional[float]):
                The average for matching score (ex. IoU, center distance).
                If there are no object results, this variable is None.
        self.matching_mode (MatchingMode):
                Matching mode like distance between the center of the object, 3d IoU
        self.matching_threshold (List[float]):
                The threshold list for matching the predicted object
        self.matching_standard_deviation (Optional[float]):
                The standard deviation for matching score (ex. IoU, center distance)
                If there are no object results, this variable is None.
        self.target_labels (List[AutowareLabel]):
                Target labels to evaluate
        self.tp_metrics (TPMetrics):
                The mode of TP (True positive) metrics. See TPMetrics class in detail.
        self.ground_truth_objects_num (int):
                The number of ground truth objects
        self.tp_list (List[float]):
                The list of the number of TP (True Positive) objects ordered by confidence
        self.fp_list (List[float]):
                The list of the number of FP (False Positive) objects ordered by confidence
    """

    def __init__(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
        target_labels: List[AutowareLabel],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            tp_metrics (TPMetrics): The mode of TP (True positive) metrics
            object_results (List[DynamicObjectWithResult]) : The results to each predicted object
            ground_truth_objects (List[DynamicObject]) : The ground truth objects for the frame
            target_labels (List[AutowareLabel]): Target labels to evaluate
            max_x_position (List[float]):
                    The threshold list of maximum x-axis position for each object.
                    Return the object that
                    - max_x_position < object x-axis position < max_x_position.
                    This param use for range limitation of detection algorithm.
            max_y_position (List[float]):
                    The threshold list of maximum y-axis position for each object.
                    Return the object that
                    - max_y_position < object y-axis position < max_y_position.
                    This param use for range limitation of detection algorithm.
            matching_mode (MatchingMode):
                    Matching mode like distance between the center of the object, 3d IoU
            matching_threshold (List[float]): The threshold list for matching the predicted object
        """

        self.tp_metrics: Union[TPMetricsAp, TPMetricsAph] = tp_metrics
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list

        # filter predicted object and results by iou_threshold and target_labels
        filtered_object_results: List[DynamicObjectWithResult] = filter_object_results(
            object_results=object_results,
            target_labels=self.target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
        )
        # sort by confidence
        lambda_func: Callable[
            [DynamicObjectWithResult], float
        ] = lambda x: x.predicted_object.semantic_score
        filtered_object_results.sort(key=lambda_func, reverse=True)
        self.objects_results_num: int = len(filtered_object_results)

        filtered_ground_truth_objects: List[DynamicObject] = filter_ground_truth_objects(
            objects=ground_truth_objects,
            target_labels=self.target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
        )
        self.ground_truth_objects_num: int = len(filtered_ground_truth_objects)

        # tp and fp from object results ordered by confidence
        self.tp_list: List[float] = []
        self.fp_list: List[float] = []
        self.tp_list, self.fp_list = self._calculate_tp_fp(
            tp_metrics=tp_metrics,
            object_results=filtered_object_results,
            ground_truth_objects_num=self.ground_truth_objects_num,
        )

        # caliculate precision recall
        precision_list: List[float] = []
        recall_list: List[float] = []
        precision_list, recall_list = self.get_precision_recall_list()

        # AP
        self.ap: float = Ap._calculate_ap(precision_list, recall_list)

        # average and standard deviation
        self.matching_average: Optional[float] = None
        self.matching_standard_deviation: Optional[float] = None
        self.matching_average, self.matching_standard_deviation = self._calculate_average_sd(
            object_results=object_results,
            matching_mode=self.matching_mode,
        )

    def save_precision_recall_graph(
        self,
        result_directory: str,
        frame_name: str,
    ) -> None:
        """[summary]
        Save visualization image of precision and recall

        Args:
            result_directory (str): The directory path to save images
            frame_name (str): The frame name
        """

        # base_name = f"{frame_name}_precision_recall_iou{self.iou_threshold}_"
        # target_str = f"{_get_flat_str(self.target_labels)}"
        # file_name = base_name + target_str + ".png"
        # file_path = os.join(result_directory, file_name)

        # precision_list: List[float] = []
        # recall_list: List[float] = []
        # precision_list, recall_list = self.get_precision_recall_list(
        #     self.tp_list,
        #     self.fp_list,
        #     self.ground_truth_objects_num,
        # )
        # save(file_path, recall_list, precision_list)

        raise NotImplementedError

    def get_precision_recall_list(
        self,
    ) -> Tuple[List[float], List[float]]:
        """[summary]
        Calculate precision recall.

        Returns:
            Tuple[List[float], List[float]]: tp_list and fp_list

        Example:
            state
                self.tp_list = [1, 1, 2, 3]
                self.fp_list = [0, 1, 1, 1]
            return
                precision_list = [1.0, 0.5, 0.67, 0.75]
                recall_list = [0.25, 0.25, 0.5, 0.75]
        """
        precisions_list: List[float] = [0.0 for i in range(len(self.tp_list))]
        recalls_list: List[float] = [0.0 for i in range(len(self.tp_list))]

        for i in range(len(precisions_list)):
            precisions_list[i] = float(self.tp_list[i]) / (i + 1)
            if self.ground_truth_objects_num > 0:
                recalls_list[i] = float(self.tp_list[i]) / self.ground_truth_objects_num
            else:
                recalls_list[i] = 0.0

        return precisions_list, recalls_list

    def _calculate_tp_fp(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects_num: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate TP (true positive) and FP (false positive).

        Args:
            tp_metrics (TPMetrics): The mode of TP (True positive) metrics
            object_results (List[DynamicObjectWithResult]): the list of objects with result
            ground_truth_objects_num (int): the number of ground truth objects

        Return:
            Tuple[tp_list, fp_list]

            tp_list (List[float]): the list of TP ordered by object confidence
            fp_list (List[float]): the list of FP ordered by object confidence

        Example:
            whether object label is correct [True, False, True, True]
            return
                tp_list = [1, 1, 2, 3]
                fp_list = [0, 1, 1, 1]
        """

        # When result num is 0
        if len(object_results) == 0:
            if ground_truth_objects_num != 0:
                logger.debug("The size of object_results is 0")
                return [], []
            else:
                return [0 * ground_truth_objects_num], list(range(1, ground_truth_objects_num))

        object_results_num = len(object_results)
        tp_list: List[float] = [0 for i in range(object_results_num)]
        fp_list: List[float] = [0 for i in range(object_results_num)]

        # label threshold
        matching_threshold_ = get_label_threshold(
            semantic_label=object_results[0].predicted_object.semantic_label,
            target_labels=self.target_labels,
            threshold_list=self.matching_threshold_list,
        )
        is_result_correct: bool = object_results[0].is_result_correct(
            matching_mode=self.matching_mode,
            matching_threshold=matching_threshold_,
        )
        tp_value: float = tp_metrics.get_value(object_results[0])
        if is_result_correct:
            tp_list[0] = tp_value
            fp_list[0] = 0
        else:
            tp_list[0] = 0
            fp_list[0] = tp_value

        for i in range(1, len(object_results)):
            matching_threshold_ = get_label_threshold(
                semantic_label=object_results[i].predicted_object.semantic_label,
                target_labels=self.target_labels,
                threshold_list=self.matching_threshold_list,
            )
            is_result_correct: bool = object_results[i].is_result_correct(
                matching_mode=self.matching_mode,
                matching_threshold=matching_threshold_,
            )
            tp_value: float = tp_metrics.get_value(object_results[i])
            if is_result_correct:
                tp_list[i] = tp_list[i - 1] + tp_value
                fp_list[i] = fp_list[i - 1]
            else:
                tp_list[i] = tp_list[i - 1]
                fp_list[i] = fp_list[i - 1] + 1.0

        return tp_list, fp_list

    @staticmethod
    def _calculate_ap(
        precision_list: List[float],
        recall_list: List[float],
    ) -> float:
        """[summary]
        Calculate AP (average precision)

        Args:
            precision_list (List[float]): The list of precision
            recall_list (List[float]): The list of recall

        Returns:
            float: AP

        Example:
            precision_list = [1.0, 0.5, 0.67, 0.75]
            recall_list = [0.25, 0.25, 0.5, 0.75]

            max_precision_list: List[float] = [0.75, 1.0, 1.0]
            max_precision_recall_list: List[float] = [0.75, 0.25, 0.0]

            ap = 0.75 * (0.75 - 0.25) + 1.0 * (0.25 - 0.0)
               = 0.625

        """

        if len(precision_list) == 0:
            return 0.0

        max_precision_list: List[float] = [precision_list[-1]]
        max_precision_recall_list: List[float] = [recall_list[-1]]

        for i in reversed(range(len(recall_list) - 1)):
            if precision_list[i] > max_precision_list[-1]:
                max_precision_list.append(precision_list[i])
                max_precision_recall_list.append(recall_list[i])

        # append min recall
        max_precision_list.append(max_precision_list[-1])
        max_precision_recall_list.append(0.0)

        ap: float = 0.0
        for i in range(len(max_precision_list) - 1):
            score: float = max_precision_list[i] * (
                max_precision_recall_list[i] - max_precision_recall_list[i + 1]
            )
            ap += score

        return ap

    @staticmethod
    def _calculate_average_sd(
        object_results: List[DynamicObjectWithResult],
        matching_mode: MatchingMode,
    ) -> Tuple[Optional[float], Optional[float]]:
        """[summary]
        Calculate average and standard deviation.

        Args:
            object_results (List[DynamicObjectWithResult]): The object results
            matching_mode (MatchingMode): [description]

        Returns:
            Tuple[float, float]: [description]
        """

        matching_score_list: List[float] = [
            object_result.get_matching(matching_mode).value for object_result in object_results
        ]
        matching_score_list_without_none = list(
            filter(lambda x: x is not None, matching_score_list)
        )
        if len(matching_score_list_without_none) == 0:
            return None, None
        mean: float = np.mean(matching_score_list_without_none)
        standard_deviation: float = np.std(matching_score_list_without_none)
        return mean, standard_deviation

    @staticmethod
    def _get_flat_str(str_list: List[str]) -> str:
        """
        Example:
            a = _get_flat_str([aaa, bbb, ccc])
            print(a) # aaa_bbb_ccc
        """
        output = ""
        for one_str in str_list:
            output = f"{output}_{one_str}"
        return output
