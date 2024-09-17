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


from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching.object_matching import MatchingMode

from .utils import prepare_path


class SoftAp:
    """[summary]
    A class to calculate path displacement errors for motion prediction task.

    Support metrics:
        ADE; Average Displacement Error
        FDE; Final Displacement Error
        Miss Rate

    Attributes:
        self.num_ground_truth (int)
        self.target_labels (List[LabelType])
        self.matching_mode (MatchingMode)
        self.matching_threshold_list (List[float])
        self.top_k (Optional[int])
        self.num_waypoints (int): Number of waypoints of path. Defaults to 10.
        self.miss_tolerance (float): Tolerance value for miss rate.
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        top_k: Optional[int] = 1,
        num_waypoints: Optional[int] = 10,
        miss_tolerance: float = 2.0,
        kernel: Optional[str] = None,
    ) -> None:
        """[summary]

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]):
            num_ground_truth (int):
            target_labels (List[LabelType]):
            matching_mode (MatchingMode):
            matching_threshold_list (List[float])
            num_waypoints (int): The Number of horizontal waypoints. Defaults to 10[frames].
            top_k (Optional[int]): Number of top kth confidential paths. If None, calculate all paths. Defaults to None.
            miss_tolerance (float): Tolerance value to determine miss. Defaults to 2.0[m].
        """

        self.num_ground_truth: int = num_ground_truth

        self.target_labels: List[LabelType] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list
        self.top_k: Optional[int] = top_k
        self.num_waypoints: Optional[int] = num_waypoints
        self.miss_tolerance: float = miss_tolerance

        all_object_results: List[DynamicObjectWithPerceptionResult] = []
        if len(object_results) == 0 or not isinstance(object_results[0], list):
            all_object_results = object_results
        else:
            for obj_results in object_results:
                all_object_results += obj_results
        self.objects_results_num: int = len(all_object_results)

        # Calculate SoftAP
        tp_list = self._calculate_path_tp(all_object_results, kernel=kernel)
        self.ap = self._calculate_soft_ap(tp_list)

    def _calculate_path_tp(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        kernel: Optional[str] = None,
    ):
        tp_list: List[float] = [0.0 for _ in range(self.objects_results_num)]
        for i, object_result in enumerate(object_results):
            matching_threshold: float = get_label_threshold(
                semantic_label=object_result.estimated_object.semantic_label,
                target_labels=self.target_labels,
                threshold_list=self.matching_threshold_list,
            )
            if not object_result.is_result_correct(
                matching_mode=self.matching_mode,
                matching_threshold=matching_threshold,
            ):
                continue

            estimation, ground_truth = prepare_path(object_result, self.top_k)
            err = estimation.get_path_error(ground_truth, self.num_waypoints)  # (K, T, 3) or None

            if err is None or len(err) == 0:
                continue

            distances: np.ndarray = np.linalg.norm(err[:, :, :2], axis=-1)
            if kernel == "min":
                distances = distances[np.argmin(distances.sum(axis=1))].reshape(1, -1)
            elif kernel == "max":
                distances = distances[np.argmax(distances.sum(axis=1))].reshape(1, -1)

            if (distances < self.miss_tolerance).all():
                tp_list[i] = 1.0

        return np.cumsum(tp_list).tolist()

    def _calculate_precision_recall_list(
        self,
        tp_list: List[float],
    ) -> Tuple[List[float], List[float]]:
        """[summary]
        Calculate precision recall.

        Returns:
            Tuple[List[float], List[float]]: tp_list and fp_list

        Example:
            state
                tp_list = [1, 1, 2, 3]
                fp_list = [0, 1, 1, 1]
            return
                precision_list = [1.0, 0.5, 0.67, 0.75]
                recall_list = [0.25, 0.25, 0.5, 0.75]
        """
        precisions_list: List[float] = [0.0 for _ in range(len(tp_list))]
        recalls_list: List[float] = [0.0 for _ in range(len(tp_list))]

        for i in range(len(precisions_list)):
            precisions_list[i] = float(tp_list[i]) / (i + 1)
            if self.num_ground_truth > 0:
                recalls_list[i] = float(tp_list[i]) / self.num_ground_truth
            else:
                recalls_list[i] = 0.0

        return precisions_list, recalls_list

    def _calculate_soft_ap(self, tp_list: List[float]) -> float:
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
        precision_list, recall_list = self._calculate_precision_recall_list(tp_list)

        if len(precision_list) == 0:
            return np.nan

        max_precision_list, max_precision_recall_list = self.interpolate_precision_recall_list(
            precision_list,
            recall_list,
        )

        ap: float = 0.0
        for i in range(len(max_precision_list) - 1):
            score: float = max_precision_list[i] * (max_precision_recall_list[i] - max_precision_recall_list[i + 1])
            ap += score

        return ap

    def interpolate_precision_recall_list(
        self,
        precision_list: List[float],
        recall_list: List[float],
    ):
        """[summary]
        Interpolate precision and recall with maximum precision value per recall bins.

        Args:
            precision_list (List[float])
            recall_list (List[float])
        """
        max_precision_list: List[float] = [precision_list[-1]]
        max_precision_recall_list: List[float] = [recall_list[-1]]

        for i in reversed(range(len(recall_list) - 1)):
            if precision_list[i] > max_precision_list[-1]:
                max_precision_list.append(precision_list[i])
                max_precision_recall_list.append(recall_list[i])

        # append min recall
        max_precision_list.append(max_precision_list[-1])
        max_precision_recall_list.append(0.0)

        return max_precision_list, max_precision_recall_list
