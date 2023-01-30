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

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching import MatchingMode

from .utils import prepare_path


class PathDisplacementError:
    """[summary]
    A class to calculate path displacement errors for motion prediction task.

    Support metrics:
        ADE; Average Displacement Error
        FDE; Final Displacement Error
        Miss Rate
        Soft mAP

    Attributes:
        self.ade (float)
        self.fde (float)
        self.miss_rate (float)
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
        top_k: int = 1,
        num_waypoints: int = 10,
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
            top_k (int): Number of top kth confidential paths to be evaluated. Defaults to 1.
            num_waypoints (int): Number of horizontal frames. Defaults to 10.
            miss_tolerance (float): Tolerance value to determine miss[m]. Defaults to 2.0.
            kernel (Optional[str]): Target error kernel, min, max or None. Defaults to None.
                If it is specified, select the mode that total error is the smallest or largest.
                Otherwise, evaluate all modes.
        """
        self.num_ground_truth: int = num_ground_truth
        self.target_labels: List[LabelType] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list
        self.top_k: Optional[int] = top_k
        self.num_waypoints: Optional[int] = num_waypoints
        self.miss_tolerance: float = miss_tolerance
        self.kernel: Optional[str] = kernel

        all_object_results: List[DynamicObjectWithPerceptionResult] = []
        if len(object_results) == 0 or not isinstance(object_results[0], list):
            all_object_results = object_results
        else:
            for obj_results in object_results:
                all_object_results += obj_results
        self.objects_results_num: int = len(all_object_results)

        self.ade, self.fde, self.miss_rate = self._calculate_displacement_error(
            all_object_results,
            kernel=kernel,
        )

    def _calculate_displacement_error(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        kernel: Optional[str] = None,
    ) -> np.ndarray:
        """[summary]
        Returns the displacement error.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): List of DynamicObjectWithPerceptionResult.
            kernel (Optional[str]): Target error kernel, min, max or None. Defaults to None.
                If it is specified, select the mode that total error is the smallest or largest.
                Otherwise, evaluate all modes.

        Returns:
            ade (float): Average Displacement Error; ADE.
            fde (float): Finale Displacement Error; FDE.
            miss_rate (float): Miss rate.
        """
        sum_ade, sum_fde, sum_miss = 0.0, 0.0, 0.0
        num_ade, num_fde, num_miss = 0, 0, 0
        for obj_result in object_results:
            matching_threshold: float = get_label_threshold(
                semantic_label=obj_result.estimated_object.semantic_label,
                target_labels=self.target_labels,
                threshold_list=self.matching_threshold_list,
            )
            if not obj_result.is_result_correct(
                matching_mode=self.matching_mode,
                matching_threshold=matching_threshold,
            ):
                continue
            estimation, ground_truth = prepare_path(obj_result, self.top_k)
            # (K, T, 3)
            err: np.ndarray = estimation.get_path_error(
                ground_truth,
                self.num_waypoints,
            )

            if len(err) == 0:
                continue

            # NOTE: K, T is different for each agent
            distances: np.ndarray = np.linalg.norm(err[:, :, :2], axis=-1)
            if kernel == "min":
                distances = distances[np.argmin(distances.sum(axis=1))].reshape(1, -1)
            elif kernel == "max":
                distances = distances[np.argmax(distances.sum(axis=1))].reshape(1, -1)

            sum_ade += distances.sum()
            num_ade += distances.size
            sum_fde += distances[:, -1].sum()
            num_fde += distances[:, -1].size
            sum_miss += (self.miss_tolerance <= distances).sum()
            num_miss += distances.size

        ade: float = sum_ade / num_ade if 0 < num_ade else np.nan
        fde: float = sum_fde / num_fde if 0 < num_fde else np.nan
        miss_rate: float = sum_miss / num_miss if 0 < num_miss else np.nan

        return ade, fde, miss_rate
