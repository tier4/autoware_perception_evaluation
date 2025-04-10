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
from perception_eval.evaluation import DynamicObjectWithPerceptionResult

from .utils import prepare_path


class PathDisplacementError:
    """A class to calculate path displacement errors for motion prediction task.

    Support Metrics:
        - ADE (Average Displacement Error)
        - FDE (Final Displacement Error)
        - Miss Rate
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_labels: List[LabelType],
        top_k: int = 3,
        miss_tolerance: float = 2.0,
        kernel: Optional[str] = None,
    ) -> None:
        """Construct a new object.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): List of object results.
            num_ground_truth (int): The number of GTs.
            target_labels (List[LabelType]): List of target label names.
            matching_mode (MatchingMode): Matching mode.
            matching_threshold_list (List[float]): List of matching thresholds.
            top_k (int, optional): The number of top K to be evaluated. Defaults to 1.
            miss_tolerance (float, optional): Threshold to determine miss. Defaults to 2.0.
            kernel (Optional[str], optional): Kernel of choose . Defaults to None.
        """
        self.num_ground_truth: int = num_ground_truth
        self.target_labels: List[LabelType] = target_labels
        self.top_k: Optional[int] = top_k
        self.miss_tolerance: float = miss_tolerance

        if kernel is not None and kernel not in ("min", "max"):
            raise ValueError(f"kernel must be min or max, but got {kernel}")

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
        num_ade, num_fde, num_path = 0, 0, 0
        for result in object_results:
            if result.ground_truth_object is None:
                continue

            estimation, ground_truth = prepare_path(result, self.top_k)

            err = estimation.get_path_error(ground_truth)  # (K, T, 3) or None

            if err is None or len(err) == 0:
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
            num_path += distances.size

        ade: float = sum_ade / num_ade if 0 < num_ade else np.nan
        fde: float = sum_fde / num_fde if 0 < num_fde else np.nan
        miss_rate: float = sum_miss / num_path if 0 < num_path else np.nan

        return ade, fde, miss_rate
