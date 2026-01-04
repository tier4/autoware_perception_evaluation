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

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .path_displacement_error import PathDisplacementError


@dataclass(frozen=True)
class DisplacementErrorScores:
    """Dataclass for prediction displacement error."""

    ade: float
    fde: float
    miss_rate: float
    predict_num: int
    ground_truth_num: int
    top_k: int


class PredictionMetricsScore:
    """Metrics score manager for motion prediction task."""

    def __init__(
        self,
        nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        top_k: int = 3,
        miss_tolerance: float = 2.0,
        kernel: Optional[Literal["min", "max", "highest"]] = None,
    ) -> None:
        """Construct a new object.

        Args:
            nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]):
                The list of object results with their matching mode, label and threshold.
            num_ground_truth (int): The number of ground truth.
            target_labels (List[LabelType]): List of target label names.
            top_k (int, optional): The number of top K to be evaluated. Defaults to 1.
            miss_tolerance (float, optional): Threshold to determine miss. Defaults to 2.0.
            kernel (Optional[Literal["min", "max", "highest"]], optional): Kernel to evaluate displacement errors.
                "min" evaluates the minimum displacements at each time step.
                "max" evaluates the maximum at each time step.
                "highest" evaluates the highest confidence mode. Defaults to None.
        """
        self.target_labels = target_labels
        self.top_k = top_k
        self.miss_tolerance = miss_tolerance

        self.displacements: Dict[MatchingMode, Dict[LabelType, Dict[float, List[PathDisplacementError]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for matching_mode, label_object_result in nuscene_object_results.items():
            for target_label in target_labels:
                threshold_object_results = label_object_result[target_label]
                num_ground_truth = num_ground_truth_dict[target_label]
                for threshold, object_results in threshold_object_results.items():
                    displacement_err = PathDisplacementError(
                        object_results=object_results,
                        num_ground_truth=num_ground_truth,
                        target_labels=[target_label],
                        top_k=top_k,
                        miss_tolerance=miss_tolerance,
                        kernel=kernel,
                    )
                    self.displacements[matching_mode][target_label][threshold] = displacement_err

        self.displacement_error_scores = self._summarize_score()

    def _summarize_score(self) -> Dict[MatchingMode, Dict[float, DisplacementErrorScores]]:
        """Summarize scores."""
        # Matching mode -> threshold -> errors -> List of errors
        displacement_errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for matching_mode, label_displacements in self.displacements.items():
            for _, thresholds in label_displacements.items():
                for threshold, displacement_err in thresholds.items():
                    if ~np.isnan(displacement_err.ade):
                        displacement_errors[matching_mode][threshold]["ade"].append(displacement_err.ade)

                    if ~np.isnan(displacement_err.fde):
                        displacement_errors[matching_mode][threshold]["fde"].append(displacement_err.fde)

                    if ~np.isnan(displacement_err.miss_rate):
                        displacement_errors[matching_mode][threshold]["miss_rate"].append(displacement_err.miss_rate)

                    displacement_errors[matching_mode][threshold]["predict_num"].append(
                        displacement_err.objects_results_num
                    )
                    displacement_errors[matching_mode][threshold]["ground_truth_num"].append(
                        displacement_err.num_ground_truth
                    )

        displacement_error_scores = defaultdict(lambda: defaultdict(DisplacementErrorScores))
        for matching_mode, threshold_errors in displacement_errors.items():
            for threshold, errors in threshold_errors.items():
                ade_score = np.mean(errors["ade"]) if len(errors["ade"]) > 0 else np.nan
                fde_score = np.mean(errors["fde"]) if len(errors["fde"]) > 0 else np.nan
                miss_rate_score = np.mean(errors["miss_rate"]) if len(errors["miss_rate"]) > 0 else np.nan
                predict_num = int(np.sum(errors["predict_num"]))
                ground_truth_num = int(np.sum(errors["ground_truth_num"]))

                displacement_error_scores[matching_mode][threshold] = DisplacementErrorScores(
                    ade=ade_score,
                    fde=fde_score,
                    miss_rate=miss_rate_score,
                    predict_num=predict_num,
                    ground_truth_num=ground_truth_num,
                    top_k=self.top_k,
                )

        return displacement_error_scores

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"

        # Summarize overall classification scores
        for matching_mode, thresholds in self.displacement_error_scores.items():
            str_ += "---- Matching Mode: {}, Top K: {} ----\n".format(matching_mode.value, self.top_k)
            str_ += "|    Threshold | Predict Num | Ground Truth Num | ADE | FDE | Miss Rate | \n"
            str_ += "| :-------: | :---------: | :------: | :------: | :-------: | :----: |\n"
            for threshold, scores in thresholds.items():
                str_ += f"| {threshold} | {scores.predict_num} | {scores.ground_truth_num} | {scores.ade:.4f} | {scores.fde:.4f} | {scores.miss_rate:.4f} |\n"

        str_ += "\n"
        # === For each label ===
        # --- Table ---
        for matching_mode, label_displacement_errors in self.displacements.items():
            str_ += "---- Matching Mode: {}, Top K: {} ----\n".format(matching_mode.value, self.top_k)
            str_ += "|   Label | Threshold | Predict Num | Ground Truth Num | ADE | FDE | Miss Rate | \n"
            str_ += "| :-----: | :---: | :-------: | :---------: | :------: | :------: | :-------: |\n"
            for label, thresholds in label_displacement_errors.items():
                str_ += f"| {label} |"

                # Threshold column
                threshold_str = " / ".join([str(threshold) for threshold in thresholds.keys()])
                str_ += f" {threshold_str} |"

                # Predict Num column
                predict_num_str = " / ".join([str(v.objects_results_num) for v in thresholds.values()])
                str_ += f" {predict_num_str} |"

                # Ground truth column
                ground_truth_str = " / ".join([str(v.num_ground_truth) for v in thresholds.values()])
                str_ += f" {ground_truth_str} |"

                # ADE column
                ade_str = " / ".join([f"{v.ade:.4f}" for v in thresholds.values()])
                str_ += f" {ade_str} |"

                # FDE column
                fde_str = " / ".join([f"{v.fde:.4f}" for v in thresholds.values()])
                str_ += f" {fde_str} |"

                # Miss Rate column
                miss_rate_str = " / ".join([f"{v.miss_rate:.4f}" for v in thresholds.values()])
                str_ += f" {miss_rate_str} |"
                str_ += "\n"
            str_ += "\n"

        return str_
