# Copyright 2025 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

logger = getLogger(__name__)

_UNMATCHED_LABEL = "unmatched"
_CONFUSION_MATRIX_FOLDER_NAME = "detection_confusion_matrix"


@dataclass(frozen=True)
class ConfusionMatrixData:
    label: LabelType
    total_gt_nums: int
    total_tp_nums: int
    matched_boxes: Dict[str, int]  # {label: number of matched predicted boxes}

    @property
    def total_fn_nums(self) -> int:
        return self.total_gt_nums - self.total_tp_nums

    @property
    def total_prediction_nums(self) -> int:
        return sum(self.matched_boxes.values())

    @property
    def total_fp_nums(self) -> int:
        return self.total_prediction_nums - self.total_tp_nums


class DetectionConfusionMatrix:
    """
    Class to visualize a confusion matrix across all labels.

    Confusion matrix will be visualized as follows:
    ----------------------------------------------------------------------------------------
    |            | Predicted_label_1    | Predicted_label_2     | Predicted_label_3        |
    | GT_label_1 |  Num Matched Boxes   |    Num Matched Boxes   |     Num Matched Boxes    |
    | GT_label_2 |  Num Matched Boxes   |    Num Matched Boxes   |     Num Matched Boxes    |
    | GT_label_3 |  Num Matched Boxes   |    Num Matched Boxes   |     Num Matched Boxes    |
    ----------------------------------------------------------------------------------------

    Args:
        output_dir: Main output directory to save confusion matrix figures.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir / _CONFUSION_MATRIX_FOLDER_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_confusion_matrix(
        self,
        object_results_dict: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]],
        num_gts: Dict[LabelType, int],
        target_labels: List[LabelType],
    ) -> Dict[float, Dict[str, Dict[str, int]]]:
        """
        Compute confusion matrix for a dict of {Target labels: {Matching threshold: [Matched results]}}.
        Args:
            nuscene_object_results (Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]):
                A nested dictionary mapping from label → threshold to a list of matched object results.
            num_gts: Dict of {label: number of ground truth boxes}.
            target_labels: Targeted label classes to compute confusion matrices.

        Returns:
            A dict of confusion matrix with each matching threshold: {matching threshold: {label: ConfusionMatixData}}.
        """
        # confusion_matrix: {matching_threshold: {predicted_label: ConfusionMatrixData}}}
        matching_threshold_confusion_matrices = defaultdict(lambda: defaultdict(ConfusionMatrixData))

        for label in target_labels:
            for threshold, object_results in object_results_dict[label].items():
                total_tp_nums = 0
                matched_boxes = defaultdict(int)
                for object_result in object_results:
                    predicted_object_label = object_result.estimated_object.semantic_label.name

                    if object_result.ground_truth_object is not None:
                        if object_result.is_label_correct:
                            total_tp_nums += 1
                        matched_boxes[predicted_object_label] += 1
                    else:
                        matched_boxes[_UNMATCHED_LABEL] += 1

                if _UNMATCHED_LABEL not in matched_boxes:
                    matched_boxes[_UNMATCHED_LABEL] = 0

                total_gt_nums = num_gts[label]
                matching_threshold_confusion_matrices[threshold][label] = ConfusionMatrixData(
                    label=label, total_tp_nums=total_tp_nums, total_gt_nums=total_gt_nums, matched_boxes=matched_boxes
                )

        return matching_threshold_confusion_matrices

    def __call__(
        self,
        nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """
        Compute confusion matrix for each matching mode and thresholds, and then visualizing
        them in a figure.
        Args:
            nuscene_object_results: A nested dictionary of detection results where:
                - The first key is a MatchingMode (e.g., IoU),
                - The second key is a label (e.g., car, pedestrian),
                - The third key is a threshold (e.g., 0.5),
                - The value is either a list of DynamicObjectWithPerceptionResult instances.
            num_ground_truth: A dictionary mapping each label to the number of ground truth objects.
        """
        for matching_mode, label_to_threshold_map in nuscene_object_results.items():
            target_labels = list(label_to_threshold_map.keys())
            num_gt_dict = {label: num_ground_truth.get(label, 0) for label in target_labels}
            matching_confusion_matrices = self.compute_confusion_matrix(
                object_results_dict=label_to_threshold_map, num_gts=num_gt_dict, target_labels=target_labels
            )
            self.draw_confusion_matrix(
                matching_confusion_matrices=matching_confusion_matrices,
                target_labels=target_labels,
                matching_mode=matching_mode,
            )

    def draw_confusion_matrix(
        self,
        matching_confusion_matrices: Dict[float, Dict[LabelType, ConfusionMatrixData]],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        fig_size: Tuple[int] = (6, 6),
    ) -> None:
        """
        Draw confusion matrices in subplot and save them to a png figure.

        Args:
            matching_confusion_matrices: A dict of confusion matrix with each matching threshold:
                {matching threshold: {label: ConfusionMatixData}}.
            target_labels: Targeted ground truth labels.
            matching_mode: Matching method to match between predicted and ground truth boxes.
            fig_size: Total figure size for the confusion matrix.
        """
        num_thresholds = len(matching_confusion_matrices)
        cols = 2
        rows = int(np.ceil(num_thresholds / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * fig_size[0], rows * fig_size[1]))
        axes = axes.flatten()  # Flatten in case of single row/column

        for ax, (threshold, label_confusion_matrices) in zip(axes, matching_confusion_matrices.items()):
            # Row for ground truths, and col for predictions
            confusion_matrices = []
            cm_col_header = []
            cm_row_header = []

            for target_label in target_labels:
                confusion_matrix_data = label_confusion_matrices[target_label]
                # Ground_truth_label_1: [predicted_label_1_num, predicted_label_2_num, ...]
                confusion_matrices.append(
                    [confusion_matrix_data.matched_boxes.get(label.value, 0) for label in target_labels]
                )

                # Add UNMATCHED GTs (FN) to the row
                confusion_matrices[-1].append(confusion_matrix_data.total_fn_nums)
                cm_row_header.append(f"{target_label} ({confusion_matrix_data.total_gt_nums})")
                cm_col_header.append(f"{target_label}")

            # Add an UNMATCHED row (FPs) for predictions
            confusion_matrices.append(
                [label_confusion_matrices[label].matched_boxes.get(_UNMATCHED_LABEL, 0) for label in target_labels]
            )

            # UNMATCHED is not a label, so the last row and column (UNMATCHED x UNMATCHED) is always 0
            confusion_matrices[-1].append(0)

            confusion_matrix: npt.NDArray[np.int32] = np.array(confusion_matrices)

            # Row-wise sum for each column, and thus get the total predicted boxes for each label
            predicted_label_sums = np.sum(confusion_matrix, axis=0)

            # Add UNMATCHED label to the column header
            cm_col_header.append("FN")

            # Add the number to each column header
            cm_col_header = [
                f"{col_header} ({label_sum})" for col_header, label_sum in zip(cm_col_header, predicted_label_sums)
            ]

            # Add UNMATCHED label to the row header
            # Total FP (last row)
            total_fp_num = sum(confusion_matrix[-1])
            cm_row_header.append(f"FP ({total_fp_num})")

            # Plot
            title = f"Threshold: {threshold}"
            im = ax.imshow(confusion_matrix, cmap='Blues')
            ax.set_title(title)

            ax.set_xticks(np.arange(len(cm_col_header)))
            ax.set_yticks(np.arange(len(cm_row_header)))
            ax.set_xticklabels(cm_col_header, rotation=45, ha='right')
            ax.set_yticklabels(cm_row_header)

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            # Annotate cells
            for i in range(len(cm_row_header)):
                for j in range(len(cm_col_header)):
                    text_color = 'white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black'
                    ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color=text_color)

            # fig.colorbar(im, ax=ax)

        # Remove any unused subplots
        for ax in axes[num_thresholds:]:
            fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 0.9, 1.0])
        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        fig.suptitle(f"Matching mode: {matching_mode}", fontsize=16)

        matching_mode_file_name = str(matching_mode).lower().replace(" ", "_")
        output_file = self.output_dir / f"confusion_matrix_{matching_mode_file_name}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')  # High resolution

        # Optional: Close the figure to free memory
        plt.close(fig)
