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
from typing import Union

from perception_eval.common.label import LabelType


class LabelThreshold:
    """Label threshold interface for function `get_label_threshold`.

    Attributes:
        semantic_label (LabelType): Target label.
        target_labels (Optional[List[LabelType]]): Target labels list.

    Args:
        semantic_label (LabelType): Target label.
        target_labels: (Optional[List[LabelType]]): Target labels list.
    """

    def __init__(
        self,
        semantic_label: LabelType,
        target_labels: Optional[List[LabelType]],
    ) -> None:
        self.semantic_label: LabelType = semantic_label
        self.target_labels: Optional[List[LabelType]] = target_labels

    def get_label_threshold(
        self,
        threshold_list: List[float],
    ) -> Optional[float]:
        """Returns threshold at corresponding label index.

        If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
        and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

        Args:
            threshold_list (List[float]): Thresholds list.

        Returns:
            Optional[float]: Threshold at corresponding label index.

        Examples:
            >>> label_thresh = LabelThreshold(AutowareLabel.CAR, [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN])
            >>> label_thresh.get_label_threshold([1.0, 2.0])
            1.0
        """
        return get_label_threshold(
            semantic_label=self.semantic_label,
            target_labels=self.target_labels,
            threshold_list=threshold_list,
        )


def get_label_threshold(
    semantic_label: LabelType,
    target_labels: Optional[List[LabelType]],
    threshold_list: Optional[List[float]],
) -> Optional[float]:
    """Returns threshold at corresponding label index.

    If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
    and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

    Args:
        semantic_label (LabelTypes): Target label.
        target_labels: (Optional[List[LabelType]]): Label list.
        threshold_list (Optional[List[float]]): Thresholds list.

    Returns:
        Optional[float]: Threshold at corresponding label index. If there is no threshold, return None.

    Examples:
        >>> get_label_threshold(AutowareLabel.CAR, [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN], [1.0, 2.0])
        1.0
    """
    label_threshold: Optional[float] = None
    if target_labels is None:
        return None
    if threshold_list is None:
        return None

    if semantic_label in target_labels:
        label_index: int = target_labels.index(semantic_label)
        label_threshold = threshold_list[label_index]
    return label_threshold


def set_thresholds(
    thresholds: Union[List[List[float]], List[float]],
    target_objects_num: int,
) -> List[List[float]]:
    """Returns thresholds list.

    If the threshold is List[float], convert to List[List[float]].

    Args:
        thresholds (Union[List[List[float]], List[float]]): Thresholds to be formatted.
        target_objects_num (int): The number of targets.

    Returns:
        List[List[float]]: Thresholds list.

    Examples:
        >>> set_thresholds([1.0, 2.0], 3)
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        >>> set_thresholds([[1.0, 2.0], [3.0, 4.0]] 2)
        [[1.0, 2.0], [3.0, 4.0]]
    """
    assert len(thresholds) > 0, "The list of thresholds must be set, but got 0."

    thresholds_list: List[List[float]] = []
    if isinstance(thresholds[0], (int, float)):
        for element in thresholds:
            thresholds_list.append([element] * target_objects_num)
    elif isinstance(thresholds[0], list):
        assert len(thresholds) * target_objects_num == sum(
            [len(e) for e in thresholds]
        ), f"Invalid input: thresholds: {thresholds}, target_object_num: {target_objects_num}"
        thresholds_list = thresholds  # type: ignore
    else:
        raise ThresholdsError(f"Unexpected type: {type(thresholds[0])}")
    return thresholds_list


class ThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def check_thresholds(
    thresholds: List[float],
    target_labels: List[LabelType],
    exception: Exception = ThresholdsError,
) -> List[float]:
    """Check whether threshold's shape is valid.

    Args:
        thresholds (Optional[List[float]]): Thresholds list.
        target_labels (List[LabelType]): Target labels.
        exception (Exception): The exception class. Defaults to ThresholdError.

    Raises:
        Exception: In case of length of thresholds and labels are not same.

    Returns:
        List[Optional[List[float]]]: Thresholds list.

    Examples:
        >>> check_thresholds([1.0, 2.0], [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN])
        >>> [1.0, 2.0]
    """
    if len(thresholds) != len(target_labels):
        raise exception(
            "Error: Threshold is not proper! "
            + "The length of the thresholds' list is not same with target labels' one.",
        )
    return thresholds


def check_thresholds_list(
    thresholds_list: List[List[float]],
    target_labels: List[LabelType],
    exception: Exception = ThresholdsError,
) -> List[List[float]]:
    """Check whether threshold's shape is valid.

    Args:
        thresholds_list (List[List[float]]): Thresholds list.
        target_labels (List[LabelType]): Target labels.
        exception (Exception): The exception class. Defaults ThresholdError.

    Raises:
        Exception: Error for metrics thresholds.

    Returns:
        List[List[float]]: Thresholds list.

    Examples:
        >>> thresholds_list = [[1.0, 1.0], [2.0, 2.0]]
        >>> target_labels = [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN]
        >>> check_thresholds_list(thresholds_list, target_labels)
        [[1.0, 1.0], [2.0, 2.0]]
    """
    for thresholds in thresholds_list:
        if len(thresholds) != 0 and len(thresholds) != len(target_labels):
            raise exception(
                "Error: Metrics threshold is not proper! "
                + "The length of the thresholds' list is not same as target labels' one.",
            )
    return thresholds_list
