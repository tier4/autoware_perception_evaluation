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

from numbers import Real
from typing import List, Optional, Union

from perception_eval.common.label import Label, LabelType


class LabelThreshold:
    """Label threshold interface for function `get_label_threshold`.

    Attributes:
    ----------
        semantic_label (Label): Label instance.
        target_labels (Optional[List[LabelType]]): List of LabeType instances.

    Args:
    ----
        semantic_label (Label): Label instance.
        target_labels: (Optional[List[LabelType]]): Target labels list.
    """

    def __init__(
        self,
        semantic_label: Label,
        target_labels: Optional[List[LabelType]],
    ) -> None:
        self.semantic_label: Label = semantic_label
        self.target_labels: Optional[List[LabelType]] = target_labels

    def get_label_threshold(
        self,
        threshold_list: List[float],
    ) -> Optional[float]:
        """Returns threshold at corresponding label index.

        If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
        and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

        Args:
        ----
            threshold_list (List[float]): Thresholds list.

        Returns:
        -------
            Optional[float]: Threshold at corresponding label index.

        Examples:
        --------
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
    semantic_label: Label,
    target_labels: Optional[List[LabelType]],
    threshold_list: Optional[List[float]],
) -> Optional[float]:
    """Returns threshold at corresponding label index.

    If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
    and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

    Args:
    ----
        semantic_label (Label): Label instance.
        target_labels (Optional[List[LabelType]]): List of LabeType instances.
        threshold_list (Optional[List[float]]): Thresholds list.

    Returns:
    -------
        Optional[float]: Threshold at corresponding label index. If there is no threshold, return None.

    Examples:
    --------
        >>> get_label_threshold(AutowareLabel.CAR, [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN], [1.0, 2.0])
        1.0
    """
    label_threshold: Optional[float] = None
    if target_labels is None:
        return None
    if threshold_list is None:
        return None

    if semantic_label.label in target_labels:
        label_index: int = target_labels.index(semantic_label.label)
        label_threshold = threshold_list[label_index]
    return label_threshold


def set_thresholds(
    thresholds: Union[Real, List[Real], List[List[Real]]],
    target_objects_num: int,
    nest: bool,
) -> Union[List[Real], List[List[Real]]]:
    """Returns thresholds list.

    If the threshold is `List[Real]`, convert to `List[List[Real]]`.

    Args:
    ----
        thresholds (Union[List[List[Real]], List[Real]]): Thresholds to be formatted.
        target_objects_num (int): The number of targets.
        nest (bool): Whether to return nested list.

    Returns:
    -------
        List[List[Real]]: Thresholds list.

    Examples:
    --------
        >>> set_thresholds([1.0, 2.0], 3)
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        >>> set_thresholds([[1.0, 2.0], [3.0, 4.0]] 2)
        [[1.0, 2.0], [3.0, 4.0]]
    """
    if nest:
        output: List[List[Real]] = __get_nested_thresholds(thresholds, target_objects_num)
        return check_nested_thresholds(output, target_objects_num)
    else:
        output: List[Real] = __get_thresholds(thresholds, target_objects_num)
        return check_thresholds(output, target_objects_num)


def __get_thresholds(threshold: Union[Real, List[Real]], num_elements: int) -> List[Real]:
    """Returns a list of thresholds.

    Args:
    ----
        thresholds (Union[Real, List[Real])
        num_elements (int): The number of elements.

    Returns:
    -------
        List[List[float]]: Thresholds list.

    Raises:
    ------
        ThresholdError

    Examples:
    --------
        >>> get_thresholds(1.0, 3)
        [1.0, 1.0, 1.0]
        >>> get_thresholds([1.0], 3)
        [1.0, 1.0, 1.0]
        >>> get_thresholds([1.0, 2.0, 3.0], 3)
        [1.0, 2.0, 3.0]
        # Error cases
        >>> get_thresholds([], 2)
        ThresholdsError: Empty list is invalid
        >>> get_threshold([1.0, [2.0]], 2)
        ThresholdsError: Type of all elements must be float, but got [1.0, [2.0]]
        >>> get_threshold([1.0, 2.0], 3)
    """
    if isinstance(threshold, Real):
        return [threshold] * num_elements

    if len(threshold) == 0:
        msg = "Empty list is invalid"
        raise ThresholdError(msg)
    elif any(not isinstance(t, Real) for t in threshold):
        msg = f"Type of all elements must be Real number, but got {threshold}"
        raise ThresholdError(msg)
    elif len(threshold) != 1 and num_elements != len(threshold):
        msg = f"Number of list elements must be {num_elements} or 1, but got {len(threshold)}"
        raise ThresholdError(msg)

    return threshold * num_elements if len(threshold) == 1 else threshold


def __get_nested_thresholds(
    threshold: Union[Real, List[Real], List[List[Real]]],
    num_elements: int,
) -> List[List[Real]]:
    """Returns a nested list of thresholds.

    Args:
    ----
        thresholds (Union[Real, List[Real], List[List[Real]]])
        num_elements (int): The number of elements.

    Returns:
    -------
        List[List[Real]]: Nested thresholds list.

    Raises:
    ------
        ThresholdError

    Examples:
    --------
        >>> get_nested_thresholds(1.0, 3)
        [[1.0, 1.0, 1.0]]
        >>> get_nested_thresholds([1.0, 2.0], 3)
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        >>> get_nested_thresholds([[1.0, 2.0], [3.0, 4.0]] 2)
        [[1.0, 2.0], [3.0, 4.0]]
        >>> get_nested_thresholds([[2.0], [3.0, 4.0]] 2)
        [[2.0, 2.0], [3.0, 4.0]]
        # Error cases
        >>> get_nested_thresholds([], 3)
        ThresholdsError: Empty list is invalid
        >>> get_nested_thresholds([1.0, [2.0]], 3)
        ThresholdsError: Type of all elements must be same, but got [1.0, [2.0]]
        >>> get_nested_thresholds([[1.0, 2.0, 3.0]], 2)
        ThresholdsError: For nested list, expected the number of each element is 2 or 1, but got [[1.0, 2.0, 3.0]]
    """
    if isinstance(threshold, Real):
        return [[threshold] * num_elements]

    if len(threshold) == 0:
        msg = "Empty list is invalid"
        raise ThresholdError(msg)

    if isinstance(threshold[0], Real):
        if any(not isinstance(t, Real) for t in threshold):
            msg = f"Type of all elements must be same, but got {threshold}"
            raise ThresholdError(msg)
        return [[t] * num_elements for t in threshold] if len(threshold) != num_elements else [threshold]
    else:
        if any(not isinstance(t, list) for t in threshold):
            msg = f"Type of all elements must be same but got {threshold}"
            raise ThresholdError(msg)
        elif any(len(t) != num_elements and len(t) != 1 for t in threshold):
            msg = f"For nested list, expected the number of each element is {num_elements} or 1, but got {threshold}"
            raise ThresholdError(
                msg,
            )
        threshold_list: List[List[Real]] = []
        for t in threshold:
            if len(t) == 1:
                threshold_list.append(t * num_elements)
            else:
                threshold_list.append(t)
        return threshold_list


class ThresholdError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def check_thresholds(thresholds: List[Real], num_elements: int) -> List[Real]:
    """Check whether threshold's shape is valid.

    Args:
    ----
        thresholds (List[Real]): Thresholds list.
        num_elements (int)

    Raises:
    ------
        ThresholdError

    Returns:
    -------
        List[Optional[List[Real]]]: Thresholds list.

    Examples:
    --------
        >>> check_thresholds([1.0, 2.0], 2)
        >>> [1.0, 2.0]
    """
    if any(not isinstance(t, Real) for t in thresholds):
        msg = f"Type of all elements must be Real number, but got {thresholds}"
        raise ThresholdError(msg)
    elif len(thresholds) != num_elements:
        msg = f"Expected the number of elements is {num_elements}, but got {len(thresholds)}"
        raise ThresholdError(
            msg,
        )
    return thresholds


def check_nested_thresholds(thresholds: List[List[Real]], num_elements: int) -> List[List[Real]]:
    """Check whether threshold's shape is valid.

    Args:
    ----
        thresholds (List[List[float]]): Thresholds list.
        num_threshold (int): Target number of elements.

    Raises:
    ------
        ThresholdError

    Returns:
    -------
        List[List[float]]: Thresholds list.

    Examples:
    --------
        >>> thresholds = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        >>> num_elements = 2
        >>> check_nested_thresholds(thresholds_list, num_elements)
        [[1.0, 1.0], [2.0, 2.0]]
    """
    if any(not isinstance(t, list) for t in thresholds):
        msg = f"Type of all elements must be list, but got {thresholds}"
        raise ThresholdError(msg)
    elif any(len(t) == 0 or len(t) != num_elements for t in thresholds):
        msg = f"Expected the number of each element is {num_elements}, but got {thresholds}"
        raise ThresholdError(msg)
    return thresholds
