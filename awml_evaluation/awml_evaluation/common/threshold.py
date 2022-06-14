from typing import List
from typing import Optional
from typing import Union

from awml_evaluation.common.label import AutowareLabel


class LabelThreshold:
    """[summary]
    Label threshold interface for function "get_label_threshold"
    """

    def __init__(
        self,
        semantic_label: AutowareLabel,
        target_labels: Optional[List[AutowareLabel]],
    ) -> None:
        """[summary]
        Args:
            semantic_label (AutowareLabel): Target label
            target_labels: (Optional[List[AutowareLabel]]): Label list
        """
        self.semantic_label: AutowareLabel = semantic_label
        self.target_labels: Optional[List[AutowareLabel]] = target_labels

    def get_label_threshold(
        self,
        threshold_list: List[float],
    ) -> Optional[float]:
        """[summary]
        If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
        and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

        Args:
            threshold_list (List[float]): Thresholds list

        Returns:
            Optional[float]: The threshold for correspond label
        """
        return get_label_threshold(
            semantic_label=self.semantic_label,
            target_labels=self.target_labels,
            threshold_list=threshold_list,
        )


def get_label_threshold(
    semantic_label: AutowareLabel,
    target_labels: Optional[List[AutowareLabel]],
    threshold_list: Optional[List[float]],
) -> Optional[float]:
    """[summary]
    If target_labels is ["CAR", "PEDESTRIAN", "BIKE"] and threshold is [0.1, 0.2, 0.3],
    and object.semantic_label is "PEDESTRIAN", then LabelThreshold return 0.2.

    Args:
        semantic_label (AutowareLabel): Target label
        target_labels: (Optional[List[AutowareLabel]]): Label list
        threshold_list (Optional[List[float]]): Thresholds list

    Returns:
        Optional[float]: The threshold for correspond label.
                         If there is no threshold, return None
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
    """[summary]
    Set List[List[float]] thresholds
    If the threshold is List[float], convert to List[List[float]].

    Args:
        thresholds (Union[List[List[float]], List[float]]): THresholds to convert
        target_objects_num (int): The number of targets

    Returns:
        List[List[float]]: Thresholds list

    Examples:
        _set_thresholds([1.0, 2.0], 3)
        # [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        _set_thresholds([[1.0, 2.0], [3.0, 4.0]] 2)
        # [[1.0, 2.0], [3.0, 4.0]]
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
    target_labels: List[AutowareLabel],
    exception: Exception = ThresholdsError,
) -> List[float]:
    """[summary]
    Check the config and set the thresholds.

    Args:
        thresholds (Optional[List[float]]): Thresholds
        target_labels (List[AutowareLabel]): Target labels
        exception (Exception): The exception class. Defaults ThresholdError.

    Raises:
        ThresholdsError: Error for use case thresholds

    Returns:
        List[Optional[List[float]]]: A thresholds
    """
    if len(thresholds) != len(target_labels):
        raise exception(
            "Error: Thresholds is not proper! \
            The length of the thresholds is not same as target labels",
        )
    return thresholds


def check_thresholds_list(
    thresholds_list: List[List[float]],
    target_labels: List[AutowareLabel],
    exception: Exception = ThresholdsError,
) -> List[List[float]]:
    """[summary]
    Check the config and set the thresholds.

    Args:
        thresholds_list (List[List[float]]): A thresholds list.
        target_labels (List[AutowareLabel]): Target labels.
        exception (Exception): The exception class. Defaults ThresholdError.

    Raises:
        MetricThresholdsError: Error for metrics thresholds

    Returns:
        List[List[float]]: A thresholds list
    """
    for thresholds in thresholds_list:
        if len(thresholds) != 0 and len(thresholds) != len(target_labels):
            raise exception(
                "Error: Metrics threshold is not proper! \
                The length of the threshold is not same as target labels"
            )
    return thresholds_list
