from typing import List
from typing import Optional

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
