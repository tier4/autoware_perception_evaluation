from __future__ import annotations

from typing import List
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .converter import LabelConverter
    from .types import LabelType


def set_target_lists(
    target_labels: Optional[List[str]],
    label_converter: LabelConverter,
) -> List[LabelType]:
    """Returns a LabelType list from a list of label names in string.

    If no label is specified, returns all LabelType elements.

    Args:
        target_labels (List[str]): The target class to evaluate
        label_converter (LabelConverter): Label Converter class

    Returns:
        List[LabelType]:  LabelType instance list.

    Examples:
        >>> converter = LabelConverter(False, "autoware")
        >>> set_target_lists(["car", "pedestrian"], converter)
        [<AutowareLabel.CAR: 'car'>, <AutowareLabel.PEDESTRIAN: 'pedestrian'>]
    """
    if target_labels is None or len(target_labels) == 0:
        return [label for label in label_converter.label_type]
    return [label_converter.convert_name(name) for name in target_labels]
