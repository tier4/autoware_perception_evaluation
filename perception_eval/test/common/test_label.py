from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelConverter
import pytest

non_merged_pairs = [
    (AutowareLabel.BICYCLE, "bicycle"),
    (AutowareLabel.MOTORBIKE, "motorbike"),
    (AutowareLabel.CAR, "car"),
    (AutowareLabel.BUS, "bus"),
    (AutowareLabel.TRUCK, "truck"),
    (AutowareLabel.PEDESTRIAN, "pedestrian"),
    (AutowareLabel.TRUCK, "truck"),
    (AutowareLabel.UNKNOWN, "unknown"),
]

merged_pairs = [
    (AutowareLabel.BICYCLE, "bicycle"),
    (AutowareLabel.BICYCLE, "motorbike"),
    (AutowareLabel.CAR, "car"),
    (AutowareLabel.CAR, "bus"),
    (AutowareLabel.CAR, "truck"),
    (AutowareLabel.PEDESTRIAN, "pedestrian"),
    (AutowareLabel.CAR, "truck"),
    (AutowareLabel.UNKNOWN, "unknown"),
]


@pytest.mark.parametrize(
    "merge_similar_labels, label_pairs",
    [
        (False, non_merged_pairs),
        (True, merged_pairs),
    ],
)
def test_label_converter(merge_similar_labels, label_pairs):
    label_converter = LabelConverter(merge_similar_labels=merge_similar_labels)
    for autoware_label, label in label_pairs:
        assert autoware_label == label_converter.convert_label(label)
