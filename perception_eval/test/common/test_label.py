from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelConverter
import pytest

non_merged_pairs = [
    (AutowareLabel.BICYCLE, "bicycle"),
    (AutowareLabel.BICYCLE, "BICYCLE"),
    (AutowareLabel.MOTORBIKE, "motorbike"),
    (AutowareLabel.MOTORBIKE, "MOTORBIKE"),
    (AutowareLabel.CAR, "car"),
    (AutowareLabel.CAR, "CAR"),
    (AutowareLabel.BUS, "bus"),
    (AutowareLabel.BUS, "BUS"),
    (AutowareLabel.TRUCK, "truck"),
    (AutowareLabel.TRUCK, "TRUCK"),
    (AutowareLabel.TRUCK, "trailer"),
    (AutowareLabel.TRUCK, "TRAILER"),
    (AutowareLabel.PEDESTRIAN, "pedestrian"),
    (AutowareLabel.PEDESTRIAN, "PEDESTRIAN"),
    (AutowareLabel.UNKNOWN, "unknown"),
]

merged_pairs = [
    (AutowareLabel.BICYCLE, "bicycle"),
    (AutowareLabel.BICYCLE, "BICYCLE"),
    (AutowareLabel.BICYCLE, "motorbike"),
    (AutowareLabel.BICYCLE, "MOTORBIKE"),
    (AutowareLabel.CAR, "car"),
    (AutowareLabel.CAR, "CAR"),
    (AutowareLabel.CAR, "bus"),
    (AutowareLabel.CAR, "BUS"),
    (AutowareLabel.CAR, "truck"),
    (AutowareLabel.CAR, "TRUCK"),
    (AutowareLabel.CAR, "trailer"),
    (AutowareLabel.CAR, "TRAILER"),
    (AutowareLabel.PEDESTRIAN, "pedestrian"),
    (AutowareLabel.PEDESTRIAN, "PEDESTRIAN"),
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
