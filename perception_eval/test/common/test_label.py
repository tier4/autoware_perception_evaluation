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
