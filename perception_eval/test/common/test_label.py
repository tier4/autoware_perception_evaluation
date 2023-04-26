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
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
import pytest

autoware_non_merged_pairs = [
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

autoware_merged_pairs = [
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


traffic_light_classification_pairs = [
    (TrafficLightLabel.GREEN, "green"),
    (TrafficLightLabel.RED, "red"),
    (TrafficLightLabel.YELLOW, "yellow"),
    (TrafficLightLabel.RED_STRAIGHT, "red_straight"),
    (TrafficLightLabel.RED_LEFT, "red_left"),
    (TrafficLightLabel.RED_LEFT_STRAIGHT, "red_left_straight"),
    (TrafficLightLabel.RED_LEFT_DIAGONAL, "red_left_diagonal"),
    (TrafficLightLabel.RED_RIGHT, "red_right"),
    (TrafficLightLabel.RED_RIGHT_STRAIGHT, "red_right_straight"),
    (TrafficLightLabel.RED_RIGHT_DIAGONAL, "red_right_diagonal"),
    (TrafficLightLabel.YELLOW_RIGHT, "yellow_right"),
    (TrafficLightLabel.UNKNOWN, "unknown"),
]

traffic_light_non_classification_pairs = [
    (TrafficLightLabel.TRAFFIC_LIGHT, "traffic_light"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "green"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "yellow"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_left"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_left_straight"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_left_diagonal"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_right"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_right_straight"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "red_right_diagonal"),
    (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_right"),
    (TrafficLightLabel.UNKNOWN, "unknown"),
]


@pytest.mark.parametrize(
    "evaluation_task, merge_similar_labels, label_prefix, label_pairs",
    [
        (EvaluationTask.DETECTION, False, "autoware", autoware_non_merged_pairs),
        (EvaluationTask.DETECTION2D, True, "autoware", autoware_merged_pairs),
        (
            EvaluationTask.DETECTION2D,
            False,
            "traffic_light",
            traffic_light_non_classification_pairs,
        ),
        (
            EvaluationTask.CLASSIFICATION2D,
            False,
            "traffic_light",
            traffic_light_classification_pairs,
        ),
    ],
)
def test_label_converter(
    evaluation_task,
    merge_similar_labels,
    label_prefix,
    label_pairs: List[Tuple[LabelType, str]],
):
    label_converter = LabelConverter(
        evaluation_task=evaluation_task,
        merge_similar_labels=merge_similar_labels,
        label_prefix=label_prefix,
        count_label_number=True,
    )
    for autoware_label, name in label_pairs:
        assert autoware_label == label_converter.convert_name(name)
