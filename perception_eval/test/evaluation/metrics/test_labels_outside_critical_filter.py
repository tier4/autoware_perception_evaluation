# Copyright 2026 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric labels outside CriticalObjectFilterConfig.target_labels must not raise KeyError.

PerceptionFrameResult groups object results and ground-truth counts by the critical object
filter's target labels, while the metrics loop over evaluation_config_dict.target_labels.
A scenario that narrows the critical filter (e.g. ["pedestrian"]) while evaluating all
labels made the metrics raise KeyError(<AutowareLabel.CAR: 'car'>) as soon as a frame had
objects.
"""

import math

from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.classification.classification_metrics_score import ClassificationMetricsScore
from perception_eval.evaluation.metrics.prediction.prediction_metrics_score import PredictionMetricsScore
from perception_eval.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore

CRITICAL = [AutowareLabel.PEDESTRIAN]
METRIC_LABELS = [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN]


def test_prediction_metrics_tolerate_labels_outside_the_critical_filter():
    score = PredictionMetricsScore(
        object_results_dict={label: [] for label in CRITICAL},
        num_ground_truth_dict={label: 0 for label in CRITICAL},
        target_labels=METRIC_LABELS,
    )
    assert len(score.displacements) == len(METRIC_LABELS)
    assert math.isnan(score.ade)


def test_tracking_metrics_tolerate_labels_outside_the_critical_filter():
    score = TrackingMetricsScore(
        object_results_dict={label: [[], []] for label in CRITICAL},
        num_ground_truth_dict={label: 0 for label in CRITICAL},
        target_labels=METRIC_LABELS,
        matching_mode=MatchingMode.CENTERDISTANCE,
        matching_threshold_list=[1.0, 1.0],
    )
    assert score is not None


def test_classification_metrics_tolerate_labels_outside_the_critical_filter():
    score = ClassificationMetricsScore(
        object_results_dict={label: [] for label in CRITICAL},
        num_ground_truth_dict={label: 0 for label in CRITICAL},
        target_labels=METRIC_LABELS,
    )
    assert score is not None
