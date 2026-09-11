# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Point-cloud semantic segmentation metrics (streaming, bounded memory)."""

from perception_eval.evaluation.metrics.segmentation.calibration import CalibrationError
from perception_eval.evaluation.metrics.segmentation.confident_error import ConfidentErrorRate
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionMatrix
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionState
from perception_eval.evaluation.metrics.segmentation.confusion_scores import Accuracy
from perception_eval.evaluation.metrics.segmentation.confusion_scores import IoU
from perception_eval.evaluation.metrics.segmentation.confusion_scores import PrecisionRecallF1
from perception_eval.evaluation.metrics.segmentation.entropy_auroc import UncertaintyUsefulness
from perception_eval.evaluation.metrics.segmentation.error_clusters import ErrorClusters
from perception_eval.evaluation.metrics.segmentation.partial_detection import PartialDetectionScore
from perception_eval.evaluation.metrics.segmentation.registry import COMPONENT_TYPES
from perception_eval.evaluation.metrics.segmentation.registry import ComponentSpec
from perception_eval.evaluation.metrics.segmentation.report import SegmentationMetricsReport
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrameSummary
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView
from perception_eval.evaluation.metrics.segmentation.suite import SegmentationMetricSuite
from perception_eval.evaluation.metrics.segmentation.tolerant_error import NeighbourhoodTolerantErrorRate

__all__ = (
    "Accuracy",
    "CalibrationError",
    "COMPONENT_TYPES",
    "ComponentSpec",
    "ConfidentErrorRate",
    "ConfusionMatrix",
    "ConfusionState",
    "ErrorClusters",
    "IoU",
    "NeighbourhoodTolerantErrorRate",
    "PartialDetectionScore",
    "PrecisionRecallF1",
    "SegmentationFrame",
    "SegmentationFrameSummary",
    "SegmentationMetricSuite",
    "SegmentationMetricsReport",
    "SegmentationView",
    "UncertaintyUsefulness",
)
