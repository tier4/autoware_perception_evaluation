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

from abc import ABCMeta
from abc import abstractmethod
from math import pi

import numpy as np
from perception_eval.common.schema import FrameID
from perception_eval.result import DynamicObjectWithPerceptionResult


class TPMetrics(metaclass=ABCMeta):
    """Tp metrics meta class

    Attributes:
        mode (str): TP metrics name.
    """

    mode: str

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_value(
        self,
        object_result: DynamicObjectWithPerceptionResult,
    ) -> float:
        """Get TP metrics value

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP metrics value
        """
        pass


class TPMetricsAp(TPMetrics):
    """
    Ap metrics class

    Attributes:
        mode (str): TP metrics name that is TPMetricsAp.
    """

    mode: str = "TPMetricsAp"

    def get_value(
        self,
        object_result: DynamicObjectWithPerceptionResult,
    ) -> float:
        """Get TP (True positive) value.

        This function always returns 1.0.

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP value, 1.0.
        """
        return 1.0


class TPMetricsAph(TPMetrics):
    """
    Aph metrics class

    Attributes:
        mode (str): TP metrics name that is TPMetricsAph.
    """

    mode: str = "TPMetricsAph"

    def get_value(
        self,
        object_result: DynamicObjectWithPerceptionResult,
    ) -> float:
        """[summary]
        Get TP (True positive) value, the heading similarity rate using for APH.
        APH is used in evaluation for waymo dataset.

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: The heading similarity rate using for APH.
                   Calculate heading accuracy (1.0 - diff_heading / pi), instead of 1.0 for AP.
                   The minimum rate is 0 and maximum rate is 1.
                   0 means the heading difference is pi, and 1 means no heading difference.

        Reference:
                https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/metrics_utils.cc#L101-L116
        """
        if object_result.ground_truth_object is None:
            return 0.0

        # TODO: add support of transformation for objects are w.r.t map.
        ego2map = np.eye(4) if object_result.estimated_object.frame_id == FrameID.MAP else None
        pd_heading: float = object_result.estimated_object.get_heading_bev(ego2map=ego2map)
        gt_heading: float = object_result.ground_truth_object.get_heading_bev(ego2map=ego2map)
        diff_heading: float = abs(pd_heading - gt_heading)

        # Normalize heading error to [0, pi] (+pi and -pi are the same).
        if diff_heading > pi:
            diff_heading = 2.0 * pi - diff_heading
        # Clamp the range to avoid numerical errors.
        return min(1.0, max(0.0, 1.0 - diff_heading / pi))


class TPMetricsConfidence(TPMetrics):
    """Confidence TP class.

    Attributes:
        mode (str): TP metrics name that is TPMetricsConfidence.
    """

    mode: str = "TPMetricsConfidence"

    def get_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """[summary]
        Get TP (True positive) value, the heading similarity rate using with confidence.

        Args:
            object_result (DynamicObjectWithPerceptionResult): The object result

        Returns:
            float: TP (True positive) value, confidence score of estimated object.
        """
        return object_result.estimated_object.semantic_score
