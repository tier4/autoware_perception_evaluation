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

from abc import ABC
from abc import abstractmethod
import stat

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class TPErrorMetric(ABC):
    """TP error metric base class

    Attributes:
        mode (str): TP error metric name.
    """

    mode: str
    average_mode: str
    mean_average_mode: str

    def __init__(self) -> None:
        super().__init__()
        self.values: np.ndarray = np.array([])
        self.confidences: np.ndarray = np.array([])
        self.avg_metric: float = np.nan
        self.interpolated_values: np.ndarray = np.array([])
        self.optimal_avg_metric: float = np.nan

    @abstractmethod
    def compute_value(
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

    @abstractmethod
    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        pass

    def compute_average_value(self, target_label: LabelType, min_recall: float, max_recall_ind: int) -> float:
        """
        Get average value of TP error metric.
        Taken from nuScenes-devkit.
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py#L181

        Args:
            min_recall (float): Minimum recall.

        Returns:
            float: Average value of TP error metric.
        """
        if self.ignore_target_labels(target_label):
            return np.nan

        first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
        last_ind = max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
        if last_ind < first_ind:
            return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0
        else:
            return np.mean(self.interpolated_values[first_ind : last_ind + 1])  # +1 to include error at max recall.

    def compute_optimal_average_value(self, optimal_conf: int) -> float:
        """
        Get optimal average value of TP error metric.
        """
        if np.isnan(optimal_conf):
            return np.nan

        valid_mask = [True if s >= optimal_conf else False for s in self.confidences]
        if np.sum(valid_mask) == 0:
            return 0.0
        valid_values = np.nanmean(self.values[valid_mask])
        return valid_values

    def compute_cummean_values(self) -> np.array:
        """
        This is taken from nuScenes-devkit.
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/utils.py#L156
        Computes the cumulative mean up to each position in a NaN sensitive way
        - If all values are NaN return an array of ones.
        - If some values are NaN, accumulate arrays discording those entries.
        """
        if sum(np.isnan(self.values)) == len(self.values):
            # Is all numbers in array are NaN's.
            return np.ones(len(self.values))  # If all errors are NaN set to error to 1 for all operating points.
        else:
            # Accumulate in a nan-aware manner.
            sum_vals = np.nancumsum(self.values.astype(float))  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(~np.isnan(self.values))  # Number of non-nans up to each position.
            return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

    def interpolate_values(self, conf_interp: np.array, ascending_sorted: bool = False) -> np.array:
        """
        Interpolate the values using the confidence interpolation to assign them to the same confidence/recall bucket.
        The code is taken from nuScenes-devkit.
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py#L150-L153

        Args:
            conf_interp (np.array): Confidence interpolation.
            ascending_sorted (bool): Whether every input is sorted in ascending order.

        Returns:
            np.array: Interpolated values.
        """
        # For each match_data, we first calculate the accumulated mean.
        cummean_value = self.compute_cummean_values()

        # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
        if ascending_sorted:
            return np.interp(conf_interp, self.confidences, cummean_value)[::-1]
        else:
            return np.interp(conf_interp, self.confidences[::-1], cummean_value[::-1])[::-1]


class TPErrorBEVCenterDistance(TPErrorMetric):
    """TP error BEV center distance metric class

    Attributes:
        mode (str): TP error BEV center distance metric name.
    """

    mode: str = "bev_center_distance_err"
    average_mode: str = "ATE"
    mean_average_mode: str = "mATE"

    def compute_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """Get TP error BEV center distance metric value

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP error BEV center distance metric value
        """
        distance_error_bev = object_result.distance_error_bev
        if distance_error_bev is None:
            return np.nan
        return distance_error_bev

    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        return False


class TPErrorOrientation(TPErrorMetric):
    """TP error orientation metric class

    Attributes:
        mode (str): TP error orientation metric name.
    """

    mode: str = "orientation_err"
    average_mode: str = "AOE"
    mean_average_mode: str = "mAOE"

    @staticmethod
    def angle_diff(x: float, y: float, period: float) -> float:
        """
        Get the smallest angle difference between 2 angles: the angle from y to x.
        The function is taken from nuScenes-devkit.
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/utils.py#L50
        :param x: To angle.
        :param y: From angle.
        :param period: Periodicity in radians for assessing angle difference.
        :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
        """

        # calculate angle difference, modulo to [0, 2*pi]
        diff = (x - y + period / 2) % period - period / 2
        if diff > np.pi:
            diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

        return diff

    def compute_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """Get TP error orientation metric value

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP error orientation metric value
        """
        if object_result.ground_truth_object is None:
            return np.nan

        yaw1, _, _ = object_result.estimated_object.state.orientation.yaw_pitch_roll
        yaw2, _, _ = object_result.ground_truth_object.state.orientation.yaw_pitch_roll

        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py#L106
        if object_result.estimated_object.semantic_label.name in [AutowareLabel.BARRIER.value]:
            period = np.pi
        else:
            period = 2 * np.pi

        return self.angle_diff(yaw1, yaw2, period)

    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        # Ignore traffic cones for orientation error calculation when computing AOE
        return label_type.name in [AutowareLabel.TRAFFIC_CONE.value]


class TPErrorScale(TPErrorMetric):
    """TP error scale metric class

    Attributes:
        mode (str): TP error orientation metric name.
    """

    mode: str = "scale_err"
    average_mode: str = "ASE"
    mean_average_mode: str = "mASE"

    def compute_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """Get TP error scale metric value between estimated and ground truth object.

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP error scale metric value
        """
        scale_iou = object_result.scale_iou
        if scale_iou is None:
            return np.nan
        return 1 - scale_iou

    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        return False


class TPErrorBEVVelocity(TPErrorMetric):
    """TP error BEV velocity metric class

    Attributes:
        mode (str): TP error BEV velocity metric name.
    """

    mode: str = "bev_velocity_err"
    average_mode: str = "AVE"
    mean_average_mode: str = "mAVE"

    def compute_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """Get TP error BEV velocity metric value

        Args:
            object_result (DynamicObjectWithPerceptionResult): Object result.

        Returns:
            float: TP error BEV velocity metric value
        """
        distance_error_bev = object_result.distance_error_bev
        if distance_error_bev is None:
            return np.nan
        return distance_error_bev

    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        # Ignore traffic cones and barriers for velocity error calculation when computing AVE
        return label_type.name in [AutowareLabel.TRAFFIC_CONE.value, AutowareLabel.BARRIER.value]


class TPErrorAttribute(TPErrorMetric):
    """TP error attribute metric class

    Attributes:
        mode (str): TP error attribute metric name.
    """

    mode: str = "attribute_err"
    average_mode: str = "AAE"
    mean_average_mode: str = "mAAE"

    def compute_value(self, object_result: DynamicObjectWithPerceptionResult) -> float:
        """Get TP error attribute metric value"""
        if object_result.ground_truth_object is None:
            return np.nan
        # Always return 1.0 for attribute errors
        return 1.0

    def ignore_target_labels(self, label_type: LabelType) -> bool:
        """Check if the target label is ignored."""
        return label_type.name in [AutowareLabel.BARRIER.value]
