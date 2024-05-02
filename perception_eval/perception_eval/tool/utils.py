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

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import TransformDict
from perception_eval.common.transform import TransformKey
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.metrics import MetricsScore
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult


class PlotAxes(Enum):
    """Enum class for plot axes.

    2D plot:
        FRAME: X-axis is the number of frame.
        TIME: X-axis is time[s].
        DISTANCE: X-axis is 2D euclidean distance[m].
        X: X-axis is x[m].
        Y: X-axis is y[m].
        VX: X-axis is vx[m/s].
        VY: X-axis is vy[m/s].
        CONFIDENCE: X-axis is confidence in [0, 100][%].
    3D plot:
        POSITION: X-axis is x[m], y-axis is y[m].
        VELOCITY: X-axis is vx[m/s], y-axis is vy[m/s].
        SIZE: X-axis is width[m], y-axis is height[m](2D) or length[m](3D).
        POLAR: X-axis is theta[deg], y-axis is r[m]
    """

    FRAME = "frame"
    TIME = "time"
    DISTANCE = "distance"
    X = "x"
    Y = "y"
    VX = "vx"
    VY = "vy"
    CONFIDENCE = "confidence"
    POSITION = "position"
    SIZE = "size"
    VELOCITY = "velocity"
    POLAR = "polar"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[PlotAxes, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def is_3d(self) -> bool:
        return self in (PlotAxes.POSITION, PlotAxes.SIZE, PlotAxes.VELOCITY, PlotAxes.POLAR)

    def is_2d(self) -> bool:
        return not self.is_3d()

    def get_axes(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Returns axes values for plot.

        Args:
            df (pandas.DataFrame): Source DataFrame.

        Returns:
            numpy.ndarray: Array of axes values. For 2D plot, returns in shape (N,). For 3D plot, returns in shape (2, N)
        """
        if self == PlotAxes.FRAME:
            axes = np.array(df["frame"], dtype=np.uint32)
        elif self == PlotAxes.TIME:
            if kwargs.get("align2origin", True):
                axes = get_aligned_timestamp(df)
            else:
                axes = np.array(df["timestamp"], dtype=np.uint64) / 1e6
        elif self == PlotAxes.DISTANCE:
            axes: np.ndarray = np.linalg.norm(df[["x", "y"]], axis=1)
        elif self == PlotAxes.X:
            axes: np.ndarray = np.array(df["x"])
        elif self == PlotAxes.Y:
            axes: np.ndarray = np.array(df["y"])
        elif self == PlotAxes.VX:
            axes: np.ndarray = np.array(df["vx"])
        elif self == PlotAxes.VY:
            axes: np.ndarray = np.array(df["vy"])
        elif self == PlotAxes.CONFIDENCE:
            axes: np.ndarray = np.array(df["confidence"]) * 100
        elif self == PlotAxes.POSITION:
            axes: np.ndarray = np.array(df[["x", "y"]]).reshape(2, -1)
        elif self == PlotAxes.SIZE:
            axes: np.ndarray = (
                np.array(df[["width", "length"]]).reshape(2, -1)
                if "length" in df.columns
                else np.array(df[["width", "height"]]).reshape(2, -1)
            )
        elif self == PlotAxes.VELOCITY:
            axes: np.ndarray = np.array(df[["vx", "vy"]]).reshape(2, -1)
        elif self == PlotAxes.POLAR:
            distances: np.ndarray = np.linalg.norm(df[["x", "y"]], axis=1)
            thetas: np.ndarray = np.arctan2(df["x"], df["y"])
            thetas[thetas > np.pi] = thetas[thetas > np.pi] - 2.0 * np.pi
            thetas = np.rad2deg(thetas)
            axes: np.ndarray = np.stack([thetas, distances], axis=0)
        else:
            raise TypeError(f"Unexpected mode: {self}")

        return axes

    def get_label(self) -> Union[str, Tuple[str]]:
        """Returns label name for plot.

        Returns:
            str: Name of label.
        """
        if self == PlotAxes.FRAME:
            return str(self)
        elif self == PlotAxes.TIME:
            return str(self) + " [s]"
        elif self in (PlotAxes.DISTANCE, PlotAxes.X, PlotAxes.Y):
            return str(self) + " [m]"
        elif self in (PlotAxes.VX, PlotAxes.VY):
            return str(self) + " [m/s]"
        elif self == PlotAxes.CONFIDENCE:
            return str(self) + " [%]"
        elif self == PlotAxes.POSITION:
            return "x [m]", "y [m]"
        elif self == PlotAxes.SIZE:
            return "width [m]", "length [m]"
        elif self == PlotAxes.VELOCITY:
            return "vx [m/s]", "vy [m/s]"
        elif self == PlotAxes.POLAR:
            return "theta [deg]", "r [m]"

    def get_bins(self) -> Union[int, Tuple[int, int]]:
        """Returns default bins.

        Returns:
            Union[int, Tuple[int, int]]
        """
        if self == PlotAxes.FRAME:
            return 1
        elif self == PlotAxes.TIME:
            return 1
        elif self in (PlotAxes.DISTANCE, PlotAxes.X, PlotAxes.Y):
            return 10
        elif self in (PlotAxes.VX, PlotAxes.VY):
            return 1
        elif self == PlotAxes.CONFIDENCE:
            return 1
        elif self == PlotAxes.POSITION:
            return (10, 10)
        elif self == PlotAxes.SIZE:
            return (5, 5)
        elif self == PlotAxes.VELOCITY:
            return (1, 1)
        elif self == PlotAxes.POLAR:
            return (5, 10)

    def setup_axis(self, ax: plt.Axes, **kwargs) -> None:
        """Setup axis limits and grid interval to plt.Axes.

        Args:
            ax (plt.Axes)
            **kwargs:
                xlim (Union[float, Sequence]): If use sequence, (left, right) order.
                ylim (Union[float, Sequence]): If use sequence, (left, right) order. Defaults to None.
                grid_interval (float): Interval of grid. Defaults to None.
        """
        setup_axis(ax, **kwargs)
        if self == PlotAxes.CONFIDENCE:
            ax.set_xlim(-5, 105)

    @property
    def projection(self) -> Optional[str]:
        """Returns type of projection.

        Returns:
            Optional[str]: If 3D, returns "3d", otherwise returns None.
        """
        return "3d" if self.is_3d() else None

    @property
    def xlabel(self) -> str:
        return self.get_label() if self.is_2d() else self.get_label()[0]

    @property
    def ylabel(self) -> str:
        return self.get_label() if self.is_2d() else self.get_label()[1]


def generate_area_points(
    num_area_division: int,
    max_x: float,
    max_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (x,y) pairs of upper right and bottom left of each separate area.
    They are arranged in numerical order as shown in below.

    num_area_division:
        1:                            3:                  9:
                    max_x_position
                    +--------+          +--------+          +--------+
                    |    0   |          |____0___|          |_0|_1|_2|
    max_y_position  |    +   |          |____1___|          |_3|_4|_5|
                    |   ego  |          |    2   |          | 6| 7| 8|
                    +--------+          +--------+          +--------+

    Args:
        num_area_division (int)
        max_x (float)
        max_y (float)

    Returns:
        upper_rights (np.ndarray)
        bottom_lefts (np.ndarray)
    """
    if num_area_division == 1:
        upper_rights: np.ndarray = np.array((max_x, -max_y)).reshape(1, -1)
        bottom_lefts: np.ndarray = np.array((-max_x, max_y)).reshape(1, -1)
    elif num_area_division == 3:
        right_x: np.ndarray = np.arange(max_x, -max_x, -2 * max_x / 3)
        left_x: np.ndarray = np.arange(-max_x, max_x, 2 * max_x / 3)[::-1]
        right_y: np.ndarray = np.repeat(-max_y, 3)
        left_y: np.ndarray = np.repeat(max_y, 3)
        upper_rights: np.ndarray = np.stack([right_x, right_y], axis=1)
        bottom_lefts: np.ndarray = np.stack([left_x, left_y], axis=1)
    elif num_area_division == 9:
        right_x: np.ndarray = np.arange(max_x, -max_x, -2 * max_x / 3)
        left_x: np.ndarray = np.arange(-max_x, max_x, 2 * max_x / 3)[::-1]
        right_y: np.ndarray = np.arange(-max_y, max_y, 2 * max_y / 3)[::-1]
        left_y: np.ndarray = np.arange(max_y, -max_y, -2 * max_y / 3)
        r_xx, r_yy = np.meshgrid(right_x, right_y)
        l_xx, l_yy = np.meshgrid(left_x, left_y)
        upper_rights: np.ndarray = np.stack([r_xx, r_yy], axis=-1).reshape(-1, 2)
        bottom_lefts: np.ndarray = np.stack([l_xx, l_yy], axis=-1).reshape(-1, 2)
    else:
        raise ValueError(f"The number of area division must be 1, 3 or 9, but got {num_area_division}")

    return upper_rights, bottom_lefts


def get_area_idx(
    object_result: Union[DynamicObject, DynamicObjectWithPerceptionResult],
    upper_rights: np.ndarray,
    bottom_lefts: np.ndarray,
    transforms: TransformDict,
) -> Optional[int]:
    """Returns the index of area.

    Args:
        object_result (Union[DynamicObject, DynamicObjectWithPerceptionResult])
        upper_rights (np.ndarray): in shape (N, 2), N is number of area division.
        bottom_lefts (np.ndarray): in shape (N, 2), N is number of area division.

    Returns:
        area_idx (Optional[int]): If the position is out of range, returns None.
    """
    if isinstance(object_result, DynamicObject):
        frame_id: FrameID = object_result.frame_id
        position: np.ndarray = np.array(object_result.state.position)
    elif isinstance(object_result, DynamicObjectWithPerceptionResult):
        frame_id: FrameID = object_result.estimated_object.frame_id
        position: np.ndarray = np.array(object_result.estimated_object.state.position)
    else:
        raise TypeError(f"Unexpected object type: {type(object_result)}")

    transform_key = TransformKey(frame_id, FrameID.BASE_LINK)
    x, y, _ = transforms.transform(transform_key, position)

    is_x_inside: np.ndarray = (x < upper_rights[:, 0]) * (x > bottom_lefts[:, 0])
    is_y_inside: np.ndarray = (y > upper_rights[:, 1]) * (y < bottom_lefts[:, 1])
    if any(is_x_inside * is_y_inside) is False:
        return None
    area_idx: int = np.where(is_x_inside * is_y_inside)[0].item()
    return area_idx


def extract_area_results(
    frame_results: List[PerceptionFrameResult],
    area: Union[int, List[int]],
    upper_rights: np.ndarray,
    bottom_lefts: np.ndarray,
) -> List[PerceptionFrameResult]:
    """[summary]
    Extract object results and ground truth of PerceptionFrameResult in area.
    Args:
        frame_results (List[PerceptionFrameResult])
        area (Union[int, List[int]])
        upper_rights (np.ndarray)
        bottom_lefts (np.ndarray)
    Returns:
        List[PerceptionFrameResult]
    """
    out_frame_results: List[PerceptionFrameResult] = deepcopy(frame_results)
    if isinstance(area, int):
        area = [area]

    for frame_result in out_frame_results:
        out_object_results: List[DynamicObjectWithPerceptionResult] = []
        out_ground_truths: List[DynamicObject] = []
        transforms = frame_result.frame_ground_truth.transforms
        for object_result in frame_result.object_results:
            object_result_area: int = get_area_idx(
                object_result,
                upper_rights,
                bottom_lefts,
                transforms,
            )
            if object_result_area in area:
                out_object_results.append(object_result)
        for ground_truth in frame_result.frame_ground_truth.objects:
            ground_truth_area: int = get_area_idx(
                ground_truth,
                upper_rights,
                bottom_lefts,
                transforms,
            )
            if ground_truth_area in area:
                out_ground_truths.append(ground_truth)

        frame_result.object_results = out_object_results
        frame_result.frame_ground_truth.objects = out_ground_truths

    return out_frame_results


def setup_axis(ax: Union[plt.Axes, np.ndarray], **kwargs) -> None:
    """[summary]
    Setup axis limits and grid interval to plt.Axes.

    Args:
        ax (plt.Axes)
        **kwargs:
            xlim (Union[float, Sequence]): If use sequence, (left, right) order.
            ylim (Union[float, Sequence]): If use sequence, (left, right) order. Defaults to None.
            grid_interval (float): Interval of grid. Defaults to None.
    """
    if isinstance(ax, np.ndarray):
        for ax_ in ax:
            setup_axis(ax_, **kwargs)
    else:
        ax.grid()
        if kwargs.get("xlim"):
            xlim: Union[float, Sequence] = kwargs.pop("xlim")
            if isinstance(xlim, float):
                ax.set_xlim(-xlim, xlim)
            elif isinstance(xlim, (list, tuple)):
                ax.set_xlim(xlim[0], xlim[1])
        if kwargs.get("ylim"):
            ylim: Union[float, Sequence] = kwargs.pop("ylim")
            if isinstance(ylim, float):
                ax.set_ylim(-ylim, ylim)
            elif isinstance(ylim, (list, tuple)):
                ax.set_ylim(ylim[0], ylim[1])

        if kwargs.get("grid_interval"):
            ax.grid(lw=kwargs.pop("grid_interval"))


def get_metrics_info(metrics_score: MetricsScore) -> Dict[str, Any]:
    """[summary]
    Returns metrics score information as dict.

    Args:
        metrics_score (MetricsScore): Calculated metrics score.

    Returns:
        data (Dict[str, Any]):
    """
    data: Dict[str, List[float]] = {}
    # detection
    for map in metrics_score.maps:
        mode: str = str(map.matching_mode)
        ap_mode: str = f"AP({mode})"
        aph_mode: str = f"APH({mode})"
        data[ap_mode] = [map.map]
        data[aph_mode] = [map.maph]
        for ap, aph in zip(map.aps, map.aphs):
            data[ap_mode].append(ap.ap)
            data[aph_mode].append(aph.ap)

    # tracking
    for tracking_score in metrics_score.tracking_scores:
        mode: str = str(tracking_score.matching_mode)
        mota_mode: str = f"MOTA({mode})"
        motp_mode: str = f"MOTP({mode})"
        id_switch_mode: str = f"IDswitch({mode})"
        mota, motp, id_switch = tracking_score._sum_clear()
        data[mota_mode] = [mota]
        data[motp_mode] = [motp]
        data[id_switch_mode] = [id_switch]
        for clear in tracking_score.clears:
            data[mota_mode].append(clear.results["MOTA"])
            data[motp_mode].append(clear.results["MOTP"])
            data[id_switch_mode].append(clear.results["id_switch"])

    # prediction
    # TODO

    # classification
    for classification_score in metrics_score.classification_scores:
        accuracy, precision, recall, f1score = classification_score._summarize()
        data["Accuracy"] = [accuracy]
        data["Precision"] = [precision]
        data["Recall"] = [recall]
        data["F1score"] = [f1score]
        for cls_acc in classification_score.accuracies:
            data["Accuracy"].append(cls_acc.accuracy)
            data["Precision"].append(cls_acc.precision)
            data["Recall"].append(cls_acc.recall)
            data["F1score"].append(cls_acc.f1score)

    return data


def get_aligned_timestamp(df: pd.DataFrame) -> np.ndarray:
    """[summary]
    Returns timestamp aligned to minimum timestamp for each scene.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        numpy.ndarray: in shape (N,)
    """
    scenes: np.ndarray = pd.unique(df["scene"])
    scenes = scenes[~np.isnan(scenes)]
    scene_axes: List[np.ndarray] = []
    for scene in scenes:
        df_ = df[df["scene"] == scene]
        axes_ = np.array(df_["timestamp"], dtype=np.uint64) / 1e6
        scene_axes += (axes_ - axes_.min()).tolist()
    return np.array(scene_axes)


def filter_frame_by_distance(
    frame: PerceptionFrameResult,
    min_distance: Optional[float],
    max_distance: Optional[float],
) -> PerceptionFrameResult:
    ret_frame = deepcopy(frame)

    min_distance_list = [min_distance] * len(ret_frame.target_labels)
    max_distance_list = [max_distance] * len(ret_frame.target_labels)

    ret_frame.object_results = filter_object_results(
        ret_frame.object_results,
        target_labels=ret_frame.target_labels,
        max_distance_list=max_distance_list,
        min_distance_list=min_distance_list,
        transforms=ret_frame.frame_ground_truth.transforms,
    )
    ret_frame.frame_ground_truth.objects = filter_objects(
        ret_frame.frame_ground_truth.objects,
        is_gt=True,
        target_labels=ret_frame.target_labels,
        max_distance_list=max_distance_list,
        min_distance_list=min_distance_list,
        transforms=ret_frame.frame_ground_truth.transforms,
    )
    ret_frame.evaluate_frame(ret_frame.frame_ground_truth.objects)

    return ret_frame
