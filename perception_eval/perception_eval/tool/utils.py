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
from perception_eval.common.status import FrameID
from perception_eval.evaluation.metrics.metrics import MetricsScore
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult


class MatchingStatus(Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[MatchingStatus, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class PlotAxes(Enum):
    """[summary]

    2D plot:
        FRAME: X-axis is the number of frame.
        TIME: X-axis is time[s].
        DISTANCE: X-axis is 2D euclidean distance[m].
        X: X-axis is x[m].
        Y: X-axis is y[m].
        VX: X-axis is vx[m/s].
        VY: X-axis is vy[m/s].
    3D plot:
        POSITION: X-axis is x[m], y-axis is y[m].
        VELOCITY: X-axis is vx[m/s], y-axis is vy[m/s].
        POLAR: X-axis is theta[rad], y-axis is r[m]
    """

    FRAME = "frame"
    TIME = "time"
    DISTANCE = "distance"
    X = "x"
    Y = "y"
    VX = "vx"
    VY = "vy"
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
        """[summary]
        Returns axes values for plot.

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
        elif self == PlotAxes.POSITION:
            axes: np.ndarray = np.array(df[["x", "y"]]).reshape(2, -1)
        elif self == PlotAxes.SIZE:
            axes: np.ndarray = np.array(df[["w", "l"]]).reshape(2, -1)
        elif self == PlotAxes.VELOCITY:
            axes: np.ndarray = np.array(df[["vx", "vy"]]).reshape(2, -1)
        elif self == PlotAxes.POLAR:
            distances: np.ndarray = np.linalg.norm(df[["x", "y"]], axis=1)
            thetas: np.ndarray = np.arctan2(df["x"], df["y"])
            thetas[thetas > np.pi] = thetas[thetas > np.pi] - 2.0 * np.pi
            axes: np.ndarray = np.stack([thetas, distances], axis=0)
        else:
            raise TypeError(f"Unexpected mode: {self}")

        return axes

    def get_label(self) -> Union[str, Tuple[str]]:
        """[summary]
        Returns label name for plot.

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
        elif self == PlotAxes.POSITION:
            return "x [m]", "y [m]"
        elif self == PlotAxes.SIZE:
            return "w [m]", "l [m]"
        elif self == PlotAxes.VELOCITY:
            return "vx [m/s]", "vy [m/s]"
        elif self == PlotAxes.POLAR:
            return "theta [rad]", "r [m]"

    def get_bin(self) -> Union[float, Tuple[float, float]]:
        """[summary]
        Returns default bins.

        Returns:
            Union[float, Tuple[float, float]]
        """
        if self == PlotAxes.FRAME:
            return 1.0
        elif self == PlotAxes.TIME:
            return 5.0
        elif self in (PlotAxes.DISTANCE, PlotAxes.X, PlotAxes.Y):
            return 0.5
        elif self in (PlotAxes.VX, PlotAxes.VY):
            return 1.0
        elif self == PlotAxes.POSITION:
            return (0.5, 0.5)
        elif self == PlotAxes.SIZE:
            return (1.0, 1.0)
        elif self == PlotAxes.VELOCITY:
            return (1.0, 1.0)
        elif self == PlotAxes.POLAR:
            return (0.2, 0.5)

    @property
    def projection(self) -> Optional[str]:
        """[summary]
        Returns type of projection.

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
    """[summary]
    Generate (x,y) pairs of upper right and bottom left of each separate area.
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
        raise ValueError(
            f"The number of area division must be 1, 3 or 9, but got {num_area_division}"
        )

    return upper_rights, bottom_lefts


def get_area_idx(
    object_result: Union[DynamicObject, DynamicObjectWithPerceptionResult],
    upper_rights: np.ndarray,
    bottom_lefts: np.ndarray,
    ego2map: Optional[np.ndarray] = None,
) -> Optional[int]:
    """[summary]
    Returns the index of area.

    Args:
        object_result (Union[DynamicObject, DynamicObjectWithPerceptionResult])
        upper_rights (np.ndarray): in shape (N, 2), N is number of area division.
        bottom_lefts (np.ndarray): in shape (N, 2), N is number of area division.
        ego2map (Optional[np.ndarray]): in shape (4, 4)

    Returns:
        area_idx (Optional[int]): If the position is out of range, returns None.
    """
    if isinstance(object_result, DynamicObject):
        frame_id: FrameID = object_result.frame_id
        obj_xyz: np.ndarray = np.array(object_result.state.position)
    elif isinstance(object_result, DynamicObjectWithPerceptionResult):
        frame_id: FrameID = object_result.estimated_object.frame_id
        obj_xyz: np.ndarray = np.array(object_result.estimated_object.state.position)
    else:
        raise TypeError(f"Unexpected object type: {type(object_result)}")

    if frame_id == FrameID.MAP:
        if ego2map is None:
            raise RuntimeError("When frame id is map, ego2map must be specified.")
        obj_xyz: np.ndarray = np.append(obj_xyz, 1.0)
        obj_xy = np.linalg.inv(ego2map).dot(obj_xyz)[:2]
    elif frame_id == "base_link":
        obj_xy = obj_xyz[:2]
    else:
        raise ValueError(f"Unexpected frame_id: {frame_id}")

    is_x_inside: np.ndarray = (obj_xy[0] < upper_rights[:, 0]) * (obj_xy[0] > bottom_lefts[:, 0])
    is_y_inside: np.ndarray = (obj_xy[1] > upper_rights[:, 1]) * (obj_xy[1] < bottom_lefts[:, 1])
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
        frame_id: str = frame_result.frame_ground_truth.frame_id
        ego2map: Optional[np.ndarray] = frame_result.frame_ground_truth.ego2map
        for object_result in frame_result.object_results:
            object_result_area: int = get_area_idx(
                frame_id,
                object_result,
                upper_rights,
                bottom_lefts,
                ego2map,
            )
            if object_result_area in area:
                out_object_results.append(object_result)
        for ground_truth in frame_result.frame_ground_truth.objects:
            ground_truth_area: int = get_area_idx(
                frame_id,
                ground_truth,
                upper_rights,
                bottom_lefts,
                ego2map,
            )
            if ground_truth_area in area:
                out_ground_truths.append(ground_truth)

        frame_result.object_results = out_object_results
        frame_result.frame_ground_truth.objects = out_ground_truths

    return out_frame_results


def setup_axis(ax: plt.Axes, **kwargs) -> None:
    """[summary]
    Setup axis limits and grid interval to plt.Axes.

    Args:
        ax (plt.Axes)
        **kwargs:
            xlim (Union[float, Sequence]): If use sequence, (left, right) order.
            ylim (Union[float, Sequence]): If use sequence, (left, right) order. Defaults to None.
            grid_interval (float): Interval of grid. Defaults to None.
    """
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


def filter_df(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """[summary]
    Filter DataFrame with args and kwargs.

    Args:
        df (pandas.DataFrame): Source DataFrame.
        *args
        **kwargs

    Returns:
        df_ (pandas.DataFrame): Filtered DataFrame.
    """
    df_ = df

    for key, item in kwargs.items():
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            df_ = df_[df_[key].isin(item)]
        else:
            df_ = df_[df_[key] == item]

    if args:
        df_ = df_[list(args)]

    return df_
