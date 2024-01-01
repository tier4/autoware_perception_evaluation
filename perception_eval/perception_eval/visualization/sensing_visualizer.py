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

import os.path as osp
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from perception_eval.config import SensingEvaluationConfig
from perception_eval.object import DynamicObject
from perception_eval.result import SensingFrameResult
from perception_eval.result import SensingObjectResult
from perception_eval.visualization.color import ColorMap
from PIL import Image
from PIL.Image import Image as PILImage
from pyquaternion import Quaternion
from tqdm import tqdm
import yaml


class SensingVisualizer:
    """The class to visualize sensing results in BEV space.

    Properties:
        config (SensingEvaluationConfig)

    Args:
        config (SensingEvaluationConfig)
        figsize (Tuple[int, int]): Figure size, (width, height) order. Defaults to (800, 600).
    """

    def __init__(
        self,
        config: SensingEvaluationConfig,
        xylim: Tuple[float, float] = (100, 100),
        figsize: Tuple[int, int] = (800, 600),
    ) -> None:
        assert config.evaluation_task.is_3d()
        self.__config: SensingEvaluationConfig = config
        self.__cmap: ColorMap = ColorMap(rgb=True)
        self.__figsize = (figsize[0] / 100.0, figsize[1] / 100.0)

        self.__figure, self.__axes = plt.subplots(figsize=self.__figsize)
        self.__animation_frames: List[PILImage] = []

        self.__xlim, self.__ylim = xylim

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
        **kwargs,
    ) -> SensingVisualizer:
        """Sensing results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.
        Returns:
            SensingVisualizer
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, any]] = yaml.safe_load(scenario_file)

        s_cfg: Dict[str, any] = scenario_obj["Evaluation"]["SensingEvaluationConfig"]
        config_dict: Dict[str, any] = s_cfg["evaluation_config_dict"]

        config_dict["label_prefix"] = "autoware"

        config: SensingEvaluationConfig = SensingEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id="base_link",
            result_root_directory=result_root_directory,
            config_dict=config_dict,
            load_raw_data=False,
        )

        return cls(config, **kwargs)

    @property
    def config(self) -> SensingEvaluationConfig:
        return self.__config

    @property
    def xlim(self) -> float:
        return self.__xlim

    @property
    def ylim(self) -> float:
        return self.__ylim

    def set_axes_limit(self, xlim: float, ylim: float) -> None:
        """Set xy-axes limit.

        Args:
            xlim (float): Limit of x-axis.
            ylim (float): Limit of y-axis.
        """
        self.__xlim = xlim
        self.__ylim = ylim

    def visualize_all(
        self,
        frame_results: List[SensingFrameResult],
        filename: Optional[str] = None,
        cache_figure: bool = False,
    ) -> None:
        """Visualize all frames in BEV space.

        Args:
            frame_results (List[SensingFrameResult]): The list of SensingFrameResult.
            filename  (Optional[str])
            cache_figure (bool): Whether cache figure for each frame. Defaults to False.
        """
        frame_result_: SensingFrameResult
        for frame_result_ in tqdm(frame_results, desc="Visualize results for each frame"):
            self.__axes: Axes = self.visualize_frame(frame_result=frame_result_, axes=self.__axes)
            if cache_figure is False:
                self.__axes.clear()

        # save animation as gif
        self.__save_animation(filename)
        self.clear()

    def clear(self) -> None:
        """Clear properties at the enf of visualize all frame."""
        self.__axes.clear()
        self.__animation_frames.clear()

    def set_figsize(self, figsize: Tuple[int, int]) -> None:
        """Set figure size.
        Args:
            figsize (Tuple[int, int]): Figure size, (width, height) order.
        """
        width, height = figsize[0] / 100.0, figsize[1] / 100.0
        self.__figure.set_figwidth(width)
        self.__figure.set_figheight(height)
        self.__figsize = (width, height)

    def visualize_frame(
        self,
        frame_result: SensingFrameResult,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """Visualize a frame result in BEV space.

        Color:
            TP estimated    : Blue
            TP GT           : Red
            FP              : Cyan
            FN              : Orange

        Args:
            frame_result (PerceptionFrameResult)
            axes (Optional[Axes]): The Axes instance. Defaults to None.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        frame_number: int = frame_result.frame_number
        axes.set_title(f"Frame: {frame_number} ({self.config.frame_ids[0].value})")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")

        # Plot ego vehicle position
        axes = self._plot_ego(axes=axes)

        handles: List[Patch] = []
        # Plot objects
        axes = self.plot_objects(
            objects=frame_result.detection_success_results,
            axes=axes,
            color="green",
        )
        handles.append(Patch(color="green", label="Success Detection"))

        axes = self.plot_objects(
            objects=frame_result.detection_warning_results,
            axes=axes,
            color="yellow",
        )
        handles.append(Patch(color="yellow", label="Warning Detection"))

        axes = self.plot_objects(
            objects=frame_result.detection_fail_results,
            axes=axes,
            color="red",
        )
        handles.append(Patch(color="red", label="Fail Detection"))

        for i, fail_pointcloud in enumerate(frame_result.pointcloud_failed_non_detection):
            color = self.__cmap[i] / 255.0
            self.plot_pointcloud(pointcloud=fail_pointcloud, axes=axes, color=color)
            handles.append(Patch(color=color, label=f"Fail Non-detection@area{i}"))

        plt.legend(
            handles=handles,
            bbox_to_anchor=(1.1, 1.1),
            loc="upper right",
            borderaxespad=0,
            markerscale=10.0,
        )

        filepath: str = osp.join(self.config.visualization_directory, f"{frame_number}.png")
        plt.savefig(filepath, format="png")
        frame = Image.open(filepath)
        self.__animation_frames.append(frame)

        return axes

    def _plot_ego(
        self,
        axes: Optional[Axes] = None,
        size: Tuple[float, float] = (5.0, 2.5),
    ) -> Axes:
        """Plot ego vehicle.

        Args:
            axes (Axes): The Axes instance.
            size (Tuple[float, float]): The size of box, (length, width). Defaults to (5.0, 2.5).

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        ego_color: np.ndarray = self.__cmap.get_simple("black")
        ego_xy: np.ndarray = np.array((0.0, 0.0))
        box_width: float = size[0]
        box_height: float = size[1]

        plt.xlim([-self.xlim + ego_xy[0], self.xlim + ego_xy[0]])
        plt.ylim([-self.ylim + ego_xy[1], self.ylim + ego_xy[1]])

        box_bottom_left: np.ndarray = ego_xy - (np.array(size) / 2.0)
        yaw: float = 0.0

        transform: Affine2D = Affine2D().rotate_around(ego_xy[0], ego_xy[1], yaw) + axes.transData
        box: Rectangle = Rectangle(
            xy=box_bottom_left,
            width=box_width,
            height=box_height,
            edgecolor=ego_color,
            fill=False,
            transform=transform,
        )
        axes.add_patch(box)

        return axes

    def plot_objects(
        self,
        objects: Union[List[DynamicObject], List[SensingObjectResult]],
        axes: Optional[Axes] = None,
        color: Union[str, np.ndarray] = "red",
    ) -> Axes:
        """Plot objects in BEV space.

        ```
                     +------------------+
        y            |                  |
        ^          height               |
        |            |                  |
        o--> x      (xy)---- width -----+
        ```

        Args:
            objects (Union[List[DynamicObject], SensingObjectResult]): The list of object being visualized.
            axes (Optional[Axes]): The Axes instance. If not specified, new Axes is created. Defaults to None.
            color (Union[str, np.ndarray]): The name of color, red/green/blue/yellow/cyan/black. Defaults to None.
                If not be specified, red is used.

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        edge_color = self.__cmap.get_simple(color) if isinstance(color, str) else color

        for object_ in objects:
            if isinstance(object_, SensingObjectResult):
                pointcloud: np.ndarray = object_.inside_pointcloud
                nearest_point: Optional[np.ndarray] = object_.nearest_point
                object_: DynamicObject = object_.ground_truth_object
            box_center: np.ndarray = np.array(object_.state.position)[:2]
            orientation: Quaternion = object_.state.orientation
            box_size: np.ndarray = np.array(object_.state.size)[:2]

            box_bottom_left: np.ndarray = box_center - (box_size[::-1] / 2.0)

            # rotate box around center
            yaw: float = orientation.yaw_pitch_roll[0]
            transform: Affine2D = Affine2D().rotate_around(box_center[0], box_center[1], yaw) + axes.transData

            box: Rectangle = Rectangle(
                xy=box_bottom_left,
                width=box_size[1],
                height=box_size[0],
                edgecolor=edge_color,
                fill=False,
                transform=transform,
            )
            axes.add_patch(box)

            if len(pointcloud) > 0:
                axes.scatter(pointcloud[:, 0], pointcloud[:, 1], c=[edge_color], s=0.5)
            if nearest_point is not None:
                color_ = self.__cmap.get_simple("blue")
                axes.scatter(
                    nearest_point[0],
                    nearest_point[1],
                    c=[color_],
                    s=0.5,
                    label="Nearest point",
                )

        return axes

    def plot_pointcloud(
        self,
        pointcloud: np.ndarray,
        axes: Optional[Axes] = None,
        color: Union[str, np.ndarray] = "red",
    ) -> Axes:
        """Plot pointcloud.

        Args:
            axes (Optional[Axes]): Axes instance. If not specified new Axes is created. Defaults to None.
            color (Union[str, np.ndarray]): Name of color. If not be specified, red is used. Defaults to None.

        Returns:
            axes (Axes): Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        if isinstance(color, str):
            color = self.__cmap.get_simple(color)

        axes.scatter(pointcloud[:, 0], pointcloud[:, 1], c=[color], s=0.5)

        return axes

    def __save_animation(self, filename: Optional[str] = None) -> None:
        """Save animation as mp4.

        Args:
            filename (Optional[str]): Video filename. If None, save as scene_result_3d.mp4. Defaults to None.
        """
        if filename is None:
            filename = "scene_result_3d.mp4"

        if not filename.endswith(".mp4"):
            filename += ".mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filepath = osp.join(self.config.visualization_directory, filename)
        video = cv2.VideoWriter(filepath, fourcc, fps=10, frameSize=self.__animation_frames[0].size)
        for frame in self.__animation_frames:
            frame_ = np.array(frame.copy())
            video.write(cv2.cvtColor(frame_, cv2.COLOR_RGB2BGR))
        video.release()
