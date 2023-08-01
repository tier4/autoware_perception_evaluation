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
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.util.math import rotation_matrix_to_euler
from perception_eval.visualization.color import ColorMap
from PIL import Image
from PIL.Image import Image as PILImage
from pyquaternion import Quaternion
from tqdm import tqdm
import yaml


class PerceptionVisualizer3D:
    """The class to visualize perception results in BEV space.

    Properties:
        config (PerceptionVisualizationConfig)

    Args:
        config (PerceptionVisualizationConfig)
        figsize (Tuple[int, int]): Figure size, (width, height) order. Defaults to (800, 600).
    """

    def __init__(
        self,
        config: PerceptionEvaluationConfig,
        figsize: Tuple[int, int] = (800, 600),
    ) -> None:
        assert config.evaluation_task.is_3d()
        self.__config: PerceptionEvaluationConfig = config
        self.__cmap: ColorMap = ColorMap(rgb=True)
        self.__figsize = (figsize[0] / 100.0, figsize[1] / 100.0)

        self.__figure, self.__axes = plt.subplots(figsize=self.__figsize)
        self.__animation_frames: List[PILImage] = []

        max_x_position_list = config.filtering_params.get("max_x_position_list")
        max_y_position_list = config.filtering_params.get("max_y_position_list")
        max_distance_list = config.filtering_params.get("max_distance_list")
        if max_x_position_list is not None and max_y_position_list is not None:
            self.__xlim: float = max(max_x_position_list)
            self.__ylim: float = max(max_y_position_list)
        elif max_distance_list is not None:
            self.__xlim: float = max(max_distance_list)
            self.__ylim: float = max(max_distance_list)
        else:
            self.__xlim: float = 100.0
            self.__ylim: float = 100.0

        if self.config.evaluation_task == EvaluationTask.TRACKING2D:
            # Each tracked path is specified by uuid.gt/est_track.label
            self.__tracked_paths: Dict[str, List[Tuple[float, float]]] = {}

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
        **kwargs,
    ) -> PerceptionVisualizer3D:
        """Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.
        Returns:
            PerceptionVisualizer3D
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, any] = p_cfg["evaluation_config_dict"]

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id="base_link" if eval_cfg_dict["evaluation_task"] == "detection" else "map",
            merge_similar_labels=p_cfg.get("merge_similar_labels", False),
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
            load_raw_data=False,
        )

        return cls(evaluation_config, **kwargs)

    @property
    def config(self) -> PerceptionEvaluationConfig:
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
        frame_results: List[PerceptionFrameResult],
        filename: Optional[str] = None,
        cache_figure: bool = False,
    ) -> None:
        """Visualize all frames in BEV space.

        Args:
            frame_results (List[PerceptionFrameResult]): The list of PerceptionFrameResult.
            save_html (bool): Wether save image as html. Defaults to False.
            animation (bool): Whether create animation as gif. Defaults to False.
            cache_figure (bool): Whether cache figure for each frame. Defaults to False.
        """
        if self.config.evaluation_task == EvaluationTask.TRACKING:
            self.__tracked_paths = {}

        frame_result_: PerceptionFrameResult
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
        if self.config.evaluation_task == EvaluationTask.TRACKING:
            self.__tracked_paths.clear()

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
        frame_result: PerceptionFrameResult,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """Visualize a frame result in BEV space.

        Color:
            TP estimated    : Blue
            TP GT           : Red
            FP              : Cyan
            TN              : Purple
            FN              : Orange

        Args:
            frame_result (PerceptionFrameResult)
            axes (Optional[Axes]): The Axes instance. Defaults to None.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        frame_number: str = frame_result.frame_ground_truth.frame_name
        axes.set_title(f"Frame: {frame_number} ({self.config.frame_ids[0].value})")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")

        # Plot ego vehicle position
        axes = self._plot_ego(
            ego2map=frame_result.frame_ground_truth.ego2map,
            axes=axes,
        )

        pointcloud: Optional[np.ndarray] = (
            frame_result.frame_ground_truth.raw_data["lidar"] if self.config.load_raw_data else None
        )

        # Plot objects
        handles: List[Patch] = []
        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_object_results,
            is_ground_truth=False,
            axes=axes,
            label="TP est",
            color="blue",
            pointcloud=pointcloud,
        )
        handles.append(Patch(color="blue", label="TP est"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_object_results,
            is_ground_truth=True,
            axes=axes,
            label="TP GT",
            color="red",
            pointcloud=pointcloud,
        )
        handles.append(Patch(color="red", label="TP GT"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.fp_object_results,
            is_ground_truth=False,
            axes=axes,
            label="FP",
            color="cyan",
            pointcloud=pointcloud,
        )
        handles.append(Patch(color="cyan", label="FP"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tn_objects,
            is_ground_truth=True,
            axes=axes,
            label="TN",
            color="purple",
            pointcloud=pointcloud,
        )
        handles.append(Patch(color="purple", label="TN"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.fn_objects,
            is_ground_truth=True,
            axes=axes,
            label="FN",
            color="orange",
            pointcloud=pointcloud,
        )
        handles.append(Patch(color="orange", label="FN"))

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
        ego2map: np.ndarray,
        axes: Optional[Axes] = None,
        size: Tuple[float, float] = (5.0, 2.5),
    ) -> Axes:
        """Plot ego vehicle.

        Args:
            ego2map (np.ndarray): The 4x4 array of transform matrix from ego coords system to map coords system.
            axes (Axes): The Axes instance.
            size (Tuple[float, float]): The size of box, (length, width). Defaults to (5.0, 2.5).

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        ego_color: np.ndarray = self.__cmap.get_simple("black")
        ego_xy: np.ndarray = (
            np.array((0.0, 0.0))
            if self.config.frame_ids[0] == FrameID.BASE_LINK
            else ego2map[:2, 3]
        )
        box_width: float = size[0]
        box_height: float = size[1]

        plt.xlim([-self.xlim + ego_xy[0], self.xlim + ego_xy[0]])
        plt.ylim([-self.ylim + ego_xy[1], self.ylim + ego_xy[1]])

        box_bottom_left: np.ndarray = ego_xy - (np.array(size) / 2.0)
        yaw: float = (
            0.0
            if self.config.frame_ids[0] == FrameID.BASE_LINK
            else rotation_matrix_to_euler(ego2map[:3, :3])[2]
        )

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
        objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]],
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        pointcloud: Optional[np.ndarray] = None,
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
            objects (Union[List[DynamicObject], DynamicObjectWithPerceptionResult]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Optional[Axes]): The Axes instance. If not specified, new Axes is created. Defaults to None.
            label (str): The label of object type, e.g. TP/FP/FP. Defaults to None.
            color (Optional[str]): The name of color, red/green/blue/yellow/cyan/black. Defaults to None.
                If not be specified, red is used.

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        color: str = "red" if color is None else color
        edge_color = self.__cmap.get_simple(color)

        cropped_pointcloud = []
        for object_ in objects:
            if isinstance(object_, DynamicObjectWithPerceptionResult):
                if is_ground_truth:
                    object_: DynamicObject = object_.ground_truth_object
                else:
                    object_: DynamicObject = object_.estimated_object
            if object_ is None:
                continue
            box_center: np.ndarray = np.array(object_.state.position)[:2]
            orientation: Quaternion = object_.state.orientation
            box_size: np.ndarray = np.array(object_.state.size)[:2]

            box_bottom_left: np.ndarray = box_center - (box_size[::-1] / 2.0)

            # rotate box around center
            yaw: float = orientation.yaw_pitch_roll[0]
            transform: Affine2D = (
                Affine2D().rotate_around(box_center[0], box_center[1], yaw) + axes.transData
            )

            box: Rectangle = Rectangle(
                xy=box_bottom_left,
                width=box_size[1],
                height=box_size[0],
                edgecolor=edge_color,
                fill=False,
                transform=transform,
            )
            axes.add_patch(box)

            if self.config.evaluation_task == EvaluationTask.TRACKING:
                box_velocity: np.ndarray = np.array(object_.state.velocity)[:2]
                # plot heading
                dx, dy = box_velocity

                axes.arrow(
                    x=box_center[0],
                    y=box_center[1],
                    dx=dx,
                    dy=dy,
                    color=edge_color,
                    shape="full",
                    length_includes_head=True,
                )

            # tracked path
            if self.config.evaluation_task == EvaluationTask.TRACKING:
                axes = self._plot_tracked_path(object_, is_ground_truth, axes=axes)

            # predicted path
            if self.config.evaluation_task == EvaluationTask.PREDICTION:
                pass

            if pointcloud is not None:
                cropped_pointcloud += object_.crop_pointcloud(
                    pointcloud=pointcloud.copy(),
                    inside=True,
                ).tolist()

        if pointcloud is not None and len(cropped_pointcloud) > 0:
            cropped_pointcloud = np.array(cropped_pointcloud)
            axes.scatter(
                x=cropped_pointcloud[:, 0],
                y=cropped_pointcloud[:, 1],
                marker=".",
                c=[edge_color],
                label=label,
                s=0.5,
            )

        return axes

    def _plot_tracked_path(
        self,
        dynamic_object: DynamicObject,
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """Plot tracked paths for one object.

        Args:
            dynamic_objects (List[DynamicObject]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Axes): The Axes instance. If not specified, new Axes is created. Defaults to None.

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        object_type_: str = ".gt_track" if is_ground_truth else ".est_track"
        object_label_: str = "." + dynamic_object.semantic_label.label.value
        uuid_: str = dynamic_object.uuid + object_type_ + object_label_
        if uuid_ not in self.__tracked_paths.keys():
            self.__tracked_paths.update({uuid_: [dynamic_object.state.position[:2]]})
        else:
            self.__tracked_paths[uuid_].append(dynamic_object.state.position[:2])
        color_: np.ndarray = self.__cmap.get(uuid_)
        paths_arr_: np.ndarray = np.array(self.__tracked_paths[uuid_])
        axes.plot(paths_arr_[:, 0], paths_arr_[:, 1], "o--", color=color_, markersize=1)

        return axes

    def _plot_predicted_path(
        self,
        dynamic_objects: List[DynamicObject],
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """Plot predicted paths for one object.

        Args:
            dynamic_objects (List[DynamicObject]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Axes): The Axes instance.

        Returns:
            axes (Axes): The Axes instance.
        """
        pass

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
