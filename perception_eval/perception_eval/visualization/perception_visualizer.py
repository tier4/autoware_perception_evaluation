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

from logging import getLogger
import os.path as osp
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.object import DynamicObject
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_tp_fp_objects
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.matching.objects_filter import get_fn_objects
from perception_eval.util.math import rotation_matrix_to_euler
from perception_eval.visualization.color import ColorMap
from pyquaternion import Quaternion
from tqdm import tqdm

from .perception_visualization_config import PerceptionVisualizationConfig

logger = getLogger(__name__)


class PerceptionVisualizer:
    """The class to visualize perception results in BEV space.

    Properties:
        self.config (PerceptionVisualizationConfig)
    """

    def __init__(self, config: PerceptionVisualizationConfig) -> None:
        """[summary]
        Args:
            config (PerceptionVisualizationConfig)
        """
        self.__config: PerceptionVisualizationConfig = config
        self.__cmap: ColorMap = ColorMap(rgb=True)
        self.__figsize: Tuple[float, float] = (
            self.config.width / 100.0,
            self.config.height / 100.0,
        )

        if self.config.evaluation_task == EvaluationTask.TRACKING:
            # Each tracked path is specified by uuid.gt/est_track.label
            self.__tracked_paths: Dict[str, List[Tuple[float, float]]] = {}

        self.__figure, self.__axes = plt.subplots(figsize=self.__figsize)
        self.__animation_frames: List[List[plt.Artist]] = []

    @classmethod
    def from_eval_cfg(
        cls,
        eval_cfg: PerceptionEvaluationConfig,
        height: int = 480,
        width: int = 640,
    ) -> PerceptionVisualizer:
        """[summary]

        Args:
            eval_cfg (PerceptionEvaluationConfig): Evaluation config for perception.
            height (int): The image height. Defaults to 640.
            width (int): The image width. Defaults to 640.
        """
        config: PerceptionVisualizationConfig = PerceptionVisualizationConfig(
            visualization_directory_path=eval_cfg.visualization_directory,
            frame_id=eval_cfg.frame_id,
            evaluation_task=eval_cfg.evaluation_task,
            height=height,
            width=width,
            **eval_cfg.filtering_params,
        )
        return cls(config)

    @classmethod
    def from_args(
        cls,
        visualization_directory_path: str,
        frame_id: str,
        evaluation_task: EvaluationTask,
        height: int = 480,
        width: int = 640,
        **kwargs,
    ) -> PerceptionVisualizer:
        config: PerceptionVisualizationConfig = PerceptionVisualizationConfig(
            visualization_directory_path=visualization_directory_path,
            frame_id=frame_id,
            evaluation_task=evaluation_task,
            height=height,
            width=width,
            **kwargs,
        )
        return cls(config)

    @property
    def config(self) -> PerceptionVisualizationConfig:
        return self.__config

    def visualize_all(
        self,
        frame_results: List[PerceptionFrameResult],
        animation: bool = False,
        cache_figure: bool = False,
        matching_mode: Optional[MatchingMode] = None,
        matching_threshold_list: Optional[List[float]] = None,
    ) -> None:
        """[summary]
        Visualize all frames in BEV space.

        Args:
            frame_results (List[PerceptionFrameResult]): The list of PerceptionFrameResult.
            save_html (bool): Wether save image as html. Defaults to False.
            animation (bool): Whether create animation as gif. Defaults to False.
            cache_figure (bool): Whether cache figure for each frame. Defaults to False.
            matching_mode (Optional[MatchingMode]): The MatchingMode instance. Defaults to None.
            matching_threshold_list (Optional[List[float]]): The list of matching threshold. Defaults to None.
        """
        if self.config.evaluation_task == EvaluationTask.TRACKING:
            self.__tracked_paths = {}

        frame_result_: PerceptionFrameResult
        for frame_result_ in tqdm(frame_results, desc="Visualize results for each frame"):
            self.__axes: Axes = self.visualize_frame(
                frame_result=frame_result_,
                matching_mode=matching_mode,
                matching_threshold_list=matching_threshold_list,
                axes=self.__axes,
            )
            if cache_figure is False:
                self.__axes.clear()

        # save animation as gif
        if animation:
            # self._save_animation(file_name)
            logger.warning("animation is under construction")
        self.clear()

    def clear(self) -> None:
        """[summary]
        Clear properties at the enf of visualize all frame.
        """
        self.__axes.clear()
        self.__animation_frames.clear()
        if self.config.evaluation_task == EvaluationTask.TRACKING:
            self.__tracked_paths.clear()

    def set_figsize(self, height: int, width: int) -> None:
        """[summary]
        Set figure size.
        Args:
            height (int): The height of figure.
            width (int): The width of figure.
        """
        self.__config.height = height
        self.__config.width = width
        self.__figure.set_figheight(height / 100.0)
        self.__figure.set_figwidth(height / 100.0)
        self.__figsize = (height / 100.0, width / 100.0)

    def visualize_frame(
        self,
        frame_result: PerceptionFrameResult,
        file_name: Optional[str] = None,
        axes: Optional[Axes] = None,
        matching_mode: Optional[MatchingMode] = None,
        matching_threshold_list: Optional[List[float]] = None,
    ) -> Axes:
        """[summary]
        Visualize a frame result in BEV space.

        Color:
            TP estimated    : Blue
            TP GT           : Red
            FP              : Cyan
            FN              : Orange

        Args:
            frame_result (PerceptionFrameResult)
            file_name (Optional[str]): The name of file. If not specified, saved by frame name. Defaults to None.
            axes (Optional[Axes]): The Axes instance. Defaults to None.
            matching_mode (Optional[MatchingMode]): The matching mode instance.
            matching_threshold_list (Optional[List[float]]): The list of matching threshold.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        frame_number: str = frame_result.frame_ground_truth.frame_name
        axes.set_title(f"Frame: {frame_number} ({self.config.frame_id})")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")

        frame_artists: List[Axes.ArtistList] = []

        # Plot ego vehicle position
        axes, artists = self._plot_ego(
            ego2map=frame_result.frame_ground_truth.ego2map,
            axes=axes,
        )
        frame_artists += artists

        # set object
        tp_objects, fp_objects, fn_objects = self._divide_objects(
            frame_result.object_results,
            frame_result.frame_ground_truth,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
        )

        # Plot objects
        axes, artists = self.plot_objects(
            objects=tp_objects,
            is_ground_truth=False,
            axes=axes,
            label="TP est",
            color="blue",
        )
        frame_artists += artists

        axes, artists = self.plot_objects(
            objects=tp_objects,
            is_ground_truth=True,
            axes=axes,
            label="TP GT",
            color="red",
        )
        frame_artists += artists

        axes, artists = self.plot_objects(
            objects=fp_objects,
            is_ground_truth=False,
            axes=axes,
            label="FP",
            color="cyan",
        )
        frame_artists += artists

        axes, artists = self.plot_objects(
            objects=fn_objects,
            is_ground_truth=True,
            axes=axes,
            label="FN",
            color="orange",
        )
        frame_artists += artists

        legend = plt.legend(
            bbox_to_anchor=(1.1, 1.1),
            loc="upper right",
            borderaxespad=0,
            markerscale=10.0,
        )
        frame_artists.append(legend)
        self.__animation_frames.append(frame_artists)

        # save_figure:
        if file_name is None:
            file_name = frame_result.frame_ground_truth.frame_name
        filepath: str = osp.join(self.config.visualization_directory_path, file_name)

        plt.savefig(filepath + ".png")

        return axes

    def _plot_ego(
        self,
        ego2map: np.ndarray,
        axes: Optional[Axes] = None,
        size: Tuple[float, float] = (5.0, 2.5),
    ) -> Tuple[Axes, List[plt.Artist]]:
        """[summary]
        Plot ego vehicle.

        Args:
            ego2map (np.ndarray): The 4x4 array of transform matrix from ego coords system to map coords system.
            axes (Axes): The Axes instance.
            size (Tuple[float, float]): The size of box, (length, width). Defaults to (5.0, 2.5).

        Returns:
            axes (Axes): The Axes instance.
            artists (List[plt.Artist]): The list of Artist instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        artists: List[plt.Artist] = []

        ego_color: np.ndarray = self.__cmap.get_simple("black")
        if self.config.frame_id == "base_link":
            ego_xy: Tuple[float, float] = np.array((0.0, 0.0))
        elif self.config.frame_id == "map":
            ego_xy: np.ndarray = ego2map[:2, 3]
        box_width: float = size[0]
        box_height: float = size[1]
        scatter_ = axes.scatter(
            ego_xy[0],
            ego_xy[1],
            color=ego_color,
            label="Ego vehicle",
            s=0.5 * self.config.width / 640,
        )
        artists.append(scatter_)

        plt.xlim([-self.config.xlim + ego_xy[0], self.config.xlim + ego_xy[0]])
        plt.ylim([-self.config.ylim + ego_xy[1], self.config.ylim + ego_xy[1]])

        box_bottom_left: np.ndarray = ego_xy - (np.array(size) / 2.0)

        if self.config.frame_id == "map":
            yaw: float = rotation_matrix_to_euler(ego2map[:3, :3])[2]
        elif self.config.frame_id == "base_link":
            yaw: float = 0.0
        else:
            raise ValueError(f"Unexpected frame_id: {self.config.frame_id}")

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

        return axes, artists

    def plot_objects(
        self,
        objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]],
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Tuple[Axes, List[plt.Artist]]:
        """[summary]
        Plot objects in BEV space.

        :                +------------------+
        :   y            |                  |
        :   ^          height               |
        :   |            |                  |
        :   o--> x      (xy)---- width -----+

        Args:
            objects (Union[List[DynamicObject], DynamicObjectWithPerceptionResult]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Optional[Axes]): The Axes instance. If not specified, new Axes is created. Defaults to None.
            label (str): The label of object type, e.g. TP/FP/FP. Defaults to None.
            color (Optional[str]): The name of color, red/green/blue/yellow/cyan/black. Defaults to None.
                If not be specified, red is used.

        Returns:
            axes (Axes): The Axes instance.
            artists (List[plt.Artist]): The list of Artist instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        artists: List[plt.Artist] = []

        box_center_x: List[float] = []
        box_center_y: List[float] = []
        color: str = "red" if color is None else color
        edge_color = self.__cmap.get_simple(color)
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
            artists.append(box)

            if self.config.evaluation_task == EvaluationTask.TRACKING:
                box_velocity: np.ndarray = np.array(object_.state.velocity)[:2]
                # plot heading
                if self.config.frame_id == "base_link":
                    dx, dy = box_velocity
                elif self.config.frame_id == "map":
                    dx, dy = box_velocity
                else:
                    raise ValueError(f"Unexpected frame_id: {self.config.frame_id}")

                arrow_ = axes.arrow(
                    x=box_center[0],
                    y=box_center[1],
                    dx=dx,
                    dy=dy,
                    color=edge_color,
                    shape="full",
                    length_includes_head=True,
                )
                artists.append(arrow_)

            # tracked path
            if self.config.evaluation_task == EvaluationTask.TRACKING:
                axes, tracking_artists = self._plot_tracked_path(
                    object_, is_ground_truth, axes=axes
                )
                artists += tracking_artists

            # predicted path
            if self.config.evaluation_task == EvaluationTask.PREDICTION:
                pass

            box_center_x.append(box_center[0])
            box_center_y.append(box_center[1])
        scatter_ = axes.scatter(box_center_x, box_center_y, color=edge_color, label=label, s=0.5)
        artists.append(scatter_)

        return axes, artists

    def _plot_tracked_path(
        self,
        dynamic_object: DynamicObject,
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
    ) -> Tuple[Axes, List[plt.Artist]]:
        """[summary]
        Plot tracked paths for one object.

        Args:
            dynamic_objects (List[DynamicObject]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Axes): The Axes instance. If not specified, new Axes is created. Defaults to None.

        Returns:
            axes (Axes): The Axes instance.
            artists (List[plt.Artist]): The list of Artist instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        artists: List[plt.Artist] = []

        object_type_: str = ".gt_track" if is_ground_truth else ".est_track"
        object_label_: str = "." + dynamic_object.semantic_label.value
        uuid_: str = dynamic_object.uuid + object_type_ + object_label_
        if uuid_ not in self.__tracked_paths.keys():
            self.__tracked_paths.update({uuid_: [dynamic_object.state.position[:2]]})
        else:
            self.__tracked_paths[uuid_].append(dynamic_object.state.position[:2])
        color_: np.ndarray = self.__cmap.get(uuid_)
        paths_arr_: np.ndarray = np.array(self.__tracked_paths[uuid_])
        plot_ = axes.plot(paths_arr_[:, 0], paths_arr_[:, 1], "o--", color=color_, markersize=1)

        artists.append(plot_)

        return axes, artists

    def _plot_predicted_path(
        self,
        dynamic_objects: List[DynamicObject],
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """[summary]
        Plot predicted paths for one object.

        Args:
            dynamic_objects (List[DynamicObject]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Axes): The Axes instance.

        Returns:
            axes (Axes): The Axes instance.
        """
        pass

    def _save_animation(self, file_name: Optional[str] = None):
        """[summary]
        Save animation as gif.

        Args:
            file_name (str)
        """
        if file_name is None:
            file_name = "all"
        filepath: str = osp.join(self.config.visualization_directory_path, file_name)
        ani: ArtistAnimation = ArtistAnimation(
            self.__figure,
            self.__animation_frames,
            interval=100,
        )
        ani.save(filepath + "_animation.gif", writer="pillow")

    def _divide_objects(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        frame_ground_truth: FrameGroundTruth,
        matching_mode: Optional[MatchingMode] = None,
        matching_threshold_list: Optional[List[float]] = None,
    ) -> Tuple[
        List[DynamicObjectWithPerceptionResult],
        List[DynamicObjectWithPerceptionResult],
        List[DynamicObject],
    ]:
        """[summary]
        Divide TP/FP object results and FN objects.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult])
            frame_ground_truth (FrameGroundTruth)
            matching_mode (Optional[MatchingMode])
            matching_threshold_list (Optional[List[float]])

        Returns:
            tp_objects (List[DynamicObjectWithPerceptionResult])
            fp_objects (List[DynamicObjectWithPerceptionResult])
            fn_objects (List[DynamicObject])
        """
        # filter object results
        filtered_estimated_objects: List[DynamicObjectWithPerceptionResult] = filter_object_results(
            frame_id=self.config.frame_id,
            object_results=object_results,
            target_labels=self.config.target_labels,
            max_x_position_list=self.config.max_x_position_list,
            max_y_position_list=self.config.max_y_position_list,
            max_distance_list=self.config.max_distance_list,
            min_distance_list=self.config.min_distance_list,
            min_point_numbers=self.config.min_point_numbers,
            target_uuids=self.config.target_uuids,
            ego2map=frame_ground_truth.ego2map,
        )
        filtered_ground_truth: List[DynamicObject] = filter_objects(
            frame_id=self.config.frame_id,
            objects=frame_ground_truth.objects,
            is_gt=True,
            target_labels=self.config.target_labels,
            max_x_position_list=self.config.max_x_position_list,
            max_y_position_list=self.config.max_y_position_list,
            max_distance_list=self.config.max_distance_list,
            min_distance_list=self.config.min_distance_list,
            min_point_numbers=self.config.min_point_numbers,
            target_uuids=self.config.target_uuids,
            ego2map=frame_ground_truth.ego2map,
        )
        # divide TP/FP objects
        tp_objects, fp_objects = divide_tp_fp_objects(
            object_results=filtered_estimated_objects,
            target_labels=self.config.target_labels,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
        )
        fn_objects = get_fn_objects(
            ground_truth_objects=filtered_ground_truth,
            object_results=object_results,
            tp_objects=tp_objects,
        )
        return tp_objects, fp_objects, fn_objects
