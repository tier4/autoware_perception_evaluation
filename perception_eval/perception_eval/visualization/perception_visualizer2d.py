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

import logging
import os.path as osp
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.visualization.color import ColorMap
from tqdm import tqdm
import yaml


class PerceptionVisualizer2D:
    """The class to visualize perception results in BEV space.

    Attributes:
        config (PerceptionEvaluationConfig)

    Args:
        config (PerceptionEvaluationConfig)
    """

    def __init__(self, config: PerceptionEvaluationConfig, **kwargs) -> None:
        assert config.evaluation_task.is_2d()
        self.__config: PerceptionEvaluationConfig = config
        self.__cmap: ColorMap = ColorMap(rgb=True)
        self.__figsize: Tuple[float, float] = (
            kwargs.get("width", 800) / 100.0,
            kwargs.get("height", 600) / 100.0,
        )

        self.__figure, self.__axes = plt.subplots(figsize=self.__figsize)
        self.__animation_frames: List[List[plt.Artist]] = []

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
        **kwargs,
    ) -> PerceptionVisualizer2D:
        """Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.
        Returns:
            PerceptionPerformanceAnalyzer
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, any] = p_cfg["evaluation_config_dict"]
        eval_task_: str = eval_cfg_dict["evaluation_task"]
        if eval_task_ in ("detection2d", "tracking2d", "classification2d"):
            frame_id = "base_link"
        else:
            raise ValueError(f"Unexpected evaluation task: {eval_task_}")

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id=frame_id,
            merge_similar_labels=p_cfg.get("merge_similar_labels", False),
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
            load_raw_data=False,
        )

        return cls(evaluation_config, **kwargs)

    @property
    def config(self) -> PerceptionEvaluationConfig:
        return self.__config

    def visualize_all(
        self,
        frame_results: List[PerceptionFrameResult],
        animation: bool = False,
        cache_figure: bool = False,
    ) -> None:
        """Visualize all frames in BEV space.

        Args:
            frame_results (List[PerceptionFrameResult]): The list of PerceptionFrameResult.
            save_html (bool): Wether save image as html. Defaults to False.
            animation (bool): Whether create animation as gif. Defaults to False.
            cache_figure (bool): Whether cache figure for each frame. Defaults to False.
        """
        if self.config.evaluation_task == EvaluationTask.TRACKING2D:
            self.__tracked_paths = {}

        frame_result_: PerceptionFrameResult
        for frame_result_ in tqdm(frame_results, desc="Visualize results for each frame"):
            self.__axes: Axes = self.visualize_frame(
                frame_result=frame_result_,
                axes=self.__axes,
            )
            if cache_figure is False:
                self.__axes.clear()

        # save animation as gif
        if animation:
            # self._save_animation(file_name)
            logging.warning("animation is under construction")
        self.clear()

    def clear(self) -> None:
        """Clear properties at the enf of visualize all frame."""
        self.__axes.clear()
        self.__animation_frames.clear()
        if self.config.evaluation_task == EvaluationTask.TRACKING2D:
            self.__tracked_paths.clear()

    def set_figsize(self, height: int, width: int) -> None:
        """Set figure size.
        Args:
            height (int): The height of figure.
            width (int): The width of figure.
        """
        self.__figure.set_figheight(height / 100.0)
        self.__figure.set_figwidth(width / 100.0)
        self.__figsize = (height / 100.0, width / 100.0)

    def visualize_frame(
        self,
        frame_result: PerceptionFrameResult,
        file_name: Optional[str] = None,
        axes: Optional[Axes] = None,
    ) -> Axes:
        """Visualize a frame result on image.

        Color:
            TP estimated    : Blue
            TP GT           : Red
            FP              : Cyan
            FN              : Orange

        Args:
            frame_result (PerceptionFrameResult)
            file_name (Optional[str]): The name of file. If not specified, saved by frame name. Defaults to None.
            axes (Optional[Axes]): The Axes instance. Defaults to None.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        axes.set_axis_off()

        if frame_result.frame_ground_truth.raw_data is None:
            raise RuntimeError("`raw_data`: must be loaded.")
        else:
            axes.imshow(frame_result.frame_ground_truth.raw_data)

        frame_number: str = frame_result.frame_ground_truth.frame_name
        axes.set_title(f"Frame: {frame_number} ({frame_result.frame_id})")

        frame_artists: List[Axes.ArtistList] = []

        # Plot objects
        handles: List[Patch] = []
        axes, artists = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_objects,
            is_ground_truth=False,
            axes=axes,
            color="blue",
        )
        frame_artists += artists
        handles.append(Patch(color="blue", label="TP est"))

        axes, artists = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_objects,
            is_ground_truth=True,
            axes=axes,
            color="red",
        )
        frame_artists += artists
        handles.append(Patch(color="red", label="TP GT"))

        axes, artists = self.plot_objects(
            objects=frame_result.pass_fail_result.fp_objects_result,
            is_ground_truth=False,
            axes=axes,
            color="cyan",
        )
        frame_artists += artists
        handles.append(Patch(color="cyan", label="FP"))

        axes, artists = self.plot_objects(
            objects=frame_result.pass_fail_result.fn_objects,
            is_ground_truth=True,
            axes=axes,
            color="orange",
        )
        frame_artists += artists
        handles.append(Patch(color="orange", label="FN"))

        legend = axes.legend(
            handles=handles,
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
        filepath: str = osp.join(self.config.visualization_directory, file_name)

        plt.savefig(filepath + ".png")

        return axes

    def plot_objects(
        self,
        objects: List[Union[DynamicObject2D, DynamicObjectWithPerceptionResult]],
        is_ground_truth: bool,
        axes: Optional[Axes] = None,
        color: Optional[str] = None,
    ) -> Tuple[Axes, List[plt.Artist]]:
        """Plot objects on image.

        :                +------------------+
        :   y            |                  |
        :   ^          height               |
        :   |            |                  |
        :   o--> x      (xy)---- width -----+

        Args:
            objects (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Optional[Axes]): The Axes instance. If not specified, new Axes is created. Defaults to None.
            color (Optional[str]): The name of color, red/green/blue/yellow/cyan/black. Defaults to None.
                If not be specified, red is used.

        Returns:
            axes (Axes): The Axes instance.
            artists (List[plt.Artist]): The list of Artist instance.
        """
        if axes is None:
            axes: Axes = plt.subplot()

        artists: List[plt.Artist] = []
        color: str = "red" if color is None else color
        edge_color = self.__cmap.get_simple(color)
        object_text = "GT" if is_ground_truth else "Est"
        for object_ in objects:
            if isinstance(object_, DynamicObjectWithPerceptionResult):
                object_: DynamicObject2D = (
                    object_.ground_truth_object if is_ground_truth else object_.estimated_object
                )
            if object_ is None or object_.roi is None:
                continue
            box_top_left: np.ndarray = np.array(object_.roi.offset)
            box_size: np.ndarray = np.array(object_.roi.size)
            box_bottom_left: np.ndarray = box_top_left
            if self.config.evaluation_task == EvaluationTask.TRACKING2D:
                edge_color = self.__cmap.get(object_.uuid)

            box_text = f"{object_text}: {str(object_.semantic_label)}"
            box: Rectangle = Rectangle(
                xy=box_bottom_left,
                width=box_size[0],
                height=box_size[1],
                edgecolor=edge_color,
                fill=False,
                label=box_text,
            )
            axes.add_patch(box)
            artists.append(box)
            axes.text(*box_bottom_left, s=box_text, fontsize="x-small", color=edge_color)

        return axes, artists

    def _save_animation(self, file_name: Optional[str] = None):
        """Save animation as gif.

        Args:
            file_name (str)
        """
        if file_name is None:
            file_name = "all"
        filepath: str = osp.join(self.config.visualization_directory, file_name)
        ani: ArtistAnimation = ArtistAnimation(
            self.__figure,
            self.__animation_frames,
            interval=100,
        )
        ani.save(filepath + "_animation.gif", writer="pillow")
