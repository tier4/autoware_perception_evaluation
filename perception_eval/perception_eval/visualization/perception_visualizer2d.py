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
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.visualization.color import ColorMap
from PIL import Image
from PIL.Image import Image as PILImage
from tqdm import tqdm
import yaml


class PerceptionVisualizer2D:
    """The class to visualize perception results in BEV space.

    Attributes:
        config (PerceptionEvaluationConfig): Evaluation config.

    Args:
        config (PerceptionEvaluationConfig): Evaluation config.
        figsize (Tuple[int, int]): Figure size, (width, height) order. Defaults to (800, 600).
    """

    def __init__(
        self,
        config: PerceptionEvaluationConfig,
        figsize: Tuple[int, int] = (800, 600),
    ) -> None:
        assert config.evaluation_task.is_2d()
        self.__config: PerceptionEvaluationConfig = config
        self.__cmap: ColorMap = ColorMap(rgb=True)
        self.__figsize = (figsize[0] / 100.0, figsize[1] / 100.0)
        self.__animation_frames: List[PILImage] = []

        self.__figure, self.__axes = self.init_figure()

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
            PerceptionVisualizer2D: Visualizer instance.
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, any] = p_cfg["evaluation_config_dict"]

        eval_cfg_dict["label_prefix"] = (
            "traffic_light" if eval_cfg_dict["UseCaseName"] == "traffic_light" else "autoware"
        )
        camera_types: Dict[str, int] = scenario_obj["Evaluation"]["Conditions"]["TargetCameras"]

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id=list(camera_types.keys()),
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
            load_raw_data=False,
        )

        return cls(evaluation_config, **kwargs)

    def init_figure(self) -> Tuple[Figure, np.ndarray]:
        """Initialize figure and axes.

        Returns:
            Figure: Figure instance.
            numpy.ndarray: NDArray of multiple Axes instances.
        """

        self.label_type: Union[AutowareLabel, TrafficLightLabel] = self.__config.label_converter.label_type

        if self.label_type == TrafficLightLabel:
            cameras = ("cam_traffic_light_near", "cam_traffic_light_far")
            fig, axes = plt.subplots(1, 2, figsize=self.__figsize, gridspec_kw=dict(wspace=0))
        elif self.label_type == AutowareLabel:
            cameras = (
                "cam_front_left",
                "cam_front",
                "cam_front_right",
                "cam_back_left",
                "cam_back",
                "cam_back_right",
            )
            fig, axes = plt.subplots(2, 3, figsize=self.__figsize, gridspec_kw=dict(wspace=0))
        else:
            raise TypeError(f"Unexpected label type: {self.label_type}")

        self.cameras = [FrameID.from_value(name) for name in cameras]

        return fig, axes

    @property
    def config(self) -> PerceptionEvaluationConfig:
        return self.__config

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
        if self.config.evaluation_task == EvaluationTask.TRACKING2D:
            self.__tracked_paths = {}

        for frame_result_ in tqdm(frame_results, desc="Visualize results for each frame"):
            self.__axes: np.ndarray = self.visualize_frame(
                frame_result=frame_result_,
                axes=self.__axes,
            )
            if cache_figure is False:
                self.__clear_axes()

        self.__save_animation(filename)
        self.clear()

    def clear(self) -> None:
        """Clear properties at the enf of visualize all frame."""
        self.__clear_axes()
        self.__animation_frames.clear()
        if self.config.evaluation_task == EvaluationTask.TRACKING2D:
            self.__tracked_paths.clear()

    def __clear_axes(self) -> None:
        """Clear each Axes instance."""
        if self.label_type == TrafficLightLabel:
            for i in range(len(self.__axes)):
                self.__axes[i].clear()
        else:
            num_rows, num_cols = self.__axes.shape
            for i in range(num_rows):
                for j in range(num_cols):
                    self.__axes[i, j].clear()

    def set_figsize(self, figsize: Tuple[int, int]) -> None:
        """Set figure size.
        Args:
            figsize (Tuple[int, int]): Figure size, (width, height).
        """
        width, height = figsize[0] / 100.0, figsize[1] / 100.0
        self.__figure.set_figwidth(width)
        self.__figure.set_figheight(height)
        self.__figsize = (width, height)

    def __get_axes_idx(self, key: FrameID) -> Tuple[int, int]:
        i: int = self.cameras.index(key)
        row, col = (0, i) if self.label_type == TrafficLightLabel else (i // 3, i - (3 * (i // 3)))
        return row, col

    def visualize_frame(
        self,
        frame_result: PerceptionFrameResult,
        axes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Visualize a frame result on image.

        Color:
            TP estimated    : Blue
            TP GT           : Red
            FP              : Cyan
            TN              : Purple
            FN              : Orange

        Args:
            frame_result (PerceptionFrameResult)
            axes (Optional[numpy.ndarray]): Axes instances. Defaults to None.

        Returns:
            numpy.ndarray: Numpy array of Axes instances.
        """
        if axes is None:
            axes = self.__axes

        raw_data = frame_result.frame_ground_truth.raw_data
        if raw_data is None:
            raise RuntimeError("`raw_data`: must be loaded.")
        else:
            for i, camera in enumerate(self.cameras):
                img: Optional[np.ndarray] = raw_data.get(camera)
                if self.label_type == TrafficLightLabel:
                    if img is not None:
                        axes[i].imshow(img)
                    axes[i].set_axis_off()
                    axes[i].set_title(f"{camera}")
                else:
                    row, col = (i // 3, i - (3 * (i // 3)))
                    if img is not None:
                        axes[row, col].imshow(img)
                    axes[row, col].set_axis_off()
                    axes[row, col].set_title(f"{camera}")

        frame_number: str = frame_result.frame_ground_truth.frame_name
        self.__figure.suptitle(f"Frame: {frame_number}")

        # Plot objects
        handles: List[Patch] = []
        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_object_results,
            is_ground_truth=False,
            axes=axes,
            color="blue",
        )
        handles.append(Patch(color="blue", label="TP est"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tp_object_results,
            is_ground_truth=True,
            axes=axes,
            color="red",
        )
        handles.append(Patch(color="red", label="TP GT"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.fp_object_results,
            is_ground_truth=False,
            axes=axes,
            color="cyan",
        )
        handles.append(Patch(color="cyan", label="FP"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.tn_objects,
            is_ground_truth=True,
            axes=axes,
            color="purple",
        )
        handles.append(Patch(color="purple", label="TN"))

        axes = self.plot_objects(
            objects=frame_result.pass_fail_result.fn_objects,
            is_ground_truth=True,
            axes=axes,
            color="orange",
        )
        handles.append(Patch(color="orange", label="FN"))

        self.__figure.legend(
            handles=handles,
            bbox_to_anchor=(1.0, 1.1),
            loc="lower right",
            ncol=4,
            borderaxespad=0,
            markerscale=10.0,
        )

        self.__figure.tight_layout()
        filepath: str = osp.join(self.config.visualization_directory, f"{frame_number}.png")
        plt.savefig(filepath, format="png")
        frame = Image.open(filepath)
        self.__animation_frames.append(frame)

        return axes

    def plot_objects(
        self,
        objects: List[Union[DynamicObject2D, DynamicObjectWithPerceptionResult]],
        is_ground_truth: bool,
        axes: Optional[np.ndarray] = None,
        color: Optional[str] = None,
    ) -> np.ndarray:
        """Plot objects on image.

        ```
                     +------------------+
        y            |                  |
        ^          height               |
        |            |                  |
        o--> x      (xy)---- width -----+
        ```

        Args:
            objects (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): The list of object being visualized.
            is_ground_truth (bool): Whether ground truth object is.
            axes (Optional[Axes]): Axes instances. If not specified, new Axes is created. Defaults to None.
            color (Optional[str]): Name of color, red/green/blue/yellow/cyan/black. Defaults to None.
                If not be specified, red is used.

        Returns:
            axes (Axes): The Axes instance.
        """
        if axes is None:
            _, axes = self.init_figure()

        color: str = "red" if color is None else color
        edge_color = self.__cmap.get_simple(color)
        object_text = "GT" if is_ground_truth else "Est"
        for object_ in objects:
            if isinstance(object_, DynamicObjectWithPerceptionResult):
                object_: DynamicObject2D = object_.ground_truth_object if is_ground_truth else object_.estimated_object
            if object_ is None or object_.roi is None:
                continue
            box_top_left: np.ndarray = np.array(object_.roi.offset)
            box_size: np.ndarray = np.array(object_.roi.size)
            box_bottom_left: np.ndarray = box_top_left
            if self.config.evaluation_task == EvaluationTask.TRACKING2D:
                edge_color = self.__cmap.get(object_.uuid)

            box_text = f"{object_text}: {str(object_.semantic_label.label)}"
            box: Rectangle = Rectangle(
                xy=box_bottom_left,
                width=box_size[0],
                height=box_size[1],
                edgecolor=edge_color,
                fill=False,
                label=box_text,
            )
            row, col = self.__get_axes_idx(object_.frame_id)
            if self.label_type == TrafficLightLabel:
                axes[col].add_patch(box)
                axes[col].text(*box_bottom_left, s=box_text, fontsize="x-small", color=edge_color)
            else:
                axes[row, col].add_patch(box)
                axes[row, col].text(*box_bottom_left, s=box_text, fontsize="x-small", color=edge_color)

        return axes

    def __save_animation(self, filename: Optional[str] = None) -> None:
        """Save animation as mp4.

        Args:
            filename (Optional[str]): Video filename. If None, save as scene_result_2d.mp4. Defaults to None.
        """
        if filename is None:
            filename = "scene_result_2d.mp4"

        if not filename.endswith(".mp4"):
            filename += ".mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filepath = osp.join(self.config.visualization_directory, filename)
        video = cv2.VideoWriter(filepath, fourcc, fps=10, frameSize=self.__animation_frames[0].size)
        for frame in self.__animation_frames:
            frame_ = np.array(frame.copy())
            video.write(cv2.cvtColor(frame_, cv2.COLOR_RGB2BGR))
        video.release()
