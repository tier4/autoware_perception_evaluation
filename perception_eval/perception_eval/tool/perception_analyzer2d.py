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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
import yaml

from .perception_analyzer_base import PerceptionAnalyzerBase
from .utils import MatchingStatus
from .utils import PlotAxes


class PerceptionAnalyzer2D(PerceptionAnalyzerBase):
    """An class to analyze perception results.

    config (PerceptionEvaluationConfig): Configurations for evaluation parameters.
        target_labels (List[str]): Target labels list. (e.g. `["car", "pedestrian", "motorbike"]`).
        all_labels (List[str]): Target labels list including "ALL". (e.g. `["ALL", "car", "pedestrian", "motorbike"]`).
        columns (List[str]): Columns in `df`. `["timestamp", "x_offset", "y_offset", "width", "height",\
            "label", "confidence", "uuid", "status", "area", "frame", "scene"]`.
        state_columns (List[str]): State names in `df`. `["x_offset", "y_offset", "width", "height"]`.
        df (pandas.DataFrame): DataFrame.
        plot_directory (str): Directory path to save plot figures.
        frame_results (Dict[str, List[PerceptionFrameResult]]): Dict that items are list of PerceptionFrameResult hashed by scene number.
        num_frame (int): Number of added frames.
        num_scene (int): Number of added scenes.
        num_ground_truth (int): Number of ground truths.
        num_estimation (int): Number of estimations.
        num_tp (int): Number of TP results.
        num_fp (int): Number of FP results.
        num_fn (int): Number of FN results.

    Args:
        evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        super().__init__(evaluation_config=evaluation_config)
        if not self.config.evaluation_task.is_2d():
            raise RuntimeError("Evaluation task must be 2D.")
        self.mode = PlotAxes.CONFIDENCE

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
    ) -> PerceptionAnalyzer2D:
        """Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.

        Returns:
            PerceptionAnalyzer2D: PerceptionAnalyzer2D instance.

        Raises:
            ValueError: When unexpected evaluation task is specified in scenario file.
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, Any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, Any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, Any] = p_cfg["evaluation_config_dict"]
        eval_task_: str = eval_cfg_dict["evaluation_task"]
        if eval_task_ == ("detection2d", "tracking2d", "classification2d"):
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

        return cls(evaluation_config)

    @property
    def columns(self) -> List[str]:
        return [
            "timestamp",
            "x_offset",
            "y_offset",
            "width",
            "height",
            "label",
            "confidence",
            "uuid",
            "status",
            "area",
            "frame",
            "scene",
        ]

    @property
    def state_columns(self) -> List[str]:
        return ["x_offset", "y_offset", "width", "height"]

    def format2dict(
        self,
        object_result: Union[DynamicObject2D, DynamicObjectWithPerceptionResult],
        status: MatchingStatus,
        frame_num: int,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """Format objects to dict.

        Args:
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.

        Returns:
            Dict[str, Dict[str, Any]]
        """
        if isinstance(object_result, DynamicObjectWithPerceptionResult):
            gt: Optional[DynamicObject2D] = object_result.ground_truth_object
            estimation: DynamicObject2D = object_result.estimated_object
        elif isinstance(object_result, DynamicObject2D):
            if status == MatchingStatus.FN:
                gt: DynamicObject2D = object_result
                estimation = None
            elif status == MatchingStatus.FP:
                estimation: DynamicObject2D = object_result
                gt = None
            else:
                raise ValueError("For DynamicObject status must be in FP or FN, but got {status}")
        elif object_result is None:
            gt, estimation = None, None
        else:
            raise TypeError(f"Unexpected object type: {type(object_result)}")

        if gt:
            if gt.roi is not None:
                gt_x_offset, gt_y_offset = gt.roi.offset
                gt_width, gt_height = gt.roi.size
            else:
                gt_x_offset, gt_y_offset = None, None
                gt_width, gt_height = None, None

            gt_ret = dict(
                timestamp=gt.unix_time,
                x_offset=gt_x_offset,
                y_offset=gt_y_offset,
                width=gt_width,
                height=gt_height,
                label=str(gt.semantic_label),
                confidence=gt.semantic_score,
                uuid=gt.uuid,
                status=status,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            gt_ret = {k: None for k in self.keys()}

        if estimation:
            if estimation.roi is not None:
                est_x_offset, est_y_offset = estimation.roi.offset
                est_width, est_height = estimation.roi.size
            else:
                est_x_offset, est_y_offset = None, None
                est_width, est_height = None, None

            est_ret = dict(
                timestamp=estimation.unix_time,
                x_offset=est_x_offset,
                y_offset=est_y_offset,
                width=est_width,
                height=est_height,
                label=str(estimation.semantic_label),
                confidence=estimation.semantic_score,
                uuid=estimation.uuid,
                num_points=None,
                status=status,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            est_ret = {k: None for k in self.keys()}

        return {"ground_truth": gt_ret, "estimation": est_ret}

    def summarize_error(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate mean, sigma, RMS, max and min of error.

        Args:
            df (Optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame
        """

        def _summarize(
            _column: Union[str, List[str]],
            _df: Optional[pd.DataFrame] = None,
        ) -> Dict[str, float]:
            err: np.ndarray = self.calculate_error(_column, _df, remove_nan=True)
            if len(err) == 0:
                logging.warning(f"The array of errors is empty for column: {_column}")
                return dict(average=np.nan, rms=np.nan, std=np.nan, max=np.nan, min=np.nan)
            err_avg = np.average(err)
            err_rms = np.sqrt(np.square(err).mean())
            err_std = np.std(err)
            err_max = np.max(np.abs(err))
            err_min = np.min(np.abs(err))
            return dict(average=err_avg, rms=err_rms, std=err_std, max=err_max, min=err_min)

        if df is None:
            df = self.df

        all_data: Dict[str, Dict[str, any]] = {}
        for label in self.__all_labels:
            data = {}
            df_ = df if label == "ALL" else df[df["label"] == label]

            data["x_offset"] = _summarize("x_offset", df_)
            data["y_offset"] = _summarize("y_offset", df_)
            data["width"] = _summarize("width", df_)
            data["height"] = _summarize("height", df_)
            all_data[str(label)] = data

        ret_df = pd.DataFrame.from_dict(
            {(i, j): all_data[i][j] for i in all_data.keys() for j in all_data[i].keys()},
            orient="index",
        )

        return ret_df

    def plot_num_object(
        self,
        show: bool = False,
        bin: int = 0.1,
        **kwargs,
    ) -> None:
        """Plot the number of objects per confidence.

        Args:
            show (bool): Whether show the plotted figure. Defaults to False.
            bin (int): Bin of axis. Defaults to False.
        """
        est_df = self.get_estimation(**kwargs)
        axes = self.mode.get_axes(est_df)
        fig: Figure = plt.figure(figsize=(16, 8))
        ax: Axes = fig.add_subplot(1, 1, 1, xlabel=self.mode.xlabel, ylabel="number of samples")
        ax.hist(axes, bins=bin)
        plt.savefig(osp.join(self.plot_directory, "number_of_samples.png"))
        if show:
            plt.show()
        plt.close()

    def plot_score(
        self,
        metrics: str,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot the specified metrics score per confidence.

        Args:
            metrics (str): Metrics name.
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        est_df = self.get_estimation(**kwargs)
        axes = self.mode.get_axes(est_df)
        score_df = self.summarize_score(scene=kwargs.get("scene"))
        num_labels: int = len(self.target_labels)
        fig: Figure = plt.figure(figsize=(16, 8 * num_labels))
        for i, target_label in enumerate(self.target_labels):
            ax: Axes = fig.add_subplot(
                1,
                i + 1,
                i + 1,
                xlabel=self.mode.xlabel,
                ylabel="number of samples",
                title=target_label,
            )
            label_idx = est_df["label"] == target_label
            label_axes = axes[label_idx]
            label_score = score_df[metrics][target_label]
            ax.scatter(np.mean(label_axes), label_score, s=300, marker="*")
        plt.savefig(osp.join(self.plot_directory, f"label_confidence_vs_{metrics}.png"))
        if show:
            plt.show()
        plt.close()
