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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import yaml

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.status import MatchingStatus
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult

from .perception_analyzer_base import PerceptionAnalyzerBase
from .utils import PlotAxes


class PerceptionAnalyzer2D(PerceptionAnalyzerBase):
    """An class to analyze perception results.

    Attributes:
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
        num_tn (int): Number of TN GT objects.
        num_fn (int): Number of FN GT objects.

    Args:
        evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        super().__init__(evaluation_config=evaluation_config)
        if not self.config.evaluation_task.is_2d():
            raise RuntimeError("Evaluation task must be 2D.")

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

        return cls(evaluation_config)

    @property
    def columns(self) -> List[str]:
        return [
            "frame_id",
            "timestamp",
            "x",
            "y",
            "width",
            "height",
            "label",
            "label_name",
            "attributes",
            "confidence",
            "uuid",
            "status",
            "frame",
            "scene",
        ]

    @property
    def state_columns(self) -> List[str]:
        return ["x", "y", "width", "height"]

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
            if status == MatchingStatus.FP:
                estimation: DynamicObject2D = object_result
                gt = None
            elif status == MatchingStatus.TN:
                estimation = None
                gt: DynamicObject2D = object_result
            elif status == MatchingStatus.FN:
                estimation = None
                gt: DynamicObject2D = object_result
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
                frame_id=gt.frame_id.value,
                timestamp=gt.unix_time,
                x=gt_x_offset,
                y=gt_y_offset,
                width=gt_width,
                height=gt_height,
                label=str(gt.semantic_label.label),
                label_name=gt.semantic_label.name,
                attributes=gt.semantic_label.attributes,
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
                frame_id=estimation.frame_id,
                timestamp=estimation.unix_time,
                x=est_x_offset,
                y=est_y_offset,
                width=est_width,
                height=est_height,
                label=str(estimation.semantic_label.label),
                label_name=estimation.semantic_label.name,
                attributes=estimation.semantic_label.attributes,
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

    def analyze(self, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Analyze TP/FP/TN/FN ratio, metrics score, error. If there is no DataFrame to be able to analyze returns None.

        Args:
            **kwargs: Specify scene, frame, area or uuid.

        Returns:
            score_df (Optional[pandas.DataFrame]): DataFrame of TP/FP/TN/FN ratios and metrics scores.
            confusion_matrix_df (Optional[pandas.DataFrame]): DataFrame of confusion matrix.
        """
        df: pd.DataFrame = self.get(**kwargs)
        if len(df) > 0:
            ratio_df = self.summarize_ratio(df=df)
            confusion_matrix_df = self.get_confusion_matrix(df=df)
            metrics_df = self.summarize_score(scene=kwargs.get("scene"))
            score_df = pd.concat([ratio_df, metrics_df], axis=1)
            return score_df, confusion_matrix_df

        logging.warning("There is no DataFrame to be able to analyze.")
        return None, None

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
            if len(_df) == 0:
                return dict(average=np.nan, rms=np.nan, std=np.nan, max=np.nan, min=np.nan)

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
        for label in self.all_labels:
            data = {}
            if label == "ALL":
                df_ = df
            else:
                gt_df = self.get_ground_truth(df=df, status=["TP", "FP", "TN"], label=label)
                index = pd.unique(gt_df.index.get_level_values(level=0))
                if len(index) == 0:
                    logging.warning(f"There is no TP/FP/TN object for {label}.")
                    df_ = pd.DataFrame()
                else:
                    df_ = self.df.loc[index]

            data["x"] = _summarize("x", df_)
            data["y"] = _summarize("y", df_)
            data["width"] = _summarize("width", df_)
            data["height"] = _summarize("height", df_)
            all_data[str(label)] = data

        ret_df = pd.DataFrame.from_dict(
            {(i, j): all_data[i][j] for i in all_data.keys() for j in all_data[i].keys()},
            orient="index",
        )

        return ret_df

    def get_confusion_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns confusion matrix as DataFrame.

        Args:
            df (Optional[pd.DataFrame]): Specify if you want to use filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame: Confusion matrix.
        """
        gt_df, est_df = self.get_pair_results(df)

        target_labels: List[str] = self.target_labels.copy()
        if self.config.label_params["allow_matching_unknown"] and "unknown" not in target_labels:
            target_labels.append("unknown")

        gt_indices: np.ndarray = gt_df["label"].apply(lambda label: target_labels.index(label)).to_numpy()
        est_indices: np.ndarray = est_df["label"].apply(lambda label: target_labels.index(label)).to_numpy()

        num_classes = len(target_labels)
        indices = num_classes * gt_indices + est_indices
        matrix: np.ndarray = np.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
        return pd.DataFrame(data=matrix, index=target_labels, columns=target_labels)

    def plot_num_object(
        self,
        show: bool = False,
        bins: int = 1,
        **kwargs,
    ) -> None:
        """Plot the number of objects per confidence.

        Args:
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (int): Bin of axis. Defaults to False.
        """
        return super().plot_num_object(mode=PlotAxes.CONFIDENCE, show=show, bins=bins, **kwargs)

    def plot_error(
        self,
        columns: Union[str, List[str]],
        heatmap: bool = False,
        show: bool = False,
        bins: int = 50,
        **kwargs,
    ) -> None:
        if self.config.evaluation_task == EvaluationTask.CLASSIFICATION2D:
            raise RuntimeError("For classification 2D, `plot_error` is not supported.")
        return super().plot_error(
            columns=columns,
            mode=PlotAxes.CONFIDENCE,
            heatmap=heatmap,
            show=show,
            bins=bins,
            **kwargs,
        )

    def plot_ratio(
        self,
        status: Union[str, MatchingStatus],
        show: bool = False,
        bins: float = 1,
        **kwargs,
    ) -> None:
        """Plot TP/FP/TN/FN ratio per confidence.

        Args:
            status (Union[str, MatchingStatus])
            show (bool): Whether show plot results. Defaults to None.
        """
        return super().plot_ratio(
            status=status,
            mode=PlotAxes.CONFIDENCE,
            show=show,
            bins=bins,
            **kwargs,
        )

    def box_plot(self, columns: Union[str, List[str]], show: bool = False, **kwargs) -> None:
        if isinstance(columns, str):
            columns = [columns]
        if set(columns) > set(["x", "y", "width", "height"]):
            raise ValueError(f"{columns} is unsupported for plot")
        return super().box_plot(columns, show, **kwargs)
