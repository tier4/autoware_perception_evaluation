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

from collections.abc import Iterable
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.status import MatchingStatus
from perception_eval.common.transform import TransformDict
from perception_eval.common.transform import TransformKey
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
import yaml

from .perception_analyzer_base import PerceptionAnalysisResult
from .perception_analyzer_base import PerceptionAnalyzerBase
from .utils import extract_area_results
from .utils import filter_frame_by_distance
from .utils import generate_area_points
from .utils import get_area_idx
from .utils import get_metrics_info
from .utils import PlotAxes

# TODO: Refactor plot methods


class PerceptionAnalyzer3D(PerceptionAnalyzerBase):
    """An analyzer class for 3D perception evaluation results.

    Attributes:
        config (PerceptionEvaluationConfig): Configurations for evaluation parameters.
        target_labels (List[str]): Target labels list. (e.g. ["car", "pedestrian", "motorbike"]).
        all_labels (List[str]): Target labels list including "ALL". (e.g. ["ALL", "car", "pedestrian", "motorbike"]).
        num_area_division (int): Number of area separations.
        upper_rights (numpy.ndarray): Upper right points of each separated area.
        bottom_lefts (numpy.ndarray): Bottom left points of each separated area.
        columns (List[str]): List of columns in `df`.
            `["frame_id", "timestamp", "x", "y", "width", "length", "height", "yaw", "vx", "vy", "speed", "nn_point1", "nn_point2",\
                "label", "label_name", "attributes", "confidence", "uuid", \
                "num_points", "status", "area", "frame", "scene"]`.
        state_columns (List[str]): List of state columns in `df`.
            `["x", "y", "width", "length", "height", "yaw", "vx", "vy", "speed", "nn_point1", "nn_point2"]`.
        plot_directory (str): Directory path to save plot.
        frame_results (Dict[str, List[PerceptionFrameResult]]):
            Hashmap of frame results, which key is the number of scene and value is frame results.
        num_frame (int): Number of frames.
        num_scene (int): Number of scenes.
        num_ground_truth (int): Number of GT objects.
        num_estimation (int): Number of estimations.
        num_tp (int): Number of TP results.
        num_fp (int): Number of FP results.
        num_tn (int): Number of TN GT objects.
        num_fn (int): Number of FN GT objects.

    Args:
        evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
        num_area_division (int): Number to divide area. Defaults to 1.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
        num_area_division: int = 1,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config)

        if not self.config.evaluation_task.is_3d():
            raise RuntimeError("Evaluation task must be 3D.")

        self.__num_area_division: int = num_area_division
        max_x: float = self.config.evaluation_config_dict.get("max_x_position", 100.0)
        max_y: float = self.config.evaluation_config_dict.get("max_y_position", 100.0)
        self.__upper_rights, self.__bottom_lefts = generate_area_points(
            self.num_area_division, max_x=max_x, max_y=max_y
        )

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
        num_area_division: int = 1,
    ) -> PerceptionAnalyzer3D:
        """Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.
            num_area_division (int): Number to divide evaluation target area.

        Returns:
            PerceptionAnalyzer3D: PerceptionAnalyzer3D instance.

        Raises:
            ValueError: When unexpected evaluation task is specified in scenario file.
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, Any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, Any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, Any] = p_cfg["evaluation_config_dict"]

        eval_cfg_dict["label_prefix"] = "autoware"

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id="base_link" if eval_cfg_dict["evaluation_task"] == "detection" else "map",
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
            load_raw_data=False,
        )

        return cls(evaluation_config, num_area_division)

    @property
    def columns(self) -> List[str]:
        return [
            "frame_id",
            "timestamp",
            "x",
            "y",
            "width",
            "length",
            "height",
            "bev_area",
            "yaw",
            "vx",
            "vy",
            "v_yaw",
            "speed",
            "nn_point1",
            "nn_point2",
            "label",
            "label_name",
            "attributes",
            "confidence",
            "uuid",
            "num_points",
            "status",
            "distance",
            "area",
            "frame",
            "scene",
        ]

    @property
    def state_columns(self) -> List[str]:
        return [
            "x",
            "y",
            "width",
            "length",
            "height",
            "bev_area",
            "yaw",
            "vx",
            "vy",
            "v_yaw",
            "speed",
            "nn_point1",
            "nn_point2",
            "nn_plane",
            "distance",
        ]

    @property
    def num_area_division(self) -> int:
        return self.__num_area_division

    @property
    def upper_rights(self) -> np.ndarray:
        return self.__upper_rights

    @property
    def bottom_lefts(self) -> np.ndarray:
        return self.__bottom_lefts

    def format2dict(
        self,
        object_result: Union[DynamicObject, DynamicObjectWithPerceptionResult],
        status: MatchingStatus,
        frame_num: int,
        transforms: Optional[TransformDict] = None,
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
            gt: Optional[DynamicObject] = object_result.ground_truth_object
            estimation: DynamicObject = object_result.estimated_object
            gt_point1, gt_point2 = object_result.plane_distance.ground_truth_nn_plane
            est_point1, est_point2 = object_result.plane_distance.estimated_nn_plane
        elif isinstance(object_result, DynamicObject):
            if status == MatchingStatus.FP:
                estimation: DynamicObject = object_result
                gt = None
            elif status == MatchingStatus.TN:
                estimation = None
                gt: DynamicObject = object_result
            elif status == MatchingStatus.FN:
                estimation = None
                gt: DynamicObject = object_result
            else:
                raise ValueError(f"For DynamicObject status must be in FP/TN/FN, but got {status}")
            gt_point1, gt_point2 = (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
            est_point1, est_point2 = (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
        elif object_result is None:
            gt, estimation = None, None
            gt_point1, gt_point2 = (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
            est_point1, est_point2 = (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
        else:
            raise TypeError(f"Unexpected object type: {type(object_result)}")

        area: int = get_area_idx(
            object_result=object_result,
            upper_rights=self.upper_rights,
            bottom_lefts=self.bottom_lefts,
            transforms=transforms,
        )

        if gt:
            if gt.state.velocity is not None:
                gt_vx, gt_vy = gt.state.velocity[:2]
                gt_speed = np.linalg.norm(gt.state.velocity[:2])
                gt_v_yaw = np.arctan2(gt_vy, gt_vx)
            else:
                gt_vx, gt_vy = np.nan, np.nan
                gt_speed = np.nan
                gt_v_yaw = np.nan

            transform_key = TransformKey(gt.frame_id, FrameID.BASE_LINK)
            gt_position, gt_rotation = transforms.transform(
                transform_key,
                gt.state.position,
                gt.state.orientation,
            )
            gt_x, gt_y, _ = gt_position
            gt_yaw, _, _ = gt_rotation.yaw_pitch_roll
            if not np.isnan(gt_point1).all():
                gt_point1 = transforms.transform(transform_key, gt_point1)
            if not np.isnan(gt_point2).all():
                gt_point2 = transforms.transform(transform_key, gt_point2)

            gt_w, gt_l, gt_h = gt.state.size
            bev_area = gt_w * gt_l

            gt_ret = dict(
                frame_id=gt.frame_id.value,
                timestamp=gt.unix_time,
                x=gt_x,
                y=gt_y,
                width=gt_w,
                length=gt_l,
                height=gt_h,
                bev_area=bev_area,
                yaw=gt_yaw,
                vx=gt_vx,
                vy=gt_vy,
                v_yaw=gt_v_yaw,
                speed=gt_speed,
                nn_point1=gt_point1,
                nn_point2=gt_point2,
                label=str(gt.semantic_label.label),
                label_name=gt.semantic_label.name,
                attributes=gt.semantic_label.attributes,
                confidence=gt.semantic_score,
                uuid=gt.uuid,
                num_points=gt.pointcloud_num,
                status=str(status),
                distance=np.linalg.norm([gt_x, gt_y]).item(),
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            gt_ret = {}
            for key in self.keys():
                if key in ("nn_point1", "nn_point2"):
                    gt_ret[key] = (np.nan, np.nan, np.nan)
                else:
                    gt_ret[key] = None

        if estimation:
            if estimation.state.velocity is not None:
                est_vx, est_vy = estimation.state.velocity[:2]
                est_v_yaw = np.arctan2(est_vy, est_vx)
                est_speed = np.linalg.norm(estimation.state.velocity[:2])
            else:
                est_vx, est_vy = np.nan, np.nan
                est_v_yaw = np.nan
                est_speed = np.nan

            transform_key = TransformKey(estimation.frame_id, FrameID.BASE_LINK)
            est_position, est_rotation = transforms.transform(
                transform_key,
                estimation.state.position,
                estimation.state.orientation,
            )
            est_x, est_y, _ = est_position
            est_yaw, _, _ = est_rotation.yaw_pitch_roll
            if not np.isnan(est_point1).all():
                est_point1 = transforms.transform(transform_key, est_point1)
            if not np.isnan(est_point2).all():
                est_point2 = transforms.transform(transform_key, est_point2)

            est_w, est_l, est_h = estimation.state.size
            bev_area = est_w * est_l

            est_ret = dict(
                frame_id=estimation.frame_id.value,
                timestamp=estimation.unix_time,
                x=est_x,
                y=est_y,
                width=est_w,
                length=est_l,
                height=est_h,
                bev_area=bev_area,
                yaw=est_yaw,
                vx=est_vx,
                vy=est_vy,
                v_yaw=est_v_yaw,
                speed=est_speed,
                nn_point1=est_point1,
                nn_point2=est_point2,
                label=str(estimation.semantic_label.label),
                label_name=estimation.semantic_label.name,
                attributes=estimation.semantic_label.attributes,
                confidence=estimation.semantic_score,
                uuid=estimation.uuid,
                num_points=np.nan,
                status=str(status),
                distance=np.linalg.norm([est_x, est_y]).item(),
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            est_ret = {}
            for key in self.keys():
                if key in ("nn_point1", "nn_point2"):
                    est_ret[key] = (np.nan, np.nan, np.nan)
                else:
                    est_ret[key] = None

        return {"ground_truth": gt_ret, "estimation": est_ret}

    def filter_by_distance(self, distance: Iterable[float], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Filter DataFrame by min, max distances.

        Args:
            distance (Iterable[float]): Range of distance ordering (min, max). The range must be min < max.
            df (Optional[pd.DataFrame], optional): Target DataFrame. If `None`, `self.df` will be used. Defaults to None.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        assert len(distance) == 2, "distance must be (min, max)"
        assert distance[0] < distance[1], f"distance must be (min, max) and min < max, but got {distance}"

        if df is None:
            df = self.df

        mask = np.array(
            [
                [np.logical_and(distance[0] <= group["distance"], group["distance"] < distance[1]).any()] * 2
                for _, group in df.groupby(level=0)
            ]
        ).reshape(-1)
        return df[mask]

    def analyze(
        self,
        scene: Optional[int] = None,
        distance: Optional[Iterable[float]] = None,
        area: Optional[int] = None,
        **kwargs,
    ) -> PerceptionAnalysisResult:
        if scene is not None:
            kwargs.update({"scene": scene})
        if area is not None:
            kwargs.update({"area": area})

        df: pd.DataFrame = self.get(**kwargs)
        if distance is not None:
            df = self.filter_by_distance(distance, df)

        if len(df) > 0:
            ratio_df = self.summarize_ratio(df=df)
            error_df = self.summarize_error(df=df)
            if "scene" in kwargs.keys():
                scene = kwargs.pop("scene")
            else:
                scene = None
            metrics_df = self.summarize_score(scene=scene, distance=distance, area=area)
            score_df = pd.concat([ratio_df, metrics_df], axis=1)
            confusion_matrix_df = self.get_confusion_matrix(df=df)
            return PerceptionAnalysisResult(score_df, error_df, confusion_matrix_df)
        else:
            logging.warning("There is no DataFrame to be able to analyze.")
            return PerceptionAnalysisResult()

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
                logging.warning(f"The array of errors is empty for column: {_column}")
                return dict(average=np.nan, rms=np.nan, std=np.nan, max=np.nan, min=np.nan, percentile_99=np.nan)

            err: np.ndarray = self.calculate_error(_column, _df, remove_nan=True)
            if len(err) == 0:
                logging.warning(f"The array of errors is empty for column: {_column}")
                return dict(average=np.nan, rms=np.nan, std=np.nan, max=np.nan, min=np.nan, percentile_99=np.nan)

            err_avg = np.average(err)
            err_rms = np.sqrt(np.square(err).mean())
            err_std = np.std(err)
            err_max = np.max(np.abs(err))
            err_min = np.min(np.abs(err))
            err_percentile_99 = np.percentile(np.abs(err), 99)
            return dict(
                average=err_avg, rms=err_rms, std=err_std, max=err_max, min=err_min, percentile_99=err_percentile_99
            )

        if df is None:
            df = self.df

        all_data = {}
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
            data["distance"] = _summarize("distance", df_)
            data["yaw"] = _summarize("yaw", df_)
            data["length"] = _summarize("length", df_)
            data["width"] = _summarize("width", df_)
            data["bev_area"] = _summarize("bev_area", df_)
            data["vx"] = _summarize("vx", df_)
            data["vy"] = _summarize("vy", df_)
            data["v_yaw"] = _summarize("v_yaw", df_)
            data["speed"] = _summarize("speed", df_)
            data["nn_plane"] = _summarize("nn_plane", df_)
            all_data[str(label)] = data

        ret_df = pd.DataFrame.from_dict(
            {(i, j): all_data[i][j] for i in all_data.keys() for j in all_data[i].keys()},
            orient="index",
        )

        return ret_df

    def summarize_score(
        self,
        scene: Optional[Union[int, List[int]]] = None,
        distance: Optional[Iterable[float]] = None,
        area: Optional[int] = None,
    ) -> pd.DataFrame:
        """Summarize MetricsScore.

        Args:
            scene (Optional[int]): Number of scene. If it is not specified, calculate metrics score for all scenes.
                Defaults to None.
            min_distance (Optional[float]): Min distance range. Defaults to None.
            max_distance (Optional[float]): Max distance range. Defaults to None.
            area (Optional[int]): Number of area. If it is not specified, calculate metrics score for all areas.
                Defaults to None.

        Returns:
            pandas.DataFrame
        """
        if scene is None:
            frame_results = [x for v in self.frame_results.values() for x in v]
        else:
            scene: List[int] = [scene] if isinstance(scene, int) else scene
            frame_results = [x for k, v in self.frame_results.items() if k in scene for x in v]

        if distance is not None:
            assert isinstance(distance, Iterable) and len(distance) == 2
            min_distance, max_distance = distance
            frame_results = [filter_frame_by_distance(frame, min_distance, max_distance) for frame in frame_results]

        if area is not None:
            frame_results = extract_area_results(
                frame_results,
                area=area,
                upper_rights=self.upper_rights,
                bottom_lefts=self.bottom_lefts,
            )

        metrics_score = self.get_metrics_score(frame_results)
        data: Dict[str, Any] = get_metrics_info(metrics_score)

        return pd.DataFrame(data, index=self.all_labels)

    def plot_state(
        self,
        uuid: str,
        columns: Union[str, List[str]],
        mode: PlotAxes = PlotAxes.TIME,
        status: Optional[MatchingStatus] = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot states for each time/distance estimated and GT object.

        Args:
            uuid (str): Target object's uuid.
            columns (Union[str, List[str]]): Target column name.
                Options: ["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "distance"].
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            status (Optional[int]): Target status TP/FP/TN/FN. If not specified, plot all status. Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]
        if set(columns) > set(["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "distance"]):
            raise ValueError(f"{columns} is unsupported for plot")
        return super().plot_state(
            uuid=uuid,
            columns=columns,
            mode=mode,
            status=status,
            show=show,
            **kwargs,
        )

    def plot_error(
        self,
        columns: Union[str, List[str]],
        mode: PlotAxes = PlotAxes.TIME,
        heatmap: bool = False,
        project: bool = False,
        show: bool = False,
        bins: int = 50,
        **kwargs,
    ) -> None:
        """Plot states for each time/distance estimated and GT object in TP.

        Args:
            columns (Union[str, List[str]]): Target column name.
                Options: ["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "nn_plane", "distance"].
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            heatmap (bool): Whether overlay heatmap. Defaults to False.
            project (bool): Whether to project heatmap on 2D. This argument is only used for heatmap plot.
                Defaults to False.
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (int): Bin size to plot heatmap. Defaults to 50.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]
        if set(columns) > set(["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "nn_plane", "distance"]):
            raise ValueError(f"{columns} is unsupported for plot")
        return super().plot_error(
            columns=columns, mode=mode, heatmap=heatmap, project=project, show=show, bins=bins, **kwargs
        )

    def box_plot(
        self,
        columns: Union[str, List[str]],
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot box-plot of errors.

        Args:
            column (Union[str, List[str]]): Target column name.
                Options: ["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "nn_plane", "distance"].
                If you want plot multiple column for one image, use List[str].
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]
        if set(columns) > set(["x", "y", "yaw", "width", "length", "vx", "vy", "speed", "nn_plane", "distance"]):
            raise ValueError(f"{columns} is unsupported for plot")
        return super().box_plot(columns=columns, show=show, **kwargs)
