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

from enum import Enum
import os
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from warnings import warn

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perception_eval.common.label import AutowareLabel
from perception_eval.common.object import DynamicObject
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics.metrics import MetricsScore
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.tool.utils import extract_area_results
from perception_eval.tool.utils import generate_area_points
from perception_eval.tool.utils import get_area_idx
from perception_eval.util.math import rotation_matrix_to_euler
from tqdm import tqdm
import yaml


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


class PlotMode(Enum):
    TIME = "time"
    DISTANCE = "distance"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[PlotMode, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class PerceptionPerformanceAnalyzer:
    """[summary]
    An class to analyze perception results.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
        num_area_division: int = 1,
    ) -> None:
        """[summary]
        Args:
            evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
            num_area_division (int): Number to divide area. Defaults to 1.
        """
        self.__config = evaluation_config
        self.__num_area_division: int = num_area_division

        self.__plot_dir: str = os.path.join(self.__config.result_root_directory, "plot")
        os.makedirs(self.__plot_dir, exist_ok=True)

        self.__frame_results: Dict[int, List[PerceptionFrameResult]] = {}

        # NOTE: all_labels = ["ALL", ...(target_labels)]
        self.__all_labels: List[str] = [label.value for label in self.config.target_labels]
        self.__all_labels.insert(0, "ALL")

        max_x: float = self.config.evaluation_config_dict.get("max_x_position", 100.0)
        max_y: float = self.config.evaluation_config_dict.get("max_y_position", 100.0)
        self.upper_rights, self.bottom_lefts = generate_area_points(
            self.num_area_division, max_x=max_x, max_y=max_y
        )
        self.__max_dist: float = max(max_x, max_y)
        self.__initialize()

    def __initialize(self) -> None:
        """[summary]
        Initialize attributes.
        """
        self.__num_scene: int = 0
        self.__num_frame: int = 0
        self.__df: pd.DataFrame = pd.DataFrame(
            columns=[
                "timestamp",
                "x",
                "y",
                "w",
                "l",
                "h",
                "yaw",
                "vx",
                "vy",
                "nn_point1",
                "nn_point2",
                "label",
                "uuid",
                "status",
                "area",
                "frame",
                "scene",
            ]
        )

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
        num_area_division: int = 1,
    ) -> PerceptionPerformanceAnalyzer:
        """[summary]
        Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.
            is_usecase (bool): Whether usecase or database evaluation is.
        Returns:
            PerceptionPerformanceAnalyzer
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, Any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, Any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, Any] = p_cfg["evaluation_config_dict"]
        eval_task_: str = eval_cfg_dict["evaluation_task"]
        if eval_task_ == "detection":
            frame_id = "base_link"
        elif eval_task_ == "tracking":
            frame_id = "map"
        elif eval_task_ == "prediction":
            frame_id = "map"
        else:
            raise ValueError(f"Unexpected evaluation task: {eval_task_}")

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id=frame_id,
            merge_similar_labels=p_cfg.get("merge_similar_labels", False),
            does_use_pointcloud=False,
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
        )

        return cls(evaluation_config, num_area_division)

    @property
    def config(self) -> PerceptionEvaluationConfig:
        return self.__config

    @property
    def num_area_division(self) -> int:
        return self.__num_area_division

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def plot_directory(self) -> str:
        return self.__plot_dir

    @property
    def frame_results(self) -> Dict[int, List[PerceptionFrameResult]]:
        return self.__frame_results

    @property
    def num_frame(self) -> int:
        return self.__num_frame

    @property
    def num_scene(self) -> int:
        return self.__num_scene

    @property
    def num_ground_truth(self) -> int:
        return self.get_num_ground_truth()

    @property
    def num_estimation(self) -> int:
        return self.get_num_estimation()

    @property
    def num_tp(self) -> int:
        return self.get_num_tp()

    @property
    def num_fp(self) -> int:
        return self.get_num_fp()

    @property
    def num_fn(self) -> int:
        return self.get_num_fn()

    def get_num_ground_truth(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """[summary]
        Returns the number of ground truths.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            int: The number of ground truths.
        """
        df_ = self.get_ground_truth(**kwargs)
        return len(df_)

    def get_num_estimation(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """[summary]
        Returns the number of estimations.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            int: The number of estimations.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return len(df_)

    def get_num_tp(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """[summary]
        Returns the number of TP.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            inf: The number of TP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "TP")

    def get_num_fp(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """[summary]
        Returns the number of FP.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            inf: The number of FP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "FP")

    def get_num_fn(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """[summary]
        Returns the number of FN.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FN that their columns are specified value.
        Returns:
            inf: The number of FN.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return sum(df_["status"] == "FN")

    def get_ground_truth(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """[summary]
        Returns the DataFrame for ground truth.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
        Returns:
            pandas.DataFrame
        """
        if df is None:
            df = self.df

        df = df.xs("ground_truth", level=1)
        df = df[~df["status"].isnull()]
        for key, item in kwargs.items():
            if item is None:
                continue
            df = df[df[key] == item]
        return df

    def get_estimation(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """[summary]
        Returns the DataFrame for estimation.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
        Returns:
            pandas.DataFrame
        """
        if df is None:
            df = self.df

        df = df.xs("estimation", level=1)
        df = df[~df["status"].isnull()]
        for key, item in kwargs.items():
            if item is None:
                continue
            df = df[df[key] == item]

        return df

    def __len__(self) -> int:
        return len(self.df)

    def get_metrics_score(self, frame_results: List[PerceptionFrameResult]) -> MetricsScore:
        """[summary]
        Returns the metrics score for each evaluator

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            metrics_score (MetricsScore): The final metrics score.
        """
        target_labels: List[AutowareLabel] = self.config.target_labels
        scene_results = {label: [[]] for label in target_labels}
        scene_num_gt = {label: 0 for label in target_labels}
        used_frame: List[int] = []

        for frame in frame_results:
            obj_results_dict = divide_objects(frame.object_results, target_labels)
            num_gt_dict = divide_objects_to_num(frame.frame_ground_truth.objects, target_labels)
            for label in target_labels:
                scene_results[label].append(obj_results_dict[label])
                scene_num_gt[label] += num_gt_dict[label]
            used_frame.append(int(frame.frame_name))

        metrics_score: MetricsScore = MetricsScore(
            config=self.config.metrics_config,
            used_frame=used_frame,
        )
        if self.__config.metrics_config.detection_config is not None:
            metrics_score.evaluate_detection(scene_results, scene_num_gt)
        if self.__config.metrics_config.tracking_config is not None:
            metrics_score.evaluate_tracking(scene_results, scene_num_gt)
        if self.__config.metrics_config.prediction_config is not None:
            pass

        return metrics_score

    def analyze(
        self,
        scene: Optional[int] = None,
        area: Optional[int] = None,
        uuid: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """[summary]
        Analyze TP/FP/FN ratio, metrics score, error.

        Args:
            scene (Optional[int]): Specify if you want to analyze specific scene. Defaults to None.
            area (Optional[int]): Specify if you want to analyze specific area. Defaults to None.
            uuid (Optional[str]): Specify if you want to analyze specific uuid. Defaults to None.

        Returns:
            score_df (pandas.DataFrame)
            error_df (pandas.DataFrame)
        """
        df: pd.DataFrame = self.get(area=area, scene=scene, uuid=uuid)
        if len(df) > 0:
            ratio_df = self.summarize_ratio(df=df)
            error_df = self.summarize_error(df=df)
            metrics_df = self.summarize_score(area=area, scene=scene)

        try:
            score_df = pd.concat([ratio_df, metrics_df], axis=1)
        except Exception as e:
            warn(str(e))
            return pd.DataFrame(), pd.DataFrame()

        return score_df, error_df

    def add(self, frame_results: List[PerceptionFrameResult]) -> pd.DataFrame:
        """[summary]
        Add frame results and update DataFrame.

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            pandas.DataFrame
        """
        self.__num_scene += 1
        # columns: ["timestamp", "xyz", "wlh", "yaw", "velocity", "nn_points", "label", "state", "area"]
        start = len(self.df) // 2
        for frame in tqdm(frame_results, "Updating DataFrame"):
            concat: List[pd.DataFrame] = []
            if len(self) > 0:
                concat.append(self.df)

            tp_df = self.format2df(
                frame.pass_fail_result.tp_objects,
                status=MatchingStatus.TP,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(tp_df) > 0:
                start += len(tp_df) // 2
                concat.append(tp_df)

            fp_df = self.format2df(
                frame.pass_fail_result.fp_objects_result,
                status=MatchingStatus.FP,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(fp_df) > 0:
                start += len(fp_df) // 2
                concat.append(fp_df)

            fn_df = self.format2df(
                frame.pass_fail_result.fn_objects,
                status=MatchingStatus.FN,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(fn_df) > 0:
                start += len(fn_df) // 2
                concat.append(fn_df)

            if len(concat) > 0:
                self.__df = pd.concat(concat)

        self.__frame_results[self.num_scene] = frame_results
        self.__num_frame += len(frame_results)

        return self.__df

    def add_from_pkl(self, pickle_path: str) -> pd.DataFrame:
        """[summary]
        Add frame results from pickle and update DataFrame.

        Args:
            pickle_path (str)

        Returns:
            pandas.DataFrame
        """
        with open(pickle_path, "rb") as pickle_file:
            frame_results: List[PerceptionFrameResult] = pickle.load(pickle_file)
        return self.add(frame_results)

    def clear(self) -> None:
        """[summary]
        Clear frame results and DataFrame.
        """
        self.__frame_results.clear()
        del self.__df
        self.__initialize()

    def format2df(
        self,
        object_results: List[Union[DynamicObject, DynamicObjectWithPerceptionResult]],
        status: MatchingStatus,
        frame_num: int,
        start: int = 0,
        ego2map: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """[summary]
        Format objects to pandas.DataFrame.

        Args:
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.
            start (int): Number of the first index. Defaults to 0.
            ego2map (Optional[np.ndarray]): Matrix to transform from ego coords to map coords. Defaults to None.

        Returns:
            df (pandas.DataFrame)
        """
        rets: Dict[int, Dict[str, Any]] = {}
        for i, obj_result in enumerate(object_results, start=start):
            rets[i] = self.format2dict(obj_result, status, frame_num, ego2map)

        df = pd.DataFrame.from_dict(
            {(i, j): rets[i][j] for i in rets.keys() for j in rets[i].keys()},
            orient="index",
            columns=self.keys(),
        )
        return df

    def format2dict(
        self,
        object_result: Union[DynamicObject, DynamicObjectWithPerceptionResult],
        status: MatchingStatus,
        frame_num: int,
        ego2map: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """[summary]
        Format objects to dict.

        Args:
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.
            ego2map (Optional[np.ndarray]): Matrix to transform from ego coords to map coords. Defaults to None.

        Returns:
            Dict[str, Dict[str, Any]]
        """
        if isinstance(object_result, DynamicObjectWithPerceptionResult):
            gt: Optional[DynamicObject] = object_result.ground_truth_object
            estimation: DynamicObject = object_result.estimated_object
            gt_point1, gt_point2 = object_result.plane_distance.ground_truth_nn_plane
            est_point1, est_point2 = object_result.plane_distance.estimated_nn_plane
        elif isinstance(object_result, DynamicObject):
            if status == MatchingStatus.FN:
                gt: DynamicObject = object_result
                estimation = None
            elif status == MatchingStatus.FP:
                estimation: DynamicObject = object_result
                gt = None
            else:
                raise ValueError("For DynamicObject status must be in FP or FN, but got {status}")
            gt_point1, gt_point2 = None, None
            est_point1, est_point2 = None, None
        elif object_result is None:
            gt = None
            estimation = None
            gt_point1, gt_point2 = None, None
            est_point1, est_point2 = None, None
        else:
            raise TypeError(f"Unexpected object type: {type(object_result)}")

        area: int = get_area_idx(
            self.config.frame_id,
            object_result,
            self.upper_rights,
            self.bottom_lefts,
            ego2map,
        )

        if gt:
            if self.config.frame_id == "map":
                src: np.ndarray = np.eye(4, 4)
                src[:3, :3] = gt.state.orientation.rotation_matrix
                src[:3, 3] = gt.state.position
                dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
                gt_x, gt_y = dst[:2, 3]
                gt_yaw = rotation_matrix_to_euler(dst[:3, :3])[-1].item()
            else:
                gt_x, gt_y = gt.state.position[:2]
                gt_yaw = gt.state.orientation.yaw_pitch_roll[0]

            gt_w, gt_l, gt_h = gt.state.size

            if gt.state.velocity:
                gt_vx, gt_vy = gt.state.velocity[:2]
            else:
                gt_vx, gt_vy = None, None

            gt_ret = dict(
                timestamp=gt.unix_time,
                x=gt_x,
                y=gt_y,
                w=gt_w,
                l=gt_l,
                h=gt_h,
                yaw=gt_yaw,
                vx=gt_vx,
                vy=gt_vy,
                nn_point1=gt_point1,
                nn_point2=gt_point2,
                label=str(gt.semantic_label),
                uuid=gt.uuid,
                status=status,
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            gt_ret = {k: None for k in self.keys()}

        if estimation:
            if self.config.frame_id == "map":
                src: np.ndarray = np.eye(4, 4)
                src[:3, :3] = estimation.state.orientation.rotation_matrix
                src[:3, 3] = estimation.state.position
                dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
                est_x, est_y = dst[:2, 3]
                est_yaw = rotation_matrix_to_euler(dst[:3, :3])[-1].item()
            else:
                est_x, est_y = estimation.state.position[:2]
                est_yaw = estimation.state.orientation.yaw_pitch_roll[0]

            est_w, est_l, est_h = estimation.state.size

            if estimation.state.velocity:
                est_vx, est_vy = estimation.state.velocity[:2]
            else:
                est_vx, est_vy = None, None

            est_ret = dict(
                timestamp=estimation.unix_time,
                x=est_x,
                y=est_y,
                w=est_w,
                l=est_l,
                h=est_h,
                yaw=est_yaw,
                vx=est_vx,
                vy=est_vy,
                nn_point1=est_point1,
                nn_point2=est_point2,
                label=str(estimation.semantic_label),
                uuid=estimation.uuid,
                status=status,
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            est_ret = {k: None for k in self.keys()}

        return {"ground_truth": gt_ret, "estimation": est_ret}

    def get(self, *args, **kwargs) -> pd.DataFrame:
        """[summary]
        Returns specified columns of DataFrame.
        Returns:
            pandas.DataFrame: Selected DataFrame.
        """
        df = self.df

        for key, item in kwargs.items():
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                df = df[df[key] in item]
            else:
                df = df[df[key] == item]

        if args:
            df = df[list(args)]

        return df

    def sortby(
        self,
        columns: Union[str, List[str]],
        df: Optional[pd.DataFrame] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """[summary]
        Sort DataFrame by specified column's values.
        Args:
            column (str): Name of column.
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            ascending (bool): Whether sort ascending order. Defaults to False.
        Returns:
            pandas.DataFrame: Sorted DataFrame.
        """
        if df is None:
            df = self.df

        return df.sort_values(columns, ascending=ascending)

    def keys(self) -> pd.Index:
        return self.df.keys()

    def shape(self, columns: Optional[Union[str, List[str]]] = None) -> Tuple[int]:
        """[summary]
        Get the shape of DataFrame or specified column(s).
        Args:
            columns (Optional[Union[str, List[str]]): Name of column(s).
        Returns:
            Tuple[int]: Shape.
        """
        if columns:
            return self.df[columns].shape
        return self.df.shape

    def head(self, n: int = 5) -> pd.DataFrame:
        """[summary]
        Returns the first `n` rows of DataFrame.
        Args:
            n (int): Number of rows to select.
        Returns:
            pandas.DataFrame: The first `n` rows of the caller object.
        """
        return self.df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """[summary]
        Returns the last `n` rows of DataFrame.
        Args:
            n (int): Number of rows to select.
        Returns:
            pandas.DataFrame: The last `n` rows of the caller object.
        """
        return self.df.tail(n)

    def calculate_error(
        self,
        column: Union[str, List[str]],
        df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """[summary]
        Calculate specified column's error for TP.

        Args:
            column (Union[str, List[str]]): name of column
            df (pandas.DataFrame): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
            np.ndarray: Array of error, in shape (N, M).
                N is number of TP, M is dimensions.
        """
        expects: Set[str] = set(
            (
                "x",
                "y",
                "w",
                "l",
                "h",
                "yaw",
                "vx",
                "vy",
                "nn_point1",
                "nn_point2",
            )
        )
        keys: Set[str] = set([column]) if isinstance(column, str) else set(column)
        if keys > expects:
            raise ValueError(f"Unexpected keys: {column}, expected: {expects}")

        if df is None:
            df = self.df

        df_ = df[df["status"] == "TP"]
        if isinstance(column, list):
            df_arr: np.ndarray = np.concatenate(
                [np.array(df_[col].to_list()) for col in column],
                axis=-1,
            )
        else:
            df_arr: np.ndarray = np.array(df_[column])
        gt_vals = df_arr[::2]
        est_vals = df_arr[1::2]
        err: np.ndarray = gt_vals - est_vals
        err = err[~np.isnan(err)]

        if column == "yaw":
            # Clip err from [-2pi, pi] to [0, pi]
            err[err > np.pi] = 2 * np.pi - err[err > np.pi]
            err[err < 0] = -((-err[err < 0] // np.pi) * np.pi + err[err < 0])
        else:
            err = np.abs(err)

        return err

    def summarize_error(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """[summary]
        Calculate mean, sigma, RMS, max and min of error.

        Args:
            df (Optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame
        """

        def _summarize(
            _column: Union[str, List[str]],
            _df: Optional[pd.DataFrame] = None,
        ) -> Dict[str, float]:
            err: np.ndarray = self.calculate_error(_column, _df)
            if len(err) == 0:
                return dict(average=np.nan, rms=np.nan, std=np.nan, max=np.nan, min=np.nan)
            err_avg = np.average(err, axis=0)
            err_rms = np.sqrt(np.square(err).mean(axis=0))
            err_std = np.std(err, axis=0)
            err_max = np.max(err, axis=0)
            err_min = np.min(err, axis=0)
            return dict(average=err_avg, rms=err_rms, std=err_std, max=err_max, min=err_min)

        if df is None:
            df = self.df

        all_data = {}
        for label in self.__all_labels:
            data = {}
            df_ = df if label == "ALL" else df[df["label"] == label]
            # xy
            data["x"] = _summarize("x", df_)
            data["y"] = _summarize("y", df_)

            # yaw
            data["yaw"] = _summarize("yaw", df_)

            # velocity
            data["vx"] = _summarize("vx", df_)
            data["vy"] = _summarize("vy", df_)

            # nn_plane
            data["nn_plane"] = _summarize(["nn_point1", "nn_point2"], df_)
            all_data[str(label)] = data

        ret_df = pd.DataFrame.from_dict(
            {(i, j): all_data[i][j] for i in all_data.keys() for j in all_data[i].keys()},
            orient="index",
        )

        return ret_df

    def summarize_ratio(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """[summary]
        Summarize TP/FP/FN ratio.

        Args:
            df (Optional[pandas.DataFrame]): Specify, if you want to use any filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame
        """
        if df is None:
            df = self.df

        data: Dict[str, List[float]] = {
            str(s): [0.0] * len(self.__all_labels) for s in MatchingStatus
        }
        for i, label in enumerate(self.__all_labels):
            if label == "ALL":
                label = None
            num_ground_truth: int = self.get_num_ground_truth(label=label)
            num_tp: int = self.get_num_tp(label=label)
            num_fn: int = self.get_num_fn(label=label)
            num_fp: int = self.get_num_fp(label=label)
            if num_ground_truth > 0:
                data["TP"][i] = num_tp / num_ground_truth
                data["FN"][i] = num_fn / num_ground_truth
                data["FP"][i] = num_fp / num_ground_truth

        return pd.DataFrame(data, index=self.__all_labels)

    def summarize_score(
        self,
        area: Optional[int] = None,
        scene: Optional[Union[int, List[int]]] = None,
    ) -> pd.DataFrame:
        """[summary]
        Summarize MetricsScore.

        Args:
            df (Optional[pandas.DataFrame]): Specify, if you want to use any filtered DataFrame. Defaults to None.
        Returns:
            pandas.DataFrame
        """
        if scene is None:
            frame_results = [x for v in self.frame_results.values() for x in v]
        else:
            scene: List[int] = [scene] if isinstance(scene, int) else scene
            frame_results = [x for k, v in self.frame_results.items() if k in scene for x in v]

        if area is not None:
            frame_results = extract_area_results(
                frame_results,
                area=area,
                upper_rights=self.upper_rights,
                bottom_lefts=self.bottom_lefts,
            )

        metrics_score = self.get_metrics_score(frame_results)
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
        return pd.DataFrame(data, index=self.__all_labels)

    def get_error_heatmap(self) -> Optional[np.ndarray]:
        pass

    def plot_num_object(
        self,
        mode: Union[str, PlotMode] = PlotMode.DISTANCE,
        bin: Optional[float] = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """[summary]
        Plot the number of objects for each time/distance range with histogram.

        Args:
            mode (Union[str, PlotMode]): Mode of plot used as x-axis, time or distance. Defaults to PlotMode.DISTANCE.
            bin (float): The interval of time/distance. If not specified, 0.1[s] for time and 0.5[m] for distance will be use.
                Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if len(kwargs) == 0:
            title = "ALL"
            filename = "ALL"
        else:
            title: str = ""
            filename: str = ""
            for key, item in kwargs.items():
                title += f"@{key.upper()}:{item} "
                filename += f"{item}_"
            title = title.rstrip(" ")
            filename = filename.rstrip("_")

        gt_df = self.get_ground_truth(**kwargs)
        est_df = self.get_estimation(**kwargs)

        if mode == PlotMode.TIME:
            gt_times = np.array(gt_df["timestamp"], dtype=np.uint64) / 1e6
            est_times = np.array(est_df["timestamp"], dtype=np.uint64) / 1e6
            time_origin: float = gt_times.min().item()
            gt_values = gt_times - time_origin
            est_values = est_times - time_origin
            max_time = max(gt_values.max(), est_values.max())
            time_bin = bin if bin else 0.1
            bins = np.arange(0, max_time, time_bin)
            xlabel: str = str(mode) + " [s]"
        elif mode == PlotMode.DISTANCE:
            gt_values = np.linalg.norm(gt_df[["x", "y"]], axis=1)
            est_values = np.linalg.norm(est_df[["x", "y"]], axis=1)
            dist_bin = bin if bin else 0.5
            bins = np.arange(0, self.__max_dist, dist_bin)
            xlabel: str = str(mode) + " [m]"
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        fig: Figure = plt.figure(figsize=(16, 8))
        ax1: Axes = fig.add_subplot(
            1,
            2,
            1,
            xlabel=xlabel,
            ylabel="num",
            title="GT",
        )
        ax2: Axes = fig.add_subplot(
            1,
            2,
            2,
            xlabel=xlabel,
            ylabel="num",
            title="Estimation",
        )

        ax1.hist(gt_values, bins=bins)
        ax2.hist(est_values, bins=bins)

        plt.suptitle(f"{title}")
        plt.savefig(os.path.join(self.plot_directory, f"num_object_{filename}.png"))
        if show:
            plt.show()
        plt.close()

    def plot_state(
        self,
        uuid: str,
        columns: Union[str, List[str]],
        mode: Union[str, PlotMode] = PlotMode.TIME,
        status: Optional[MatchingStatus] = None,
        show: bool = False,
    ) -> None:
        """[summary]
        Plot states for each time/distance estimated and GT object in TP.

        Args:
            uuid (str): Target object's uuid.
            columns (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            mode (Union[str, PlotMode]): Mode of plot used as x-axis, time or distance. Defaults to PlotMode.TIME.
            status (Optional[int]): Target status TP/FP/FN. If not specified, plot all status. Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

        gt_df = self.get_ground_truth(uuid=uuid, status=status)
        gt_df["timestamp"] = gt_df["timestamp"].astype(np.uint64)
        index = pd.unique(gt_df.index.get_level_values(level=0))

        if len(index) == 0:
            warn(f"There is no object ID: {uuid}")
            return

        est_df = self.get_estimation(df=self.df.loc[index])
        est_df["timestamp"] = est_df["timestamp"].astype(np.uint64)

        if mode == PlotMode.TIME:
            gt_times = np.array(gt_df["timestamp"], dtype=np.uint64) / 1e6
            est_times = np.array(est_df["timestamp"], dtype=np.uint64) / 1e6
            time_origin: float = gt_times.min().item()
            gt_xaxes = gt_times - time_origin
            est_xaxes = est_times - time_origin
            xlabel: str = str(mode) + " [s]"
        elif mode == PlotMode.DISTANCE:
            gt_xaxes = np.linalg.norm(gt_df[["x", "y"]], axis=1)
            est_xaxes = np.linalg.norm(est_df[["x", "y"]], axis=1)
            xlabel: str = str(mode) + " [m]"
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        # Plot GT and estimation
        num_cols = len(columns)
        fig: Figure = plt.figure(figsize=(8 * num_cols, 4))
        for n, col in enumerate(columns):
            ylabel: str = f"{col}"
            ax: Axes = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=f"{ylabel} / {xlabel}",
            )
            ax.grid(lw=0.5)
            gt_states = np.array(gt_df[col].tolist())
            est_states = np.array(est_df[col].tolist())
            ax.scatter(gt_xaxes, gt_states, c="red", s=100)
            ax.scatter(est_xaxes, est_states)

        plt.suptitle(f"{columns} @uuid:{uuid}")
        plt.tight_layout()
        columns_str: str = "".join(columns)
        plt.savefig(
            os.path.join(
                self.plot_directory,
                f"state_{columns_str}_{uuid}_{str(mode)}.png",
            )
        )
        if show:
            plt.show()
        plt.close()

    def plot_error(
        self,
        columns: Union[str, List[str]],
        mode: Union[str, PlotMode] = PlotMode.TIME,
        heatmap: bool = False,
        gridsize: int = 30,
        show: bool = False,
        **kwargs,
    ) -> None:
        """[summary]
        Plot states for each time/distance estimated and GT object in TP.

        Args:
            columns (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            mode (Union[str, PlotMode]): Mode of plot used as x-axis, time or distance. Defaults to PlotMode.TIME.
            heatmap (bool): Whether overlay heatmap. Defaults to False.
            gridsize (int): Grid size to plot heatmap. Defaults to 30.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

        tp_gt_df = self.get_ground_truth(status="TP", **kwargs)
        tp_index = pd.unique(tp_gt_df.index.get_level_values(level=0))

        if len(tp_index) == 0:
            warn("There is no TP object")
            return

        tp_df = self.df.loc[tp_index]

        if mode == PlotMode.TIME:
            tp_times = np.array(tp_gt_df["timestamp"], dtype=np.uint64) / 1e6
            xaxes: np.ndarray = tp_times - tp_times.min().item()
            xlabel: str = str(mode) + " [s]"
        elif mode == PlotMode.DISTANCE:
            xaxes: np.ndarray = np.linalg.norm(tp_gt_df[["x", "y"]], axis=1)
            xlabel = str(mode) + " [m]"
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        # Plot GT and estimation
        num_cols = len(columns)
        fig: Figure = plt.figure(figsize=(8 * num_cols, 8))
        for n, col in enumerate(columns):
            ylabel: str = f"err_{col}"
            ax: Axes = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=f"{ylabel} / {xlabel}",
            )
            ax.grid(lw=0.5)
            err: np.ndarray = self.calculate_error(col, df=tp_df)
            if heatmap:
                ax.hexbin(xaxes, err, gridsize=gridsize, cmap="jet")
            else:
                ax.scatter(xaxes, err)

        plt.suptitle(f"{columns}")
        plt.tight_layout()
        columns_str: str = "".join(columns)
        columns_str += "_heatmap" if heatmap else ""
        plt.savefig(os.path.join(self.plot_directory, f"error_{columns_str}_{str(mode)}.png"))
        if show:
            plt.show()
        plt.close()

    def box_plot(
        self,
        columns: Union[str, List[str]],
        show: bool = False,
    ) -> None:
        """[summary]
        Plot box-plot of errors.

        Args:
            column (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

        errs: List[np.ndarray] = []
        for col in columns:
            errs.append(self.calculate_error(col))
        _, ax = plt.subplots()
        ax.boxplot(errs)
        ax.set_xticklabels(columns)

        plt.title("Box-Plot of Errors")
        plt.grid()
        plt.tight_layout()
        columns_str: str = "".join(columns)
        plt.savefig(os.path.join(self.plot_directory, f"box_plot_{columns_str}.png"))
        if show:
            plt.show()
        plt.close()
