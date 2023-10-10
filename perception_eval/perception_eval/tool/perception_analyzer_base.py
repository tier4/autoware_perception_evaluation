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
import os
import os.path as osp
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from tqdm import tqdm

from perception_eval.common.status import MatchingStatus
from perception_eval.evaluation.matching.objects_filter import divide_objects, divide_objects_to_num
from perception_eval.evaluation.metrics.metrics import MetricsScore

from .utils import PlotAxes, filter_df, get_metrics_info, setup_axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

    from perception_eval.common.label import LabelType
    from perception_eval.common.object import DynamicObject
    from perception_eval.config import PerceptionEvaluationConfig
    from perception_eval.evaluation import DynamicObjectWithPerceptionResult, PerceptionFrameResult


class PerceptionAnalyzerBase(ABC):
    """An abstract base class class for perception evaluation results analyzer.

    Attributes:
    ----------
        config (PerceptionEvaluationConfig): Configurations for evaluation parameters.
        target_labels (List[str]): Target labels list. (e.g. ["car", "pedestrian", "motorbike"]).
        all_labels (List[str]): Target labels list including "ALL". (e.g. ["ALL", "car", "pedestrian", "motorbike"]).
        df (pandas.DataFrame): Data frame.
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
    ----
        evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        self.__config = evaluation_config

        self.__plot_dir: str = os.path.join(self.__config.result_root_directory, "plot")
        if not os.path.exists(self.__plot_dir):
            os.makedirs(self.__plot_dir)

        self.__target_labels: list[str] = [label.value for label in self.config.target_labels]
        self.__all_labels: list[str] = self.__target_labels.copy()
        self.__all_labels.insert(0, "ALL")
        self.__initialize()

    def __initialize(self) -> None:
        """Initialize data cached in `self.add()` method."""
        self.__num_scene: int = 0
        self.__num_frame: int = 0
        self.__frame_results: dict[int, list[PerceptionFrameResult]] = {}
        self.__ego2maps: dict[str, dict[str, np.ndarray]] = {}
        self.__df: pd.DataFrame = pd.DataFrame(columns=self.columns)

    @classmethod
    @abstractmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
    ) -> PerceptionAnalyzerBase:
        pass

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def state_columns(self) -> list[str]:
        pass

    @property
    def config(self) -> PerceptionEvaluationConfig:
        return self.__config

    @property
    def target_labels(self) -> list[str]:
        return self.__target_labels

    @property
    def all_labels(self) -> list[str]:
        return self.__all_labels

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def plot_directory(self) -> str:
        return self.__plot_dir

    @property
    def frame_results(self) -> dict[int, list[PerceptionFrameResult]]:
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
    def num_tn(self) -> int:
        return self.get_num_tn()

    @property
    def num_fn(self) -> int:
        return self.get_num_fn()

    def get(self, *args, **kwargs) -> pd.DataFrame:
        """Returns specified columns of DataFrame.

        Returns:
        -------
            pandas.DataFrame: Selected DataFrame.
        """
        df = self.df
        return filter_df(df, *args, **kwargs)

    def sortby(
        self,
        columns: str | list[str],
        df: pd.DataFrame | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Sort DataFrame by specified column's values.

        Args:
        ----
            column (str): Name of column.
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            ascending (bool): Whether sort ascending order. Defaults to False.

        Returns:
        -------
            pandas.DataFrame: Sorted DataFrame.
        """
        if df is None:
            df = self.df

        return df.sort_values(columns, ascending=ascending)

    def keys(self) -> pd.Index:
        return self.df.keys()

    def shape(self, columns: str | list[str] | None = None) -> tuple[int]:
        """Get the shape of DataFrame or specified column(s).

        Args:
        ----
            columns (Optional[Union[str, List[str]]): Name of column(s).

        Returns:
        -------
            Tuple[int]: Shape.
        """
        if columns:
            return self.df[columns].shape
        return self.df.shape

    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns the first `n` rows of DataFrame.

        Args:
        ----
            n (int): Number of rows to select.

        Returns:
        -------
            pandas.DataFrame: The first `n` rows of the caller object.
        """
        return self.df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Returns the last `n` rows of DataFrame.

        Args:
        ----
            n (int): Number of rows to select.

        Returns:
        -------
            pandas.DataFrame: The last `n` rows of the caller object.
        """
        return self.df.tail(n)

    def get_num_ground_truth(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of ground truths.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.

        Returns:
        -------
            int: The number of ground truths.
        """
        df_ = self.get_ground_truth(**kwargs)
        return len(df_)

    def get_num_estimation(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of estimations.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.

        Returns:
        -------
            int: The number of estimations.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return len(df_)

    def get_status_num(
        self,
        status: str | MatchingStatus,
        df: pd.DataFrame | None = None,
        **kwargs,
    ) -> int:
        """Returns number of matching status TP/FP/TN/FN.

        Args:
        ----
            status (Union[str, MatchingStatus]): Status.
            df (Optional[pandas.DataFrame]): Target DataFrame. Defaults to None.
            **kwargs

        Returns:
        -------
            int: Number of matching status.
        """
        if status == MatchingStatus.TP:
            return self.get_num_tp(df, **kwargs)
        elif status == MatchingStatus.FP:
            return self.get_num_fp(df, **kwargs)
        elif status == MatchingStatus.TN:
            return self.get_num_tn(df, **kwargs)
        elif status == MatchingStatus.FN:
            return self.get_num_fn(df, **kwargs)
        else:
            msg = f"Expected status is TP/FP/TN/FN, but got {status}"
            raise ValueError(msg)

    def get_num_tp(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of TP.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.

        Returns:
        -------
            inf: The number of TP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "TP")

    def get_num_fp(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of FP.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.

        Returns:
        -------
            inf: The number of FP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "FP")

    def get_num_tn(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of TN.

        Args:
        ----

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of TN that their columns are specified value.

        Returns:
        -------
            inf: The number of TN.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return sum(df_["status"] == "TN")

    def get_num_fn(self, df: pd.DataFrame | None = None, **kwargs) -> int:
        """Returns the number of FN.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FN that their columns are specified value.

        Returns:
        -------
            inf: The number of FN.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return sum(df_["status"] == "FN")

    def get_ground_truth(self, df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """Returns the DataFrame for ground truth.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
        -------
            pandas.DataFrame.
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

    def get_estimation(self, df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """Returns the DataFrame for estimation.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
        -------
            pandas.DataFrame.
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

    def get_pair_results(
        self,
        df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns paired results, which means both the row of GT and estimation valid values.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
        -------
            pandas.DataFrame: GT DataFrame.
            pandas.DataFrame: Estimation DataFrame.
        """
        if df is None:
            df = self.df

        gt_df = df.xs("ground_truth", level=1)
        est_df = df.xs("estimation", level=1)
        valid_idx = np.bitwise_and(~gt_df["status"].isnull(), ~est_df["status"].isnull())
        gt_df = gt_df[valid_idx]
        est_df = est_df[valid_idx]
        return gt_df, est_df

    def get_scenes(self, df: pd.DataFrame | None = None, **kwargs) -> np.ndarray:
        """Returns numpy array of unique scenes.

        Args:
        ----
            df (optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
        -------
            numpy.ndarray.
        """
        if df is None:
            df = self.get(**kwargs)

        scenes: np.ndarray = pd.unique(df["scene"])
        return scenes[~np.isnan(scenes)]

    def get_ego2map(self, scene: int, frame: int) -> np.ndarray:
        """Returns 4x4 ego2map transform matrix.

        Args:
        ----
            scene (int): Number of scene.
            frame (int): Number of frame.

        Returns:
        -------
            numpy.ndarray: In shape (4, 4).
        """
        return self.__ego2maps[str(scene)][str(frame)]

    def __len__(self) -> int:
        return len(self.df)

    def get_metrics_score(self, frame_results: list[PerceptionFrameResult]) -> MetricsScore:
        """Returns the metrics score for each evaluator.

        Args:
        ----
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
        -------
            metrics_score (MetricsScore): The final metrics score.
        """
        target_labels: list[LabelType] = self.config.target_labels
        scene_results = {label: [[]] for label in target_labels}
        scene_num_gt = {label: 0 for label in target_labels}
        used_frame: list[int] = []

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
        if self.config.metrics_config.detection_config is not None:
            metrics_score.evaluate_detection(scene_results, scene_num_gt)
        if self.config.metrics_config.tracking_config is not None:
            metrics_score.evaluate_tracking(scene_results, scene_num_gt)
        if self.config.metrics_config.prediction_config is not None:
            pass
        if self.config.metrics_config.classification_config is not None:
            metrics_score.evaluate_classification(scene_results, scene_num_gt)

        return metrics_score

    def analyze(self, **kwargs) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Analyze TP/FP/FN ratio, metrics score, error. If there is no DataFrame to be able to analyze returns None.

        Args:
        ----
            **kwargs: Specify scene, frame, area or uuid.

        Returns:
        -------
            score_df (Optional[pandas.DataFrame]): DataFrame of TP/FP/FN ratios and metrics scores.
            error_df (Optional[pandas.DataFrame]): DataFrame of errors.
        """
        df: pd.DataFrame = self.get(**kwargs)
        if len(df) > 0:
            ratio_df = self.summarize_ratio(df=df)
            error_df = self.summarize_error(df=df)
            metrics_df = self.summarize_score(scene=kwargs.get("scene"), **kwargs)
            score_df = pd.concat([ratio_df, metrics_df], axis=1)
            return score_df, error_df

        logging.warning("There is no DataFrame to be able to analyze.")
        return None, None

    def add(self, frame_results: list[PerceptionFrameResult]) -> pd.DataFrame:
        """Add frame results and update DataFrame.

        Args:
        ----
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
        -------
            pandas.DataFrame
        """
        self.__num_scene += 1
        start = len(self.df) // 2
        self.__ego2maps[str(self.num_scene)] = {}
        for frame in tqdm(frame_results, "Updating DataFrame"):
            concat: list[pd.DataFrame] = []
            if len(self) > 0:
                concat.append(self.df)

            self.__ego2maps[str(self.num_scene)][str(frame.frame_name)] = frame.frame_ground_truth.ego2map

            tp_df = self.format2df(
                frame.pass_fail_result.tp_object_results,
                status=MatchingStatus.TP,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(tp_df) > 0:
                start += len(tp_df) // 2
                concat.append(tp_df)

            fp_df = self.format2df(
                frame.pass_fail_result.fp_object_results,
                status=MatchingStatus.FP,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(fp_df) > 0:
                start += len(fp_df) // 2
                concat.append(fp_df)

            tn_df = self.format2df(
                frame.pass_fail_result.tn_objects,
                status=MatchingStatus.TN,
                start=start,
                frame_num=int(frame.frame_name),
                ego2map=frame.frame_ground_truth.ego2map,
            )
            if len(tn_df) > 0:
                start += len(tn_df) // 2
                concat.append(tn_df)

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
        ----
            pickle_path (str)

        Returns:
        -------
            pandas.DataFrame
        """
        with open(pickle_path, "rb") as pickle_file:
            frame_results: list[PerceptionFrameResult] = pickle.load(pickle_file)
        return self.add(frame_results)

    def clear(self) -> None:
        """Clear frame results and DataFrame."""
        self.__frame_results.clear()
        del self.__df
        self.__initialize()

    def format2df(
        self,
        object_results: list[DynamicObject | DynamicObjectWithPerceptionResult],
        status: MatchingStatus,
        frame_num: int,
        start: int = 0,
        ego2map: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Format objects to pandas.DataFrame.

        Args:
        ----
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]):
                List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.
            start (int): Number of the first index. Defaults to 0.
            ego2map (Optional[np.ndarray]): Matrix to transform from ego coords to map coords. Defaults to None.

        Returns:
        -------
            df (pandas.DataFrame)
        """
        rets: dict[int, dict[str, Any]] = {}
        for i, obj_result in enumerate(object_results, start=start):
            rets[i] = self.format2dict(obj_result, status, frame_num, ego2map)

        return pd.DataFrame.from_dict(
            {(i, j): rets[i][j] for i in rets for j in rets[i]},
            orient="index",
            columns=self.keys(),
        )

    @abstractmethod
    def format2dict(
        self,
        object_result: DynamicObject | DynamicObjectWithPerceptionResult,
        status: MatchingStatus,
        frame_num: int,
        ego2map: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Format objects to dict.

        Args:
        ----
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]):
                List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.
            ego2map (Optional[np.ndarray]): Matrix to transform from ego coords to map coords. Defaults to None.

        Returns:
        -------
            Dict[str, Dict[str, Any]]
        """

    def calculate_error(
        self,
        column: str | list[str],
        df: pd.DataFrame | None = None,
        remove_nan: bool = False,
    ) -> np.ndarray:
        """Calculate specified column's error for TP.

        TODO: plot not only TP status, but also matched objects FP/TN.

        Args:
        ----
            column (Union[str, List[str]]): name of column
            df (pandas.DataFrame): Specify if you want use filtered DataFrame. Defaults to None.
            remove_nan (bool): Whether remove nan value. Defaults to False.

        Returns:
        -------
            np.ndarray: Array of error, in shape (N, M).
                N is number of TP, M is dimensions.
        """
        expects: set[str] = set(self.state_columns)
        keys: set[str] = {column} if isinstance(column, str) else set(column)
        if keys > expects:
            msg = f"Unexpected keys: {column}, expected: {expects}"
            raise ValueError(msg)

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
        if remove_nan:
            err = err[~np.isnan(err)]

        if column == "yaw":
            # Clip err from [-2pi, 2pi] to [-pi, pi]
            err[err > np.pi] = -2 * np.pi + err[err > np.pi]
            err[err < -np.pi] = 2 * np.pi + err[err < -np.pi]

        return err

    @abstractmethod
    def summarize_error(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculate mean, sigma, RMS, max and min of error.

        Args:
        ----
            df (Optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
        -------
            pd.DataFrame
        """

    def summarize_ratio(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Summarize TP/FP/FN ratio.

        Args:
        ----
            df (Optional[pandas.DataFrame]): Specify, if you want to use any filtered DataFrame. Defaults to None.

        Returns:
        -------
            pd.DataFrame
        """
        if df is None:
            df = self.df

        data: dict[str, list[float]] = {str(s): [0.0] * len(self.all_labels) for s in MatchingStatus}
        for i, label in enumerate(self.all_labels):
            if label == "ALL":
                label = None
            num_ground_truth: int = self.get_num_ground_truth(label=label)
            if num_ground_truth > 0:
                data["TP"][i] = self.get_num_tp(label=label) / num_ground_truth
                data["FP"][i] = self.get_num_fp(label=label) / num_ground_truth
                data["TN"][i] = self.get_num_tn(label=label) / num_ground_truth
                data["FN"][i] = self.get_num_fn(label=label) / num_ground_truth
        return pd.DataFrame(data, index=self.all_labels)

    def summarize_score(self, scene: int | list[int] | None = None) -> pd.DataFrame:
        """Summarize MetricsScore.

        Args:
        ----
            scene (Optional[int]): Number of scene. If it is not specified, calculate metrics score for all scenes.
                Defaults to None.

        Returns:
        -------
            pandas.DataFrame
        """
        if scene is None:
            frame_results = [x for v in self.frame_results.values() for x in v]
        else:
            scene: list[int] = [scene] if isinstance(scene, int) else scene
            frame_results = [x for k, v in self.frame_results.items() if k in scene for x in v]

        metrics_score = self.get_metrics_score(frame_results)
        data: dict[str, Any] = get_metrics_info(metrics_score)

        return pd.DataFrame(data, index=self.all_labels)

    def plot_num_object(
        self,
        mode: PlotAxes = PlotAxes.DISTANCE,
        show: bool = False,
        bins: float | None = None,
        **kwargs,
    ) -> None:
        """Plot the number of objects for each time/distance range with histogram.

        Args:
        ----
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.DISTANCE (1-dimensional).
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (float): The interval of time/distance. If not specified, 0.1[s] for time and 0.5[m]
                for distance will be use. Defaults to None.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if len(kwargs) == 0:
            title = "Number of Objects @all"
            filename = "all"
        else:
            title: str = f"Num Object @{self.all_labels!s}"
            filename: str = ""
            for key, item in kwargs.items():
                title += f"@{key.upper()}:{item} "
                filename += f"{item}_"
            title = title.rstrip(" ")
            filename = filename.rstrip("_")

        if mode.is_2d():
            xlabel: str = mode.xlabel
            ylabel: str = "Number of samples"
        else:
            xlabel, ylabel = mode.get_label()

        fig: Figure = plt.figure(figsize=(16, 8))
        ax: Axes | Axes3D = fig.add_subplot(
            xlabel=xlabel,
            ylabel=ylabel,
            title="Number of samples",
            projection=mode.projection,
        )
        mode.setup_axis(ax, **kwargs)
        gt_axes = mode.get_axes(self.get_ground_truth(**kwargs))
        est_axes = mode.get_axes(self.get_estimation(**kwargs))

        if mode.is_2d():
            min_value = 0 if mode == PlotAxes.CONFIDENCE else _get_min_value(gt_axes, est_axes)
            max_value = 100 if mode == PlotAxes.CONFIDENCE else _get_max_value(gt_axes, est_axes)
            step = bins if bins else mode.get_bins()
            hist_bins = np.arange(min_value, max_value + step, step)
            gt_hist, xaxis = np.histogram(gt_axes, bins=hist_bins)
            est_hist, _ = np.histogram(est_axes, bins=hist_bins)
            width = step if mode == PlotAxes.CONFIDENCE else 0.25 * ((max_value - min_value) / step)
            ax.bar(xaxis[:-1] - 0.5 * width, gt_hist, width, label="GT")
            ax.bar(xaxis[:-1] + 0.5 * width, est_hist, width, label="Estimation")
        else:
            ax.set_zlabel("Number of samples")
            gt_xaxes, gt_yaxes = gt_axes[:, ~np.isnan(gt_axes).any(0)]
            est_xaxes, est_yaxes = est_axes[:, ~np.isnan(est_axes).any(0)]
            gt_hist, gt_x_edges, gt_y_edges = np.histogram2d(gt_xaxes, gt_yaxes)
            est_hist, est_x_edges, est_y_edges = np.histogram2d(est_xaxes, est_yaxes)
            gt_x, gt_y = np.meshgrid(gt_x_edges[:-1], gt_y_edges[:-1])
            est_x, est_y = np.meshgrid(est_x_edges[:-1], est_y_edges[:-1])
            if bins is not None and isinstance(bins, float):
                bins = (bins, bins)
            dx, dy = mode.get_bins() if bins is None else bins
            dx *= 0.5
            dy *= 0.5
            ax.bar3d(gt_x.ravel(), gt_y.ravel(), 0, dx, dy, gt_hist.ravel(), alpha=0.6)
            ax.bar3d(est_x.ravel(), est_y.ravel(), 0, dx, dy, est_hist.ravel(), alpha=0.6)

        self.__post_process_figure(
            fig=fig,
            title=title,
            legend=True,
            filename=f"num_object_{mode!s}_{filename}",
            show=show,
        )

    def plot_state(
        self,
        uuid: str,
        columns: str | list[str],
        mode: PlotAxes = PlotAxes.TIME,
        status: MatchingStatus | None = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot states for each time/distance estimated and GT object in TP.

        Args:
        ----
            uuid (str): Target object's uuid.
            columns (Union[str, List[str]]): Target column name.
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            status (Optional[int]): Target status TP/FP/FN. If not specified, plot all status. Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs
        """
        if isinstance(columns, str):
            columns: list[str] = [columns]

        gt_df = self.get_ground_truth(uuid=uuid, status=status)
        index = pd.unique(gt_df.index.get_level_values(level=0))

        if len(index) == 0:
            logging.warning(f"There is no object ID: {uuid}")
            return

        est_df = self.get_estimation(df=self.df.loc[index])

        gt_axes = mode.get_axes(gt_df)
        est_axes = mode.get_axes(est_df)

        # Plot GT and estimation
        num_cols = len(columns)
        fig: Figure = plt.figure(figsize=(8 * num_cols, 4))
        for n, col in enumerate(columns):
            if mode.is_2d():
                xlabel: str = mode.xlabel
                ylabel: str = f"{col}"
                title: str = f"State {ylabel} = F({xlabel})"
            else:
                xlabel, ylabel = mode.get_label()
                title: str = f"State {col} = F({xlabel}, {ylabel})"
            ax: Axes = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                projection=mode.projection,
            )
            mode.setup_axis(ax, **kwargs)
            gt_states = np.array(gt_df[col].tolist())
            est_states = np.array(est_df[col].tolist())
            if mode.is_2d():
                ax.scatter(gt_axes, gt_states, label="GT", c="red", s=100)
                ax.scatter(est_axes, est_states, label="Estimation")
            else:
                ax.set_zlabel(f"{col}")
                gt_xaxes, gt_yaxes = gt_axes
                est_xaxes, est_yaxes = est_axes
                ax.scatter(gt_xaxes, gt_yaxes, gt_states, label="GT", c="red", s=100)
                ax.scatter(est_xaxes, est_yaxes, est_states, label="Estimation")
            ax.legend(loc="upper right", framealpha=0.4)

        columns_str: str = "".join(columns)
        self.__post_process_figure(
            fig=fig,
            title=f"State of {columns} @uuid:{uuid}",
            legend=False,
            filename=f"state_{columns_str}_{uuid}_{mode!s}",
            show=show,
        )

    def plot_error(
        self,
        columns: str | list[str],
        mode: PlotAxes = PlotAxes.TIME,
        heatmap: bool = False,
        show: bool = False,
        bins: int = 50,
        **kwargs,
    ) -> None:
        """Plot states for each time/distance estimated and GT object in TP.

        TODO: plot not only TP status, but also matched objects FP/TN.

        Args:
        ----
            columns (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "w", "l", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            heatmap (bool): Whether overlay heatmap. Defaults to False.
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (int): Bin size to plot heatmap. Defaults to 50.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if isinstance(columns, str):
            columns: list[str] = [columns]

        tp_gt_df = self.get_ground_truth(status="TP", **kwargs)
        tp_index = pd.unique(tp_gt_df.index.get_level_values(level=0))

        if len(tp_index) == 0:
            logging.warning("There is no TP object. Could not calculate error.")
            return

        tp_df = self.df.loc[tp_index]

        num_cols = len(columns)
        fig: Figure = plt.figure(figsize=(8 * num_cols, 8))
        for n, col in enumerate(columns):
            if mode.is_2d():
                xlabel: str = mode.xlabel
                ylabel: str = f"err_{col}"
                title: str = f"Error({col}) = F({xlabel})"
            else:
                xlabel, ylabel = mode.get_label()
                title: str = f"Error({col}) = F({xlabel}, {ylabel})"
            ax: Axes | Axes3D = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                projection=mode.projection,
            )

            mode.setup_axis(ax, **kwargs)
            err: np.ndarray = self.calculate_error(col, df=tp_df)
            axes: np.ndarray = mode.get_axes(tp_gt_df)
            if mode.is_2d():
                non_nan = ~np.isnan(err) * ~np.isnan(axes)
                axes = axes[non_nan]
                err = err[non_nan]
                if heatmap:
                    ax.hist2d(axes, err, bins=(bins, bins), cmap=cm.jet)
                else:
                    ax.scatter(axes, err)
            else:
                ax.set_zlabel(f"err_{col}")
                non_nan = ~np.isnan(err) * ~np.isnan(axes).any(0)
                xaxes, yaxes = axes[:, non_nan]
                err = err[non_nan]
                if heatmap:
                    ax.scatter(xaxes, yaxes, err, c=err, cmap=cm.jet)
                    color_map = cm.ScalarMappable(cmap=cm.jet)
                    color_map.set_array([err])
                    plt.colorbar(color_map)
                else:
                    ax.scatter(xaxes, yaxes, err)

        columns_str: str = "".join(columns)
        columns_str += "_heatmap" if heatmap else ""
        self.__post_process_figure(
            fig=fig,
            title=f"Errors@{columns}",
            legend=False,
            filename=f"error_{columns_str}_{mode!s}",
            show=show,
        )

    def box_plot(
        self,
        columns: str | list[str],
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot box-plot of errors.

        Args:
        ----
            column (Union[str, List[str]]): Target column name.
                If you want plot multiple column for one image, use List[str].
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: list[str] = [columns]

        fig, ax = plt.subplots()
        setup_axis(ax, **kwargs)

        df = self.get(**kwargs)
        errors: list[np.ndarray] = [self.calculate_error(col, df) for col in columns]
        ax.boxplot(errors)
        ax.set_xticklabels(columns)
        columns_str: str = "".join(columns)
        self.__post_process_figure(
            fig=fig,
            title=f"Box-plot of errors @{columns}",
            legend=True,
            filename=f"box_plot_{columns_str}",
            show=show,
        )

    def plot_ratio(
        self,
        status: str | MatchingStatus,
        mode: PlotAxes = PlotAxes.DISTANCE,
        show: bool = False,
        bins: float | None = None,
        plot_range: tuple[float] | None = None,
        **kwargs,
    ) -> None:
        """Plot TP/FP/TN/FN ratio.

        Args:
        ----
            status (Union[str, MatchingStatus]): Matching status, TP/FP/TN/FN.
            mode (PlotAxes): PlotAxes instance.
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (Optional[float]): Plot binds. Defaults to None.
            plot_range (Optional[Tuple[float, float]]): Range of plot. Defaults to None.
        """
        if mode.is_3d():
            msg = "3D plot is under construction."
            raise NotImplementedError(msg)

        gt_df = self.get_ground_truth(**kwargs)
        est_df = self.get_estimation(**kwargs)
        gt_df = gt_df[gt_df["status"] != np.nan]
        est_df = est_df[est_df["status"] != np.nan]
        gt_values: np.ndarray = mode.get_axes(gt_df)
        est_values: np.ndarray = mode.get_axes(est_df)

        xlabel: str = mode.xlabel
        ylabel: str = f"{status!s} ratio"
        if plot_range is not None:
            min_value, max_value = plot_range
        else:
            min_value = 0 if mode == PlotAxes.CONFIDENCE else _get_min_value(gt_values, est_values)
            max_value = 100 if mode == PlotAxes.CONFIDENCE else _get_max_value(gt_values, est_values)
        step = bins if bins else mode.get_bins()
        hist_bins = np.arange(min_value, max_value + step, step)
        _, axis = np.histogram(est_values, bins=hist_bins)

        fig: Figure = plt.figure(figsize=(16, 8))
        ax: Axes | Axes3D = fig.add_subplot(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{status!s} ratio",
            projection=mode.projection,
        )

        num_labels: int = len(self.all_labels)
        offsets = np.arange(-num_labels, num_labels, step=2.0)
        for n, target_label in enumerate(self.all_labels):
            axes = []
            ratios = []
            for i in range(len(axis) - 1):
                est_idx = (axis[i] <= est_values) * (est_values <= axis[i + 1])
                est_df_ = est_df[est_idx]
                gt_idx = (axis[i] <= gt_values) * (gt_values <= axis[i + 1])
                gt_df_ = gt_df[gt_idx]
                if target_label != "ALL":
                    est_df_ = est_df_[est_df_["label"] == target_label]
                    gt_df_ = gt_df_[gt_df_["label"] == target_label]
                if len(est_df_) > 0 and len(gt_df_) > 0:
                    num_gt: int = len(gt_df_)
                    if status == "TP":
                        ratios.append(np.sum(est_df_["status"] == status) / num_gt)
                    elif status == "FP":
                        ratios.append(1 - (np.sum(est_df_["status"] == "TP") / num_gt))
                    elif status == "TN":
                        ratios.append(np.sum(gt_df["status"] == status) / num_gt)
                    elif status == "FN":
                        ratios.append(np.sum(gt_df_["status"] == status) / num_gt)
                    else:
                        msg = f"Unexpected status: {status}"
                        raise ValueError(msg)
                else:
                    ratios.append(0) if mode.is_2d() else ratios.append((0, 0))
                axes.append((axis[i] + axis[i + 1]) * 0.5)
            axes = np.array(axes)
            # TODO: update not to exceed 1.0
            ratios = np.clip(ratios, 0.0, 1.0)
            width: float = 1.5
            ax.bar(axes + offsets[n] * width * 0.5, ratios, width=width, label=target_label)

        self.__post_process_figure(
            fig=fig,
            title=f"{status!s} ratio",
            legend=True,
            filename=f"{str(status).lower()}_ratio_{mode.value}",
            show=show,
        )

    def __post_process_figure(
        self,
        fig: Figure,
        title: str,
        legend: bool,
        filename: str,
        show: bool,
    ) -> None:
        """Post process of figure."""
        fig.suptitle(title)
        if legend:
            fig.legend()
        fig.tight_layout()
        fig.savefig(osp.join(self.plot_directory, f"{filename}.png"))
        if show:
            plt.show()
        fig.clear()
        plt.close()


def _get_min_value(value1: np.ndarray, value2: np.ndarray) -> float:
    return min(value1[~np.isnan(value1)].min(), value2[~np.isnan(value2)].min())


def _get_max_value(value1: np.ndarray, value2: np.ndarray) -> float:
    return max(value1[~np.isnan(value1)].max(), value2[~np.isnan(value2)].max())
