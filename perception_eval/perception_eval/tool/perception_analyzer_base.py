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

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from numbers import Number
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.status import MatchingStatus
from perception_eval.common.transform import TransformDict
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics.metrics import MetricsScore
from tqdm import tqdm

from .utils import get_metrics_info
from .utils import PlotAxes
from .utils import setup_axis


@dataclass
class PerceptionAnalysisResult:
    score: pd.DataFrame | None = None
    error: pd.DataFrame | None = None
    confusion_matrix: pd.DataFrame | None = None


class PerceptionAnalyzerBase(ABC):
    """An abstract base class class for perception evaluation results analyzer.

    Attributes:
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
        evaluation_config (PerceptionEvaluationConfig): Config used in evaluation.
    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        self.__config = evaluation_config

        self.__plot_dir: str = os.path.join(self.__config.result_root_directory, "plot")
        if not os.path.exists(self.__plot_dir):
            os.makedirs(self.__plot_dir)

        # NOTE: all_labels = ["ALL", ...(target_labels)]
        self.__target_labels: List[str] = [label.value for label in self.config.target_labels]
        self.__all_labels: List[str] = self.__target_labels.copy()
        self.__all_labels.insert(0, "ALL")
        self.__initialize()

    def __initialize(self) -> None:
        """Initialize data cached in `self.add()` method."""
        self.__num_scene: int = 0
        self.__num_frame: int = 0
        self.__frame_results: Dict[int, List[PerceptionFrameResult]] = {}
        self.__transforms: Dict[str, Dict[str, TransformDict]] = {}
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
    def columns(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def state_columns(self) -> List[str]:
        pass

    @property
    def config(self) -> PerceptionEvaluationConfig:
        return self.__config

    @property
    def target_labels(self) -> List[str]:
        return self.__target_labels

    @property
    def all_labels(self) -> List[str]:
        return self.__all_labels

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
    def num_tn(self) -> int:
        return self.get_num_tn()

    @property
    def num_fn(self) -> int:
        return self.get_num_fn()

    def get(self, *args, **kwargs) -> pd.DataFrame:
        """Returns specified columns of DataFrame.
        Returns:
            pandas.DataFrame: Selected DataFrame.
        """
        return self.filter(*args, **kwargs)

    def filter(self, *args, **kwargs) -> pd.DataFrame:
        """

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        df = self.df

        mask = np.ones(len(self), dtype=np.bool8)
        for key, item in kwargs.items():
            if item is None:
                continue
            elif not isinstance(item, str) and isinstance(item, Iterable):
                cur_mask = df[key].isin(item)
            else:
                cur_mask = df[key] == item
            mask *= cur_mask.groupby(level=0).any().repeat(2).values

        df = df[mask]

        if args:
            df = df[list(args)]

        return df

    def sortby(
        self,
        columns: Union[str, List[str]],
        df: Optional[pd.DataFrame] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Sort DataFrame by specified column's values.
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
        """Get the shape of DataFrame or specified column(s).
        Args:
            columns (Optional[Union[str, List[str]]): Name of column(s).
        Returns:
            Tuple[int]: Shape.
        """
        if columns:
            return self.df[columns].shape
        return self.df.shape

    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns the first `n` rows of DataFrame.
        Args:
            n (int): Number of rows to select.
        Returns:
            pandas.DataFrame: The first `n` rows of the caller object.
        """
        return self.df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Returns the last `n` rows of DataFrame.
        Args:
            n (int): Number of rows to select.
        Returns:
            pandas.DataFrame: The last `n` rows of the caller object.
        """
        return self.df.tail(n)

    def get_num_ground_truth(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of ground truths.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            int: The number of ground truths.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return len(df_)

    def get_num_estimation(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of estimations.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            int: The number of estimations.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return len(df_)

    def get_status_num(
        self,
        status: Union[str, MatchingStatus],
        df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> int:
        """Returns number of matching status TP/FP/TN/FN.

        Args:
            status (Union[str, MatchingStatus]): Status.
            df (Optional[pandas.DataFrame]): Target DataFrame. Defaults to None.
            **kwargs

        Returns:
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
            raise ValueError(f"Expected status is TP/FP/TN/FN, but got {status}")

    def get_num_tp(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of TP.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            inf: The number of TP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "TP")

    def get_num_fp(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of FP.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FP that their columns are specified value.
        Returns:
            inf: The number of FP.
        """
        df_ = self.get_estimation(df=df, **kwargs)
        return sum(df_["status"] == "FP")

    def get_num_tn(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of TN.
        Args:
            Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of TN that their columns are specified value.
        Returns:
            inf: The number of TN.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return sum(df_["status"] == "TN")

    def get_num_fn(self, df: Optional[pd.DataFrame] = None, **kwargs) -> int:
        """Returns the number of FN.
        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
            **kwargs: Specify if you want to get the number of FN that their columns are specified value.
        Returns:
            inf: The number of FN.
        """
        df_ = self.get_ground_truth(df=df, **kwargs)
        return sum(df_["status"] == "FN")

    def get_ground_truth(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """Returns the DataFrame for ground truth.
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
            elif not isinstance(item, str) and isinstance(item, Iterable):
                df = df[df[key].isin(item)]
            else:
                df = df[df[key] == item]
        return df

    def get_estimation(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """Returns the DataFrame for estimation.
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
            elif not isinstance(item, str) and isinstance(item, Iterable):
                df = df[df[key].isin(item)]
            else:
                df = df[df[key] == item]

        return df

    def get_pair_results(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns paired results, which means both the row of GT and estimation valid values.

        Args:
            df (Optional[pandas.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
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

    def get_scenes(self, df: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """Returns numpy array of unique scenes.
        Args:
            df (optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.
        Returns:
            numpy.ndarray
        """
        if df is None:
            df = self.get(**kwargs)

        scenes: np.ndarray = pd.unique(df["scene"])
        return scenes[~np.isnan(scenes)]

    def get_ego2map(self, scene: int, frame: int) -> np.ndarray:
        """Returns 4x4 ego2map transform matrix.
        Args:
            scene (int): Number of scene.
            frame (int): Number of frame.
        Returns:
            np.ndarray: 4x4 transform matrix.
        """
        transforms = self.get_transforms(scene, frame)
        return transforms[(FrameID.BASE_LINK, FrameID.MAP)].matrix

    def get_transforms(self, scene: int, frame: int) -> TransformDict:
        """Return transform matrix container at the specified scene and frame.
        Args:
            scene (int): Number of scene.
            frame (int): Number of frame.
        Returns:
            TransformDict: Transform matrix container.
        """
        return self.__transforms[str(scene)][str(frame)]

    def __len__(self) -> int:
        return len(self.df)

    def get_metrics_score(self, frame_results: List[PerceptionFrameResult]) -> MetricsScore:
        """Returns the metrics score for each evaluator

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            metrics_score (MetricsScore): The final metrics score.
        """
        target_labels: List[LabelType] = self.config.target_labels
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
        if self.config.metrics_config.detection_config is not None:
            metrics_score.evaluate_detection(scene_results, scene_num_gt)
        if self.config.metrics_config.tracking_config is not None:
            metrics_score.evaluate_tracking(scene_results, scene_num_gt)
        if self.config.metrics_config.prediction_config is not None:
            pass
        if self.config.metrics_config.classification_config is not None:
            metrics_score.evaluate_classification(scene_results, scene_num_gt)

        return metrics_score

    @abstractmethod
    def analyze(self, *args, **kwargs) -> PerceptionAnalysisResult:
        """Analyze TP/FP/FN ratio, metrics score, error. If there is no DataFrame to be able to analyze returns None.

        Args:
            **kwargs: Specify scene, frame, area or uuid.

        Returns:
            PerceptionAnalysisResult:
                score (Optional[pandas.DataFrame]): DataFrame of TP/FP/FN ratios and metrics scores.
                error (Optional[pandas.DataFrame]): DataFrame of errors.
                confusion_matrix (Optional[pandas.DataFrame]): DataFrame of the confusion matrix.
        """
        pass

    def add_frame(self, frame: PerceptionFrameResult) -> pd.DataFrame:
        """Add single frame result and update DataFrame.

        Args:
            frame (PerceptionFrameResult): Single frame result.

        Returns:
            pd.DataFrame:
        """
        start = len(self.df) // 2
        concat: List[pd.DataFrame] = []
        if len(self) > 0:
            concat.append(self.df)
        self.__transforms[str(self.num_scene)][str(frame.frame_name)] = frame.frame_ground_truth.transforms

        tp_df = self.format2df(
            frame.pass_fail_result.tp_object_results,
            status=MatchingStatus.TP,
            start=start,
            frame_num=int(frame.frame_name),
            transforms=frame.frame_ground_truth.transforms,
        )
        if len(tp_df) > 0:
            start += len(tp_df) // 2
            concat.append(tp_df)

        fp_df = self.format2df(
            frame.pass_fail_result.fp_object_results,
            status=MatchingStatus.FP,
            start=start,
            frame_num=int(frame.frame_name),
            transforms=frame.frame_ground_truth.transforms,
        )
        if len(fp_df) > 0:
            start += len(fp_df) // 2
            concat.append(fp_df)

        tn_df = self.format2df(
            frame.pass_fail_result.tn_objects,
            status=MatchingStatus.TN,
            start=start,
            frame_num=int(frame.frame_name),
            transforms=frame.frame_ground_truth.transforms,
        )
        if len(tn_df) > 0:
            start += len(tn_df) // 2
            concat.append(tn_df)

        fn_df = self.format2df(
            frame.pass_fail_result.fn_objects,
            status=MatchingStatus.FN,
            start=start,
            frame_num=int(frame.frame_name),
            transforms=frame.frame_ground_truth.transforms,
        )
        if len(fn_df) > 0:
            start += len(fn_df) // 2
            concat.append(fn_df)

        if len(concat) > 0:
            self.__df = pd.concat(concat)

        return self.__df

    def add(self, frame_results: List[PerceptionFrameResult]) -> pd.DataFrame:
        """Add frame results and update DataFrame.

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            pandas.DataFrame
        """
        self.__frame_results[self.num_scene] = frame_results
        self.__num_frame += len(frame_results)

        self.__transforms[str(self.num_scene)] = {}
        for frame in tqdm(frame_results, "Updating DataFrame"):
            self.add_frame(frame)
        self.__num_scene += 1

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
        """Clear frame results and DataFrame."""
        self.__frame_results.clear()
        del self.__df
        self.__initialize()

    def format2df(
        self,
        object_results: List[Union[DynamicObject, DynamicObjectWithPerceptionResult]],
        status: MatchingStatus,
        frame_num: int,
        start: int = 0,
        transforms: Optional[TransformDict] = None,
    ) -> pd.DataFrame:
        """Format objects to pandas.DataFrame.

        Args:
            object_results (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]): List of objects or object results.
            status (MatchingStatus): Object's status.
            frame_num (int): Number of frame.
            start (int): Number of the first index. Defaults to 0.

        Returns:
            df (pandas.DataFrame)
        """
        rets: Dict[int, Dict[str, Any]] = {}
        for i, obj_result in enumerate(object_results, start=start):
            rets[i] = self.format2dict(obj_result, status, frame_num, transforms)

        df = pd.DataFrame.from_dict(
            {(i, j): rets[i][j] for i in rets.keys() for j in rets[i].keys()},
            orient="index",
            columns=self.keys(),
        )
        return df

    @abstractmethod
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
        pass

    def calculate_error(
        self,
        column: Union[str, List[str]],
        df: Optional[pd.DataFrame] = None,
        remove_nan: bool = False,
    ) -> np.ndarray:
        """Calculate specified column's error for TP/TN and matched FP.

        Args:
            column (Union[str, List[str]]): name of column
            df (pandas.DataFrame): Specify if you want use filtered DataFrame. Defaults to None.
            remove_nan (bool): Whether remove nan value. Defaults to False.

        Returns:
            np.ndarray: Array of error, in shape (N, M).
                N is number of TP, M is dimensions.
        """
        expects: Set[str] = set(self.state_columns)
        if isinstance(column, str):
            column = [column]
        keys: Set[str] = set(column)

        if keys > expects:
            raise ValueError(f"Unexpected keys: {column}, expected: {expects}")

        if df is None:
            df = self.df

        df_ = df[df["status"].isin(["TP", "FP", "TN"])]
        errors = []
        for col in column:
            if col == "distance":
                df_arr = np.array(df_[["x", "y"]])  # (N, 2)
            elif col == "nn_plane":
                df_arr = np.stack(
                    (
                        df_["nn_point1"].tolist(),
                        df_["nn_point2"].tolist(),
                    ),
                    axis=1,
                )  # (N, 2, 3)
            else:
                df_arr = np.array(df_[col])
            gt_vals = df_arr[::2]
            est_vals = df_arr[1::2]
            err: np.ndarray = gt_vals - est_vals
            if remove_nan:
                err = err[~np.isnan(err)]

            if col == "yaw":
                # Clip err from [-2pi, 2pi] to [-pi, pi]
                err[err > np.pi] = -2 * np.pi + err[err > np.pi]
                err[err < -np.pi] = 2 * np.pi + err[err < -np.pi]
            elif col == "distance":
                err = err.reshape(-1, 2)
                err = np.linalg.norm(err, axis=1)
            elif col == "nn_plane":
                err = err.reshape(-1, 2, 3)
                err = np.linalg.norm(err, axis=2)
                err = np.mean(err, axis=1)
            errors.append(err)

        if len(column) == 1:
            errors = np.array(errors).reshape(-1)
        else:
            errors = np.stack(errors, axis=1)

        return errors

    @abstractmethod
    def summarize_error(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate mean, sigma, RMS, max and min of error.

        Args:
            df (Optional[pd.DataFrame]): Specify if you want use filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame
        """
        pass

    def summarize_ratio(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Summarize TP/FP/FN ratio.

        Args:
            df (Optional[pandas.DataFrame]): Specify, if you want to use any filtered DataFrame. Defaults to None.

        Returns:
            pd.DataFrame
        """
        if df is None:
            df = self.df

        data: Dict[str, List[float]] = {str(s): [0.0] * len(self.all_labels) for s in MatchingStatus}
        for i, label in enumerate(self.all_labels):
            if label == "ALL":
                label = None
            num_ground_truth: int = self.get_num_ground_truth(df=df, label=label)
            if num_ground_truth > 0:
                data["TP"][i] = self.get_num_tp(df=df, label=label) / num_ground_truth
                data["FP"][i] = self.get_num_fp(df=df, label=label) / num_ground_truth
                data["TN"][i] = self.get_num_tn(df=df, label=label) / num_ground_truth
                data["FN"][i] = self.get_num_fn(df=df, label=label) / num_ground_truth
        return pd.DataFrame(data, index=self.all_labels)

    def summarize_score(self, scene: Optional[Union[int, List[int]]] = None, *args, **kwargs) -> pd.DataFrame:
        """Summarize MetricsScore.

        Args:
            scene (Optional[int]): Number of scene. If it is not specified, calculate metrics score for all scenes.
                Defaults to None.

        Returns:
            pandas.DataFrame
        """
        if scene is None:
            frame_results = [x for v in self.frame_results.values() for x in v]
        else:
            scene: List[int] = [scene] if isinstance(scene, int) else scene
            frame_results = [x for k, v in self.frame_results.items() if k in scene for x in v]

        metrics_score = self.get_metrics_score(frame_results)
        data: Dict[str, Any] = get_metrics_info(metrics_score)

        return pd.DataFrame(data, index=self.all_labels)

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
        mode: PlotAxes = PlotAxes.DISTANCE,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        heatmap: bool = False,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot the number of objects for each time/distance range with histogram.

        Args:
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.DISTANCE (1-dimensional).
            bins (Optional[Union[int, Tuple[int, int]]]): The interval of time/distance.
                If not specified, 0.1[s] for time and 0.5[m] for distance will be use. Defaults to None.
            heatmap (bool): Whether to visualize heatmap of the number of objects for corresponding axes.
                The heatmap can be visualized only 3D axes. Defaults to False.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if len(kwargs) == 0:
            title = "Number of Objects @all"
            filename = "all"
        else:
            title: str = "Num Object "
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
        if mode.is_3d() and heatmap:
            filename += "_heatmap"
            ax = fig.subplots(nrows=1, ncols=2)
        else:
            ax: Union[Axes, Axes3D] = fig.add_subplot(
                xlabel=xlabel,
                ylabel=ylabel,
                title="Number of samples",
                projection=mode.projection if not heatmap else None,
            )
        mode.setup_axis(ax, **kwargs)
        gt_axes = mode.get_axes(self.get_ground_truth(**kwargs))
        est_axes = mode.get_axes(self.get_estimation(**kwargs))

        if bins is None:
            bins = mode.get_bins()

        if mode.is_2d():
            assert isinstance(bins, Number), f"For 2D plot, bins must be number but got {type(bins)}"
            min_value = 0 if mode == PlotAxes.CONFIDENCE else _get_min_value(gt_axes, est_axes)
            max_value = 100 if mode == PlotAxes.CONFIDENCE else _get_max_value(gt_axes, est_axes)
            hist_bins = np.arange(min_value, max_value + bins, bins)
            gt_hist, xaxis = np.histogram(gt_axes, bins=hist_bins)
            est_hist, _ = np.histogram(est_axes, bins=hist_bins)
            width = bins if mode == PlotAxes.CONFIDENCE else 0.25 * ((max_value - min_value) / bins)
            ax.bar(xaxis[:-1] - 0.5 * width, gt_hist, width, label="GT")
            ax.bar(xaxis[:-1] + 0.5 * width, est_hist, width, label="Estimation")
        else:
            gt_xaxes, gt_yaxes = gt_axes[:, ~np.isnan(gt_axes).any(0)]
            est_xaxes, est_yaxes = est_axes[:, ~np.isnan(est_axes).any(0)]

            bins = int(bins) if isinstance(bins, Number) else (int(bins[0]), int(bins[1]))
            gt_hist, gt_x_edges, gt_y_edges = np.histogram2d(gt_xaxes, gt_yaxes, bins=bins)
            est_hist, est_x_edges, est_y_edges = np.histogram2d(est_xaxes, est_yaxes, bins=bins)
            if heatmap:
                gt_extent = [gt_x_edges[0], gt_x_edges[-1], gt_y_edges[0], gt_y_edges[-1]]
                est_extent = [est_x_edges[0], est_x_edges[-1], est_y_edges[0], est_y_edges[-1]]

                ax[0].set_title("GT")
                ax[1].set_title("Estimation")
                gt_img = ax[0].imshow(gt_hist, extent=gt_extent)
                fig.colorbar(gt_img, ax=ax[0])
                est_img = ax[1].imshow(est_hist, extent=est_extent)
                fig.colorbar(est_img, ax=ax[1])
            else:
                ax.set_zlabel("Number of samples")
                gt_x, gt_y = np.meshgrid(gt_x_edges[:-1], gt_y_edges[:-1])
                est_x, est_y = np.meshgrid(est_x_edges[:-1], est_y_edges[:-1])
                dx, dy = (bins, bins) if isinstance(bins, Number) else bins
                dx *= 0.5
                dy *= 0.5
                ax.bar3d(gt_x.ravel(), gt_y.ravel(), 0, dx, dy, gt_hist.ravel(), alpha=0.6)
                ax.bar3d(est_x.ravel(), est_y.ravel(), 0, dx, dy, est_hist.ravel(), alpha=0.6)

        self.__post_process_figure(
            fig=fig,
            title=title,
            legend=True,
            filename=f"num_object_{str(mode)}_{filename}",
            show=show,
        )

    def plot_state(
        self,
        uuid: str,
        columns: Union[str, List[str]],
        mode: PlotAxes = PlotAxes.TIME,
        status: Optional[MatchingStatus] = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Plot states for each time/distance estimated and GT object in TP.

        Args:
            uuid (str): Target object's uuid.
            columns (Union[str, List[str]]): Target column name.
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            status (Optional[int]): Target status TP/FP/FN. If not specified, plot all status. Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

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
            filename=f"state_{columns_str}_{uuid}_{str(mode)}",
            show=show,
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
        """Plot error between estimated and GT object in TP/TN and matched FP.

        Args:
            columns (Union[str, List[str]]): Target column name.
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            heatmap (bool): Whether overlay heatmap. Defaults to False.
            project (bool): Whether to project heatmap on 2D. This argument is only used for 3D heatmap plot.
                Defaults to False.
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (int): Bin size to plot heatmap. Defaults to 50.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        gt_df = self.get_ground_truth(status=["TP", "FP", "TN"], **kwargs)
        index = pd.unique(gt_df.index.get_level_values(level=0))

        if len(index) == 0:
            logging.warning("There is no TP/FP/TN object. Could not calculate error.")
            return

        project *= heatmap  # if heatmap=False, always project=False

        tp_df = self.df.loc[index]

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
            ax: Union[Axes, Axes3D] = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                projection=None if project else mode.projection,
            )

            mode.setup_axis(ax, **kwargs)
            err: np.ndarray = self.calculate_error(col, df=tp_df)
            axes: np.ndarray = mode.get_axes(gt_df)
            if mode.is_2d():
                non_nan = ~np.isnan(err) * ~np.isnan(axes)
                axes = axes[non_nan]
                err = err[non_nan]
                if heatmap:
                    ax.hist2d(axes, err, bins=(bins, bins), cmap=cm.jet)
                else:
                    ax.scatter(axes, err)
            else:
                if not project:
                    ax.set_zlabel(f"err_{col}")
                non_nan = ~np.isnan(err) * ~np.isnan(axes).any(0)
                xaxes, yaxes = axes[:, non_nan]
                err = err[non_nan]
                if heatmap:
                    if project:
                        # TODO(ktro2828): This is wrong projection
                        hist, x_edges, y_edges = np.histogram2d(xaxes, yaxes, bins=bins, weights=err)
                        ax.pcolormesh(x_edges, y_edges, hist, cmap=cm.jet)
                    else:
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
            filename=f"error_{columns_str}_{str(mode)}",
            show=show,
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
                If you want plot multiple column for one image, use List[str].
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        fig, ax = plt.subplots()
        setup_axis(ax, **kwargs)

        df = self.get(**kwargs)
        errors: List[np.ndarray] = [self.calculate_error(col, df) for col in columns]
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
        status: Union[str, MatchingStatus],
        mode: PlotAxes = PlotAxes.DISTANCE,
        show: bool = False,
        bins: Optional[float] = None,
        plot_range: Optional[Tuple[float]] = None,
        **kwargs,
    ) -> None:
        """Plot TP/FP/TN/FN ratio.

        Args:
            status (Union[str, MatchingStatus]): Matching status, TP/FP/TN/FN.
            mode (PlotAxes): PlotAxes instance.
            show (bool): Whether show the plotted figure. Defaults to False.
            bins (Optional[float]): Plot binds. Defaults to None.
            plot_range (Optional[Tuple[float, float]]): Range of plot. Defaults to None.
        """
        if mode.is_3d():
            raise NotImplementedError("3D plot is under construction.")

        gt_df = self.get_ground_truth(**kwargs)
        est_df = self.get_estimation(**kwargs)
        gt_df = gt_df[gt_df["status"] != np.nan]
        est_df = est_df[est_df["status"] != np.nan]
        gt_values: np.ndarray = mode.get_axes(gt_df)
        est_values: np.ndarray = mode.get_axes(est_df)

        xlabel: str = mode.xlabel
        ylabel: str = f"{str(status)} ratio"
        if plot_range is not None:
            min_value, max_value = plot_range
        else:
            min_value = 0 if mode == PlotAxes.CONFIDENCE else _get_min_value(gt_values, est_values)
            max_value = 100 if mode == PlotAxes.CONFIDENCE else _get_max_value(gt_values, est_values)
        step = bins if bins else mode.get_bins()
        hist_bins = np.arange(min_value, max_value + step, step)
        _, axis = np.histogram(est_values, bins=hist_bins)

        fig: Figure = plt.figure(figsize=(16, 8))
        ax: Union[Axes, Axes3D] = fig.add_subplot(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{str(status)} ratio",
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
                        raise ValueError(f"Unexpected status: {status}")
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
            title=f"{str(status)} ratio",
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
        else:
            plt.close()


def _get_min_value(value1: np.ndarray, value2: np.ndarray) -> float:
    return min(value1[~np.isnan(value1)].min(), value2[~np.isnan(value2)].min())


def _get_max_value(value1: np.ndarray, value2: np.ndarray) -> float:
    return max(value1[~np.isnan(value1)].max(), value2[~np.isnan(value2)].max())
