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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import pandas as pd
from perception_eval.common.object import DynamicObject
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.util.math import get_pose_transform_matrix
from perception_eval.util.math import rotation_matrix_to_euler
import yaml

from .perception_analyzer_base import PerceptionAnalyzerBase
from .utils import extract_area_results
from .utils import generate_area_points
from .utils import get_area_idx
from .utils import get_metrics_info
from .utils import MatchingStatus
from .utils import PlotAxes
from .utils import setup_axis

# TODO: Refactor plot methods


class PerceptionAnalyzer3D(PerceptionAnalyzerBase):
    """An analyzer class for 3D perception evaluation results.

    Attributes:
        config (PerceptionEvaluationConfig): Configurations for evaluation parameters.
        target_labels (List[str]): Target labels list. (e.g. ["car", "pedestrian", "motorbike"]).
        all_labels (List[str]): Target labels list including "ALL". (e.g. ["ALL", "car", "pedestrian", "motorbike"]).
        num_area_division (int): Number
        upper_rights (numpy.ndarray)
        bottom_lefts (numpy.ndarray)
        columns (List[str])
        state_columns (List[str])
        df (pandas.DataFrame)
        plot_directory (str)
        frame_results (Dict[str, List[PerceptionFrameResult]])
        num_frame (int)
        num_scene (int)
        num_ground_truth (int)
        num_estimation (int)
        num_tp (int)
        num_fp (int)
        num_fn (int)

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
        eval_task_: str = eval_cfg_dict["evaluation_task"]
        if eval_task_ == "detection":
            frame_id = "base_link"
        elif eval_task_ in ("tracking", "prediction"):
            frame_id = "map"
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

        return cls(evaluation_config, num_area_division)

    @property
    def columns(self) -> List[str]:
        return [
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
            "confidence",
            "uuid",
            "num_points",
            "status",
            "area",
            "frame",
            "scene",
        ]

    @property
    def state_columns(self) -> List[str]:
        return [
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
            gt, estimation = None, None
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
            if gt.state.velocity:
                gt_vx, gt_vy = gt.state.velocity[:2]
                gt_vel = np.array([gt_vx, gt_vy, 0])
            else:
                gt_vx, gt_vy = None, None

            if self.config.frame_id == "map":
                src: np.ndarray = get_pose_transform_matrix(
                    position=gt.state.position,
                    rotation=gt.state.orientation.rotation_matrix,
                )
                dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
                gt_x, gt_y = dst[:2, 3]
                gt_yaw = rotation_matrix_to_euler(dst[:3, :3])[-1].item()
                if gt.state.velocity:
                    gt_vx, gt_vy = np.linalg.inv(ego2map[:3, :3]).dot(gt_vel)[:2]
            else:
                gt_x, gt_y = gt.state.position[:2]
                gt_yaw = gt.state.orientation.yaw_pitch_roll[0]

            gt_w, gt_l, gt_h = gt.state.size

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
                confidence=gt.semantic_score,
                uuid=gt.uuid,
                num_points=gt.pointcloud_num,
                status=status,
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            gt_ret = {k: None for k in self.keys()}

        if estimation:
            if estimation.state.velocity:
                est_vx, est_vy = estimation.state.velocity[:2]
                est_vel = np.array([est_vx, est_vy, 0.0])
            else:
                est_vx, est_vy = None, None

            if self.config.frame_id == "map":
                src: np.ndarray = get_pose_transform_matrix(
                    position=estimation.state.position,
                    rotation=estimation.state.orientation.rotation_matrix,
                )
                dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
                est_x, est_y = dst[:2, 3]
                est_yaw = rotation_matrix_to_euler(dst[:3, :3])[-1].item()
                if estimation.state.velocity:
                    est_vx, est_vy = dst[:3, :3].dot(est_vel)[:2]
            else:
                est_x, est_y = estimation.state.position[:2]
                est_yaw = estimation.state.orientation.yaw_pitch_roll[0]
                if estimation.state.velocity:
                    est_rot = estimation.state.orientation.rotation_matrix
                    est_vx, est_vy = np.linalg.inv(est_rot).dot(est_vel)[:2]

            est_w, est_l, est_h = estimation.state.size

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
                confidence=estimation.semantic_score,
                uuid=estimation.uuid,
                num_points=None,
                status=status,
                area=area,
                frame=frame_num,
                scene=self.num_scene,
            )
        else:
            est_ret = {k: None for k in self.keys()}

        return {"ground_truth": gt_ret, "estimation": est_ret}

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

        all_data = {}
        for label in self.__all_labels:
            data = {}
            df_ = df if label == "ALL" else df[df["label"] == label]

            data["x"] = _summarize("x", df_)
            data["y"] = _summarize("y", df_)
            data["yaw"] = _summarize("yaw", df_)
            data["length"] = _summarize("l", df_)
            data["width"] = _summarize("w", df_)
            data["vx"] = _summarize("vx", df_)
            data["vy"] = _summarize("vy", df_)
            data["nn_plane"] = _summarize(["nn_point1", "nn_point2"], df_)
            all_data[str(label)] = data

        ret_df = pd.DataFrame.from_dict(
            {(i, j): all_data[i][j] for i in all_data.keys() for j in all_data[i].keys()},
            orient="index",
        )

        return ret_df

    def summarize_score(
        self,
        scene: Optional[Union[int, List[int]]] = None,
        area: Optional[int] = None,
    ) -> pd.DataFrame:
        """[summary]
        Summarize MetricsScore.

        Args:
            area (Optional[int]): Number of area. If it is not specified, calculate metrics score for all areas.
                Defaults to None.
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

        if area is not None:
            frame_results = extract_area_results(
                frame_results,
                area=area,
                upper_rights=self.upper_rights,
                bottom_lefts=self.bottom_lefts,
            )

        metrics_score = self.get_metrics_score(frame_results)
        data: Dict[str, Any] = get_metrics_info(metrics_score)

        return pd.DataFrame(data, index=self.__all_labels)

    def plot_num_object(
        self,
        mode: PlotAxes = PlotAxes.DISTANCE,
        show: bool = False,
        bin: Optional[float] = None,
        **kwargs,
    ) -> None:
        """[summary]
        Plot the number of objects for each time/distance range with histogram.

        Args:
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.DISTANCE (1-dimensional).
            show (bool): Whether show the plotted figure. Defaults to False.
            bin (float): The interval of time/distance. If not specified, 0.1[s] for time and 0.5[m] for distance will be use.
                Defaults to None.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """

        def _get_min_value(value1: np.ndarray, value2: np.ndarray) -> float:
            return min(value1[~np.isnan(value1)].min(), value2[~np.isnan(value2)].min())

        def _get_max_value(value1: np.ndarray, value2: np.ndarray) -> float:
            return max(value1[~np.isnan(value1)].max(), value2[~np.isnan(value2)].max())

        if len(kwargs) == 0:
            title = "Num Object @all"
            filename = "all"
        else:
            title: str = "Num Object "
            filename: str = ""
            for key, item in kwargs.items():
                title += f"@{key.upper()}:{item} "
                filename += f"{item}_"
            title = title.rstrip(" ")
            filename = filename.rstrip("_")

        gt_axes = mode.get_axes(self.get_ground_truth(**kwargs))
        est_axes = mode.get_axes(self.get_estimation(**kwargs))

        if mode.is_2d():
            xlabel: str = mode.xlabel
            ylabel: str = "num"
        else:
            xlabel, ylabel = mode.get_label()

        # TODO: Arrange to single figure
        fig: Figure = plt.figure(figsize=(16, 8))
        ax1: Union[Axes, Axes3D] = fig.add_subplot(
            1,
            2,
            1,
            xlabel=xlabel,
            ylabel=ylabel,
            title="GT",
            projection=mode.projection,
        )
        ax2: Union[Axes, Axes3D] = fig.add_subplot(
            1,
            2,
            2,
            xlabel=xlabel,
            ylabel=ylabel,
            title="Estimation",
            projection=mode.projection,
        )

        setup_axis(ax1, **kwargs)
        setup_axis(ax2, **kwargs)

        if mode.is_2d():
            min_value = _get_min_value(gt_axes, est_axes)
            max_value = _get_max_value(gt_axes, est_axes)
            step = bin if bin else mode.get_bin()
            bins = np.arange(min_value, max_value, step=step)
            ax1.hist(gt_axes, bins=bins)
            ax2.hist(est_axes, bins=bins)
        else:
            ax1.set_zlabel("num")
            ax2.set_zlabel("num")
            gt_xaxes, gt_yaxes = gt_axes[:, ~np.isnan(gt_axes).any(0)]
            est_xaxes, est_yaxes = est_axes[:, ~np.isnan(est_axes).any(0)]
            gt_hist, gt_x_edges, gt_y_edges = np.histogram2d(gt_xaxes, gt_yaxes)
            est_hist, est_x_edges, est_y_edges = np.histogram2d(est_xaxes, est_yaxes)
            gt_x, gt_y = np.meshgrid(gt_x_edges[:-1], gt_y_edges[:-1])
            est_x, est_y = np.meshgrid(est_x_edges[:-1], est_y_edges[:-1])
            if bin is None:
                dx, dy = mode.get_bin()
            else:
                if isinstance(bin, float):
                    bin = (bin, bin)
                if not isinstance(bin, (list, tuple)) or len(bin) != 2:
                    raise RuntimeError(f"bin for 3D plot must be 2-length, but got {bin}")
                dx, dy = bin
            ax1.bar3d(gt_x.ravel(), gt_y.ravel(), 0, dx, dy, gt_hist.ravel())
            ax2.bar3d(est_x.ravel(), est_y.ravel(), 0, dx, dy, est_hist.ravel())

        plt.suptitle(f"{title}")
        plt.savefig(os.path.join(self.plot_directory, f"num_object_{str(mode)}_{filename}.png"))
        if show:
            plt.show()
        plt.close()

    def plot_state(
        self,
        uuid: str,
        columns: Union[str, List[str]],
        mode: PlotAxes = PlotAxes.TIME,
        status: Optional[MatchingStatus] = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """[summary]
        Plot states for each time/distance estimated and GT object in TP.

        Args:
            uuid (str): Target object's uuid.
            columns (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            status (Optional[int]): Target status TP/FP/FN. If not specified, plot all status. Defaults to None.
            show (bool): Whether show the plotted figure. Defaults to False.
            **kwargs
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "w", "l", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

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
            setup_axis(ax, **kwargs)
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

        plt.suptitle(f"State of {columns} @uuid:{uuid}")
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
        mode: PlotAxes = PlotAxes.TIME,
        heatmap: bool = False,
        show: bool = False,
        bin: int = 50,
        **kwargs,
    ) -> None:
        """[summary]
        Plot states for each time/distance estimated and GT object in TP.

        Args:
            columns (Union[str, List[str]]): Target column name. Options: ["x", "y", "yaw", "w", "l", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            mode (PlotAxes): Mode of plot axis. Defaults to PlotAxes.TIME (1-dimensional).
            heatmap (bool): Whether overlay heatmap. Defaults to False.
            show (bool): Whether show the plotted figure. Defaults to False.
            bin (int): Bin size to plot heatmap. Defaults to 50.
            **kwargs: Specify if you want to plot for the specific conditions.
                For example, label, area, frame or scene.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "w", "l", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

        tp_gt_df = self.get_ground_truth(status="TP", **kwargs)
        tp_index = pd.unique(tp_gt_df.index.get_level_values(level=0))

        if len(tp_index) == 0:
            logging.warning("There is no TP object")
            return

        tp_df = self.df.loc[tp_index]

        num_cols = len(columns)
        fig: Figure = plt.figure(figsize=(8 * num_cols, 8))
        for n, col in enumerate(columns):
            if mode.is_2d():
                xlabel: str = mode.xlabel
                ylabel: str = f"err_{col}"
                title: str = f"Error {ylabel} = F({xlabel})"
            else:
                xlabel, ylabel = mode.get_label()
                title: str = f"Error {col} = F({xlabel}, {ylabel})"
            ax: Union[Axes, Axes3D] = fig.add_subplot(
                1,
                num_cols,
                n + 1,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                projection=mode.projection,
            )

            setup_axis(ax, **kwargs)
            err: np.ndarray = self.calculate_error(col, df=tp_df)
            axes: np.ndarray = mode.get_axes(tp_gt_df)
            if mode.is_2d():
                non_nan = ~np.isnan(err) * ~np.isnan(axes)
                axes = axes[non_nan]
                err = err[non_nan]
                if heatmap:
                    ax.hist2d(axes, err, bins=(bin, bin), cmap=cm.jet)
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

        plt.suptitle(f"Error of {columns}")
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
        **kwargs,
    ) -> None:
        """[summary]
        Plot box-plot of errors.

        Args:
            column (Union[str, List[str]]): Target column name.
                Options: ["x", "y", "yaw", "w", "l", "vx", "vy"].
                If you want plot multiple column for one image, use List[str].
            show (bool): Whether show the plotted figure. Defaults to False.
        """
        if isinstance(columns, str):
            columns: List[str] = [columns]

        if set(columns) > set(["x", "y", "yaw", "w", "l", "vx", "vy"]):
            raise ValueError(f"{columns} is unsupported for plot")

        _, ax = plt.subplots()
        setup_axis(ax, **kwargs)

        df = self.get(**kwargs)
        errs: List[np.ndarray] = []
        for col in columns:
            errs.append(self.calculate_error(col, df))
        ax.boxplot(errs)
        ax.set_xticklabels(columns)

        plt.suptitle("Box-Plot of Errors")
        plt.tight_layout()
        columns_str: str = "".join(columns)
        plt.savefig(os.path.join(self.plot_directory, f"box_plot_{columns_str}.png"))
        if show:
            plt.show()
        plt.close()
