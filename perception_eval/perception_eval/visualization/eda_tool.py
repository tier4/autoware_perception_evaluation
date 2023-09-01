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

from logging import getLogger
import os
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pdp
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_tp_fp_objects
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import get_fn_objects
from plotly import graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

logger = getLogger(__name__)


def without_result_objects_to_df(objects: List[DynamicObject]) -> pd.DataFrame:
    names = np.stack([gt_object.semantic_label.label.value for gt_object in objects])
    xyz = np.stack([gt_object.state.position for gt_object in objects])
    wlh = np.stack([gt_object.state.size for gt_object in objects])
    pcd_nums = np.stack([gt_object.pointcloud_num for gt_object in objects])

    df: pd.DataFrame = pd.DataFrame(
        dict(
            name=names,
            w=wlh[:, 0],
            l=wlh[:, 1],
            h=wlh[:, 2],
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            num_points=pcd_nums,
        )
    )
    df["distance_2d"] = np.sqrt((df["x"]) ** 2 + (df["y"]) ** 2)
    return df


def with_result_objects_to_df(objects: List[DynamicObjectWithPerceptionResult]) -> pd.DataFrame:
    names = np.stack(
        [
            estimated_object.estimated_object.semantic_label.label.value
            for estimated_object in objects
        ]
    )
    xyz = np.stack(
        [estimated_object.estimated_object.state.position for estimated_object in objects]
    )
    wlh = np.stack([estimated_object.estimated_object.state.size for estimated_object in objects])
    # todo: pcd_nums for estimated objects is always none (?)
    pcd_nums = np.stack(
        [estimated_object.estimated_object.pointcloud_num for estimated_object in objects]
    )
    center_distances = np.stack(
        [estimated_object.center_distance.value for estimated_object in objects]
    )
    confidences = np.stack(
        [estimated_object.estimated_object.semantic_score for estimated_object in objects]
    )
    gt_ids = [
        estimated_object.ground_truth_object.uuid
        if estimated_object.ground_truth_object is not None
        else None
        for estimated_object in objects
    ]

    df: pd.DataFrame = pd.DataFrame(
        dict(
            name=names,
            w=wlh[:, 0],
            l=wlh[:, 1],
            h=wlh[:, 2],
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            num_points=pcd_nums,
            center_distance=center_distances,
            confidence=confidences,
            gt_id=gt_ids,
        )
    )
    df["distance_2d"] = np.sqrt((df["x"]) ** 2 + (df["y"]) ** 2)
    return df


def with_result(objects):
    # todo maybe assert all objects are same type?
    if hasattr(objects[0], "estimated_object"):
        return True
    else:
        return False


class EDAVisualizer:
    """[summary]
    Visualization class for EDA

    Attributes:
        self.visualize_df (pd.DataFrame): pd.DataFrame converted from objects.
        self.save_dir (str): save directory for each graph.
        self.is_gt (bool): Ground truth objects or not.
    """

    def __init__(
        self,
        objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]],
        save_dir: str,
        show: bool = False,
        visualized_results_name: str = "some kind of results, e.g. false negatives",
        objects_source_name: str = "results from some model",
    ) -> None:
        """[summary]

        Args:
            objects (Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]):
                    estimated objects(List[DynamicObject]) or ground truth objects(List[DynamicObjectWithPerceptionResult]]) which you want to visualize
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
            show (bool): Whether show visualized figures. Defaults to False.
        """
        self.is_gt: bool = not with_result(objects)
        self.visualize_df: pd.DataFrame = self.objects_to_df(objects)
        self.save_dir: str = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.show: show = show
        self.visualized_results_name = visualized_results_name
        self.objects_source_name = objects_source_name

    # todo think bout class_names as class attribute vs param to vis funcs
    def visualize(self, class_names, ranges_xy, xylim_dict, width_lim_dict, length_lim_dict):
        # self.hist_object_count_for_each_distance(class_names, ranges_xy=ranges_xy)
        # self.hist_object_count_for_each_distance_acc(class_names, ranges_xy=ranges_xy)
        # self.hist_object_dist2d_for_each_class(class_names)
        # self.hist2d_object_wl_for_each_class(
        #     class_names,
        #     width_lim_dict=width_lim_dict,
        #     length_lim_dict=length_lim_dict,
        # )
        self.hist2d_object_center_xy_for_each_class(
            class_names, xlim_dict=xylim_dict, ylim_dict=xylim_dict
        )
        # if self.is_gt:
        #     self.hist2d_object_num_points_for_each_class(class_names)

        # visualizer.get_pandas_profiling(class_names, "profiling_" + objects_name)

    def objects_to_df(
        self, objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]
    ) -> pd.DataFrame:
        """[summary]
        Convert List[DynamicObject] or List[DynamicObjectWithPerceptionResult]] to pd.DataFrame.

        Args:
            objects (Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]):
                    estimated objects(List[DynamicObject]) or ground truth objects(List[DynamicObjectWithPerceptionResult]]) which you want to visualize

        Returns:
            df (pd.DataFrame):
                    converted pd.DataFrame from objects
        """
        return (
            without_result_objects_to_df(objects)
            if self.is_gt
            else with_result_objects_to_df(objects)
        )

    def get_subplots(self, class_names: List[str]) -> None:
        """[summary]
        Get subplots

        Args:
            class_names (List[str]):
                    names of class you want to visualize.

        Return:
            axes (numpy.ndarray):
                    axes of subplots
        """
        col_size = len(class_names)
        fig, axes = plt.subplots(col_size, 1, figsize=(16, 6 * col_size))
        axes = axes.flatten()
        return axes

    def hist_object_count_for_each_distance_acc(
        self, class_names: List[str], ranges_xy: List[Union[int, float]]
    ) -> None:
        """[summary]
        Show histogram of number of objects that are less than the certain distance in x-y plane.
        Distance is specified by ranges_xy.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            ranges_xy (List[Union[int, float]]):
                    distances in x-y plane.
        """
        subplot_titles: List[str] = []
        for i, range_xy in enumerate(ranges_xy):
            subplot_titles.append(f"#objects: ~{range_xy}m")

        fig: Figure = make_subplots(rows=1, cols=len(ranges_xy), subplot_titles=subplot_titles)

        visualize_df = self.visualize_df[self.visualize_df.name.isin(class_names)]
        for i, range_xy in enumerate(ranges_xy):
            _df: pd.DataFrame = visualize_df[visualize_df.distance_2d < range_xy]
            fig.add_trace(
                go.Histogram(
                    x=_df["name"], name=f"#objects: ~{range_xy}m", marker=dict(color="blue")
                ),
                row=1,
                col=i + 1,
            )

        fig.update_yaxes(range=[0, 8000])  # len(visualize_df)])
        filtered_classes = visualize_df.name.unique().tolist()
        fig.update_xaxes(
            categoryorder="array", categoryarray=sorted(filtered_classes, key=class_names.index)
        )
        if self.show:
            fig.show()

        fig.write_html(self.save_dir + "/hist_object_count_for_each_distance_acc.html")

    def hist_object_count_for_each_distance(
        self, class_names: List[str], ranges_xy: List[Union[int, float]]
    ) -> None:
        """[summary]
        Show histogram of number of objects that are less than the certain distance in x-y plane.
        Distance is specified by ranges_xy.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            ranges_xy (List[Union[int, float]]):
                    distances in x-y plane.
        """
        ranges_xy = ranges_xy + [0]

        subplot_titles: List[str] = []
        for i, _ in enumerate(ranges_xy[:-1]):
            subplot_titles.append(f"#objects: <{ranges_xy[i+1]}m, {ranges_xy[i]}m)")

        fig: Figure = make_subplots(rows=1, cols=len(ranges_xy) - 1, subplot_titles=subplot_titles)

        visualize_df = self.visualize_df[self.visualize_df.name.isin(class_names)]
        max_bar_height = 0
        for i, _ in enumerate(ranges_xy[:-1]):
            _df: pd.DataFrame = visualize_df[
                (visualize_df.distance_2d >= ranges_xy[i + 1])
                & (visualize_df.distance_2d < ranges_xy[i])
            ]
            fig.add_trace(
                go.Histogram(
                    x=_df["name"],
                    name=f"#objects: <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                    marker=dict(color="blue"),
                ),
                row=1,
                col=i + 1,
            )

            curr_max_bar_height = max(_df.name.value_counts()) if len(_df) > 0 else 0

            max_bar_height = (
                curr_max_bar_height if curr_max_bar_height > max_bar_height else max_bar_height
            )

        fig.update_yaxes(range=(0, 1.1 * max_bar_height))

        filtered_classes = visualize_df.name.unique().tolist()
        fig.update_xaxes(
            categoryorder="array", categoryarray=sorted(filtered_classes, key=class_names.index)
        )
        fig.update_layout(title=self.objects_source_name + "    " + self.visualized_results_name)
        if self.show:
            fig.show()

        fig.write_html(self.save_dir + "/hist_object_count_for_each_distance.html")

    def hist_object_dist2d_for_each_class(
        self, class_names: List[str], x_range: List[float] = [-2, 120], y_range: List[float] = None
    ) -> None:
        """[summary]
        Show histogram of distance in x-y plane of objects.
        X axis shows the distance. Y axis shows the count of objects.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            x_range (List[float]):
                    range of distance in x-y plane.
            y_range (List[float]):
                    range of count of objects in that distance.
        """
        subplot_titles: List[str] = []
        for class_name in class_names:
            subplot_titles.append(f"{class_name}")

        fig: Figure = make_subplots(
            rows=1, cols=len(class_names), subplot_titles=subplot_titles, horizontal_spacing=0.1
        )

        for cls_i, class_name in enumerate(class_names):
            _df_cls = self.visualize_df[self.visualize_df.name == class_name]

            dist2d = np.linalg.norm(np.stack((_df_cls.x, _df_cls.y), axis=1), axis=1)

            if y_range is None:
                y_range = [0, len(dist2d) / 40]

            layout = go.Histogram(
                x=dist2d, name=f"{class_name}: #objects={len(_df_cls):,}", nbinsx=400
            )
            fig.add_trace(layout, row=1, col=cls_i + 1)

            fig.layout[f"xaxis{cls_i+1}"].title = "distance(xy plane) [m]"
            fig.layout[f"yaxis{cls_i+1}"].title = "frequency"

        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
        if self.show:
            fig.show()

        fig.write_html(self.save_dir + "/hist_object_dist2d_for_each_class.html")

    def hist2d_object_wl_for_each_class(
        self,
        class_names: List[str],
        width_lim_dict: Dict[str, List[float]] = None,
        length_lim_dict: Dict[str, List[float]] = None,
    ) -> None:
        """[summary]
        Show 2d-histogram of width and length in each class.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            width_lim_dict, length_lim_dict (Dict[str, List[float]]):
                    width_lim, length_lim for each class
                    e.g. width_lim['car'] is [width_min, width_max] for car
        """
        axes = self.get_subplots(class_names)

        for cls_i, class_name in enumerate(class_names):
            _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            if len(_df_cls) > 0:
                w_mean, w_std = _df_cls.w.mean(), _df_cls.w.std()
                l_mean, l_std = _df_cls.l.mean(), _df_cls.l.std()
                h_mean, h_std = _df_cls.h.mean(), _df_cls.h.std()

                hist = axes[cls_i].hist2d(_df_cls.w, _df_cls.l, bins=50, norm=mpl.colors.LogNorm())
                axes[cls_i].plot(
                    w_mean, l_mean, marker="x", color="r", markersize=10, markeredgewidth=3
                )
                axes[cls_i].set_title(
                    f"{class_name}: (w, l, h)=({w_mean:.2f}±{w_std:.2f}, {l_mean:.2f}±{l_std:.2f}, {h_mean:.2f}±{h_std:.2f})"
                )
                axes[cls_i].set_xlabel("width")
                axes[cls_i].set_ylabel("length")
                if width_lim_dict:
                    axes[cls_i].set_xlim(
                        width_lim_dict[class_name][0], width_lim_dict[class_name][1]
                    )
                if length_lim_dict:
                    axes[cls_i].set_ylim(
                        length_lim_dict[class_name][0], length_lim_dict[class_name][1]
                    )
                plt.colorbar(hist[3], ax=axes[cls_i])

        plt.savefig(self.save_dir + "/hist2d_object_wl_for_each_class.svg")

    def hist2d_object_center_xy_for_each_class(
        self,
        class_names: List[str],
        xlim_dict: Dict[str, List[float]] = None,
        ylim_dict: Dict[str, List[float]] = None,
    ) -> None:
        """[summary]
        Show 2d-histogram of x and y in each class.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            xlim_dict, ylim_dict (Dict[str, List[float]]):
                    xlim, ylim for each class
                    e.g. xlim_dict['car'] is [xmin, xmax] for car

        """

        axes = self.get_subplots(class_names)

        for cls_i, class_name in enumerate(class_names):
            _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            if len(_df_cls) > 0:
                x_mean, x_std = _df_cls.x.mean(), _df_cls.x.std()
                y_mean, y_std = _df_cls.y.mean(), _df_cls.y.std()
                z_mean, z_std = _df_cls.z.mean(), _df_cls.z.std()

                hist = axes[cls_i].hist2d(_df_cls.x, _df_cls.y, bins=50, norm=mpl.colors.LogNorm())
                axes[cls_i].plot(
                    x_mean, y_mean, marker="x", color="r", markersize=10, markeredgewidth=3
                )
                axes[cls_i].set_title(
                    f"{class_name}: (x, y, z)=({x_mean:.2f}±{x_std:.2f}, {y_mean:.2f}±{y_std:.2f}, {z_mean:.2f}±{z_std:.2f})"
                )
                axes[cls_i].set_xlabel("x [m] ")
                axes[cls_i].set_ylabel("y [m] ")
                if xlim_dict:
                    axes[cls_i].set_xlim(xlim_dict[class_name][0], xlim_dict[class_name][1])
                if ylim_dict:
                    axes[cls_i].set_ylim(ylim_dict[class_name][0], ylim_dict[class_name][1])
                plt.colorbar(hist[3], ax=axes[cls_i])

        plt.savefig(self.save_dir + "/hist2d_object_center_xy_for_each_class.svg")

    def hist2d_object_num_points_for_each_class(
        self, class_names: List[str], max_pts: int = 500
    ) -> None:
        """[summary]
        Show 2d-histogram of number of point clouds in each class.
        Ground truth objects only have the number of point cloud in bbox, so this method works only for ground truth objects.

        Args:
            class_names (List[str]):
                    names of class you want to visu:alize.
            max_pts (int):
                    max points to visualize.
        """

        if not self.is_gt:
            raise ValueError("You should use this method only for ground truth objects")

        axes = self.get_subplots(class_names)

        for cls_i, class_name in enumerate(class_names):
            _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            if len(_df_cls) > 0:
                num_pts = _df_cls.num_points.copy()
                num_pts[num_pts > max_pts] = max_pts

                dist2d = np.linalg.norm(np.stack((_df_cls.x, _df_cls.y), axis=1), axis=1)
                hist = axes[cls_i].hist2d(dist2d, num_pts, bins=50, norm=mpl.colors.LogNorm())
                axes[cls_i].set_title(f"{class_name}: ")
                axes[cls_i].set_xlabel("dist")
                axes[cls_i].set_ylabel("num_points")
                plt.colorbar(hist[3], ax=axes[cls_i])

        plt.savefig(self.save_dir + "/hist2d_object_num_points_for_each_class.svg")

    def get_pandas_profiling(self, class_names: List[str], file_name: str) -> None:
        """[summary]
        Get pandas profiling report for pd.DataFrame.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            file_name (str):
                    file name for profiling report.
        """
        report = pdp.ProfileReport(self.visualize_df)
        report.to_file(self.save_dir + "/" + file_name + "_all.html")

        for class_name in class_names:
            _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            report = pdp.ProfileReport(_df_cls)
            report.to_file(self.save_dir + "/" + file_name + f"_{class_name}.html")


class EDAResultsComparisonVisualizerDfs:
    def __init__(
        self,
        df_objects_1: pd.DataFrame,
        df_objects_2: pd.DataFrame,
        df_gt: pd.DataFrame,
        save_dir: str,
        show: bool = False,
        objects_1_source_name: str = "results from first model",
        objects_2_source_name: str = "results from second model",
        visualized_results_name: str = "some kind of results, e.g. false negatives",
        show_gt=True,
    ):
        self.objects_1_source_name = objects_1_source_name
        self.objects_2_source_name = objects_2_source_name
        self.visualized_results_name = visualized_results_name
        self.show = show
        self.df_objects_1 = df_objects_1
        self.df_objects_2 = df_objects_2
        self.df_gt = df_gt
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.show_gt = show_gt

    def get_subplots(self, class_names: List[str]) -> None:
        """[summary]
        Get subplots

        Args:
            class_names (List[str]):
                    names of class you want to visualize.

        Return:
            axes (numpy.ndarray):
                    axes of subplots
        """
        col_size = len(class_names)
        fig, axes = plt.subplots(col_size, 1, figsize=(16, 6 * col_size))
        axes = axes.flatten()
        return axes

    def get_subplots_2(self, class_names: List[str], n_cols=1) -> None:
        """[summary]
        Get subplots

        Args:
            class_names (List[str]):
                    names of class you want to visualize.

        Return:
            axes (numpy.ndarray):
                    axes of subplots
        """
        col_size = len(class_names)
        fig, axes = plt.subplots(col_size, n_cols, figsize=(16, 6 * col_size))
        return axes

    # todo:numbers over hists
    def hist_object_count_for_each_distance_comparison(
        self, class_names: List[str], ranges_xy: List[Union[int, float]] = [125, 100, 75, 50, 25]
    ) -> None:
        """[summary]
        Show histogram of number of objects that are less than the certain distance in x-y plane.
        Distance is specified by ranges_xy.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            ranges_xy (List[Union[int, float]]):
                    distances in x-y plane.
        """
        ranges_xy = ranges_xy + [0]

        subplot_titles: List[str] = []
        for i, _ in enumerate(ranges_xy[:-1]):
            subplot_titles.append(f"<{ranges_xy[i+1]}m, {ranges_xy[i]}m)")

        fig: Figure = make_subplots(rows=1, cols=len(ranges_xy) - 1, subplot_titles=subplot_titles)

        visualize_df_1 = self.df_objects_1[self.df_objects_1.semantic_label.isin(class_names)]
        visualize_df_2 = self.df_objects_2[self.df_objects_2.semantic_label.isin(class_names)]
        df_gt = self.df_gt[self.df_gt.semantic_label.isin(class_names)]
        # print(len(visualize_df_1), len(visualize_df_2), len(df_gt))
        max_bar_height = 0
        for i, _ in enumerate(ranges_xy[:-1]):
            visualize_df_filt_1: pd.DataFrame = visualize_df_1[
                (visualize_df_1.distance_2d >= ranges_xy[i + 1])
                & (visualize_df_1.distance_2d < ranges_xy[i])
            ]
            visualize_df_filt_2: pd.DataFrame = visualize_df_2[
                (visualize_df_2.distance_2d >= ranges_xy[i + 1])
                & (visualize_df_2.distance_2d < ranges_xy[i])
            ]
            # print(_, len(visualize_df_filt_1), len(visualize_df_filt_2))
            if self.show_gt:
                df_gt_filt = df_gt[
                    (df_gt.distance_2d >= ranges_xy[i + 1]) & (df_gt.distance_2d < ranges_xy[i])
                ]
                fig.add_trace(
                    go.Histogram(
                        x=df_gt_filt.semantic_label,
                        name=f"gt <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                        marker=dict(color="yellowgreen"),
                    ),
                    row=1,
                    col=i + 1,
                )
            class_names_1 = sorted(
                visualize_df_filt_1.semantic_label.unique().tolist(), key=class_names.index
            )
            class_names_2 = sorted(
                visualize_df_filt_2.semantic_label.unique().tolist(), key=class_names.index
            )

            # # todo: map class names to colors, not working correctly
            # colors_1 = [class_color_mapping[cn] for cn in class_names_1]
            # colors_2 = [class_color_mapping[cn] for cn in class_names_2]

            fig.add_trace(
                go.Histogram(
                    x=visualize_df_filt_1.semantic_label,
                    name=f"{self.objects_1_source_name} <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                    marker=dict(color="blue"),
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Histogram(
                    x=visualize_df_filt_2.semantic_label,
                    name=f"{self.objects_2_source_name} <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                    marker=dict(color="orange"),
                ),
                row=1,
                col=i + 1,
            )

            # # todo ValueError: max() arg is an empty sequence for fp all sscenes
            if self.show_gt:
                curr_max_bar_height = (
                    max(df_gt_filt.semantic_label.value_counts()) if len(df_gt_filt) > 0 else 0
                )
            else:
                max_bar_height_1 = (
                    max(visualize_df_filt_1.semantic_label.value_counts())
                    if len(visualize_df_filt_1) > 0
                    else 0
                )
                max_bar_height_2 = (
                    max(visualize_df_filt_2.semantic_label.value_counts())
                    if len(visualize_df_filt_2) > 0
                    else 0
                )
                curr_max_bar_height = max(max_bar_height_1, max_bar_height_2)
            max_bar_height = (
                curr_max_bar_height if curr_max_bar_height > max_bar_height else max_bar_height
            )
            filtered_classes_1 = visualize_df_filt_1.semantic_label.unique().tolist()
            filtered_classes_2 = visualize_df_filt_2.semantic_label.unique().tolist()
            filtered_classes = list(set(filtered_classes_1 + filtered_classes_2))
            if self.show_gt:
                filtered_classes_gt = df_gt_filt.semantic_label.unique().tolist()
                filtered_classes = list(set(filtered_classes + filtered_classes_gt))
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=sorted(filtered_classes, key=class_names.index),
                row=1,
                col=i + 1,
            )

        fig.update_yaxes(range=(0, 1.1 * max_bar_height))

        # todo put these into legend also
        fig.update_layout(
            title="<b>Number of objects in various 2d distance ranges</b>"
            + f"<br>Compared: <b>{self.objects_1_source_name}</b> vs <b>{self.objects_2_source_name}</b>"
            + f"<br>Objects used for comparison: <b>{self.visualized_results_name}</b>",
            font=dict(size=7),  # title size
        )
        fig.update_annotations(font_size=9)  # ranges description size over each hist
        if self.show:
            fig.show()

        fig.write_html(self.save_dir + "/hist_object_count_for_each_distance.html")

    # note(pawel-kotowski): this will be replaced with px.scatter
    def hist2d_object_center_xy_for_each_class(
        self,
        class_names: List[str],
        xlim_dict: Dict[str, List[float]] = None,
        ylim_dict: Dict[str, List[float]] = None,
    ) -> None:
        """[summary]
        Show 2d-histogram of x and y in each class.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            xlim_dict, ylim_dict (Dict[str, List[float]]):
                    xlim, ylim for each class
                    e.g. xlim_dict['car'] is [xmin, xmax] for car

        """
        def plot(df, col):
            hist = axes[cls_i, col].hist2d(
                df.x, df.y, norm=mpl.colors.LogNorm()
            )
            # axes[cls_i, col].plot(
            #     x_mean_1, y_mean_1, marker="x", color="r", markersize=10, markeredgewidth=3
            # )
            axes[cls_i, col].set_title(
                f"{class_name}"#: (x, y, z)=({x_mean_1:.2f}±{x_std_1:.2f}, {y_mean_1:.2f}±{y_std_1:.2f}, {z_mean_1:.2f}±{z_std_1:.2f})"
            )
            axes[cls_i, col].set_xlabel("x [m] ")
            axes[cls_i, col].set_ylabel("y [m] ")
            if xlim_dict:
                axes[cls_i, col].set_xlim(xlim_dict[class_name][0], xlim_dict[class_name][1])
            if ylim_dict:
                axes[cls_i, col].set_ylim(ylim_dict[class_name][0], ylim_dict[class_name][1])
            # plt.colorbar(hist[3], ax=axes[cls_i, col])

        def get_result_id(row):
            if not pd.isna(row["gt_instance_token"]):
                return row["gt_instance_token"]
            else:
                return row["fp_id"] # it's tmp (not available in ape)

        axes = self.get_subplots_2(class_names, n_cols=2)

        for cls_i, class_name in enumerate(class_names):
            # _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            visualize_df_1 = self.df_objects_1[self.df_objects_1.semantic_label == class_name]
            visualize_df_2 = self.df_objects_2[self.df_objects_2.semantic_label == class_name]
            # df_gt = self.df_gt[self.df_gt.semantic_label == class_name]

            # visualize_df_1 = self.df_objects_1
            # visualize_df_2 = self.df_objects_2

            if len(visualize_df_1) > 0 or len(visualize_df_2) > 0:
                x_mean_1, x_std_1 = 10, 10  # visualize_df_1.x.mean(), visualize_df_1.x.std()
                y_mean_1, y_std_1 = 10, 10  # visualize_df_1.y.mean(), visualize_df_1.y.std()
                z_mean_1, z_std_1 = 10, 10  # visualize_df_1.z.mean(), visualize_df_1.z.std()

                if len(visualize_df_1) > 0 and len(visualize_df_2) > 0:
                    visualize_df_1['result_id'] = visualize_df_1.apply(
                        lambda row: get_result_id(row), axis=1
                    )
                    visualize_df_2['result_id'] = visualize_df_2.apply(
                        lambda row: get_result_id(row), axis=1
                    )
                    # # create df from visualize_df_1 and visualize_df_2 with rows where sample_token and result_id is different in visualize_df_1 and visualize_df_2
                    df_diff = pd.merge(
                        visualize_df_1,
                        visualize_df_2,
                        how="outer",
                        indicator=True,
                        on=["sample_token", "result_id"],
                    )
                    # drop if in both
                    df_diff = df_diff[df_diff._merge != "both"]
                    for col in df_diff.columns:
                        if col.endswith('_x'):
                            new_col = col[:-2]
                            df_diff[new_col] = df_diff[col].combine_first(df_diff[new_col + '_y'])
                            df_diff = df_diff.drop(columns=[col, new_col + '_y'])

                    df_diff_l = df_diff[df_diff._merge == "left_only"]
                    df_diff_r = df_diff[df_diff._merge == "right_only"]
                    plot(df_diff_l, col=0)
                    plot(df_diff_r, col=1)
                else:
                    if len(visualize_df_1) > 0:
                        plot(visualize_df_1, col=0)
                    else:
                        plot(visualize_df_2, col=1)

        #         plt.savefig(self.save_dir + "/hist2d_object_center_xy_for_each_class.svg")


class EDAResultsComparisonVisualizer:
    def __init__(
        self,
        objects_1: Union[
            Iterable[List[DynamicObjectWithPerceptionResult]], Iterable[List[DynamicObject]]
        ],
        objects_2: Union[
            Iterable[List[DynamicObjectWithPerceptionResult]], Iterable[List[DynamicObject]]
        ],
        gt_objects: Iterable[List[DynamicObject]],
        save_dir: str,
        show: bool = False,
        objects_1_source_name: str = "results from first model",
        objects_2_source_name: str = "results from second model",
        visualized_results_name: str = "some kind of results, e.g. false negatives",
        show_gt=True,
    ):
        self.objects_1_source_name = objects_1_source_name
        self.objects_2_source_name = objects_2_source_name
        self.visualized_results_name = visualized_results_name
        self.show = show
        self.show_gt = show_gt
        self.save_dir: str = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if with_result(objects_1) and with_result(objects_2):
            self.df_objects_1 = with_result_objects_to_df(objects_1)
            self.df_objects_2 = with_result_objects_to_df(objects_2)
        else:
            self.df_objects_1 = without_result_objects_to_df(objects_1)
            self.df_objects_2 = without_result_objects_to_df(objects_2)
        self.df_gt = without_result_objects_to_df(gt_objects)

    def get_subplots(self, class_names: List[str]) -> None:
        """[summary]
        Get subplots

        Args:
            class_names (List[str]):
                    names of class you want to visualize.

        Return:
            axes (numpy.ndarray):
                    axes of subplots
        """
        col_size = len(class_names)
        fig, axes = plt.subplots(col_size, 1, figsize=(16, 6 * col_size))
        axes = axes.flatten()
        return axes

    # todo:numbers over hists
    def hist_object_count_for_each_distance_comparison(
        self, class_names: List[str], ranges_xy: List[Union[int, float]] = [125, 100, 75, 50, 25]
    ) -> None:
        """[summary]
        Show histogram of number of objects that are less than the certain distance in x-y plane.
        Distance is specified by ranges_xy.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            ranges_xy (List[Union[int, float]]):
                    distances in x-y plane.
        """
        ranges_xy = ranges_xy + [0]

        subplot_titles: List[str] = []
        for i, _ in enumerate(ranges_xy[:-1]):
            subplot_titles.append(f"<{ranges_xy[i+1]}m, {ranges_xy[i]}m)")

        fig: Figure = make_subplots(rows=1, cols=len(ranges_xy) - 1, subplot_titles=subplot_titles)

        visualize_df_1 = self.df_objects_1[self.df_objects_1.name.isin(class_names)]
        visualize_df_2 = self.df_objects_2[self.df_objects_2.name.isin(class_names)]
        df_gt = self.df_gt[self.df_gt.name.isin(class_names)]
        max_bar_height = 0
        for i, _ in enumerate(ranges_xy[:-1]):
            visualize_df_filt_1: pd.DataFrame = visualize_df_1[
                (visualize_df_1.distance_2d >= ranges_xy[i + 1])
                & (visualize_df_1.distance_2d < ranges_xy[i])
            ]
            visualize_df_filt_2: pd.DataFrame = visualize_df_2[
                (visualize_df_2.distance_2d >= ranges_xy[i + 1])
                & (visualize_df_2.distance_2d < ranges_xy[i])
            ]
            if self.show_gt:
                df_gt_filt = df_gt[
                    (df_gt.distance_2d >= ranges_xy[i + 1]) & (df_gt.distance_2d < ranges_xy[i])
                ]
                fig.add_trace(
                    go.Histogram(
                        x=df_gt_filt["name"],
                        name=f"gt <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                        marker=dict(color="yellowgreen"),
                    ),
                    row=1,
                    col=i + 1,
                )
            class_names_1 = sorted(
                visualize_df_filt_1.name.unique().tolist(), key=class_names.index
            )
            class_names_2 = sorted(
                visualize_df_filt_2.name.unique().tolist(), key=class_names.index
            )

            # # todo: map class names to colors, not working correctly
            # colors_1 = [class_color_mapping[cn] for cn in class_names_1]
            # colors_2 = [class_color_mapping[cn] for cn in class_names_2]

            fig.add_trace(
                go.Histogram(
                    x=visualize_df_filt_1["name"],
                    name=f"{self.objects_1_source_name} <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                    marker=dict(color="blue"),
                    # customdata=#list of series
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Histogram(
                    x=visualize_df_filt_2["name"],
                    name=f"{self.objects_2_source_name} <{ranges_xy[i+1]}m, {ranges_xy[i]}m)",
                    marker=dict(color="orange"),
                ),
                row=1,
                col=i + 1,
            )

            # # todo ValueError: max() arg is an empty sequence for fp all sscenes
            if self.show_gt:
                curr_max_bar_height = (
                    max(df_gt_filt.name.value_counts()) if len(df_gt_filt) > 0 else 0
                )
            else:
                max_bar_height_1 = (
                    max(visualize_df_filt_1.name.value_counts())
                    if len(visualize_df_filt_1) > 0
                    else 0
                )
                max_bar_height_2 = (
                    max(visualize_df_filt_2.name.value_counts())
                    if len(visualize_df_filt_2) > 0
                    else 0
                )
                curr_max_bar_height = max(max_bar_height_1, max_bar_height_2)
            max_bar_height = (
                curr_max_bar_height if curr_max_bar_height > max_bar_height else max_bar_height
            )
        # todo:check if ok
        filtered_classes_1 = visualize_df_1.name.unique().tolist()
        filtered_classes_2 = visualize_df_2.name.unique().tolist()
        filtered_classes = list(set(filtered_classes_1 + filtered_classes_2))
        # fig.update_xaxes(autorange="reversed") #didnt work
        fig.update_xaxes(
            categoryorder="array", categoryarray=sorted(filtered_classes, key=class_names.index)
        )
        fig.update_yaxes(range=(0, 1.1 * max_bar_height))

        # todo put these into legend also
        fig.update_layout(
            title="<b>Number of objects in various 2d distance ranges</b>"
            + f"<br>Compared: <b>{self.objects_1_source_name}</b> vs <b>{self.objects_2_source_name}</b>"
            + f"<br>Objects used for comparison: <b>{self.visualized_results_name}</b>",
            font=dict(size=7),  # title size
        )
        fig.update_annotations(font_size=9)  # ranges description size over each hist
        if self.show:
            fig.show()

        fig.write_html(self.save_dir + "/hist_object_count_for_each_distance.html")

    def hist2d_object_center_xy_for_each_class(
        self,
        class_names: List[str],
        xlim_dict: Dict[str, List[float]] = None,
        ylim_dict: Dict[str, List[float]] = None,
    ) -> None:
        """[summary]
        Show 2d-histogram of x and y in each class.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
            xlim_dict, ylim_dict (Dict[str, List[float]]):
                    xlim, ylim for each class
                    e.g. xlim_dict['car'] is [xmin, xmax] for car

        """

        axes = self.get_subplots(class_names)

        for cls_i, class_name in enumerate(class_names):
            # _df_cls = self.visualize_df[self.visualize_df.name == class_name]
            visualize_df_1 = self.df_objects_1[self.df_objects_1.name.isin(class_names)]
            visualize_df_2 = self.df_objects_2[self.df_objects_2.name.isin(class_names)]
            df_gt = self.df_gt[self.df_gt.name.isin(class_names)]
            if len(visualize_df_1) > 0:
                x_mean_1, x_std_1 = visualize_df_1.x.mean(), visualize_df_1.x.std()
                y_mean_1, y_std_1 = visualize_df_1.y.mean(), visualize_df_1.y.std()
                z_mean_1, z_std_1 = visualize_df_1.z.mean(), visualize_df_1.z.std()

                hist = axes[cls_i].hist2d(
                    visualize_df_1.x, visualize_df_1.y, bins=50, norm=mpl.colors.LogNorm()
                )

                # axes[cls_i].plot(
                #     x_mean_1, y_mean_1, marker="x", color="r", markersize=10, markeredgewidth=3
                # )
                axes[cls_i].set_title(
                    f"{class_name}: (x, y, z)=({x_mean_1:.2f}±{x_std_1:.2f}, {y_mean_1:.2f}±{y_std_1:.2f}, {z_mean_1:.2f}±{z_std_1:.2f})"
                )
                axes[cls_i].set_xlabel("x [m] ")
                axes[cls_i].set_ylabel("y [m] ")
                if xlim_dict:
                    axes[cls_i].set_xlim(xlim_dict[class_name][0], xlim_dict[class_name][1])
                if ylim_dict:
                    axes[cls_i].set_ylim(ylim_dict[class_name][0], ylim_dict[class_name][1])
                plt.colorbar(hist[3], ax=axes[cls_i])

        plt.savefig(self.save_dir + "/hist2d_object_center_xy_for_each_class.svg")


class EDAManager:
    """[summary]
    EDA Manager class.
    EDA needs TP, FP, FN, FP with high confidence visualization and reporting some rates.
    This class includes some methods used in EDA.

    Attributes:
        self.root_path (str): root path for saving graphs
        self.class_names (List[str]): names of class you want to visualize
        self.ranges_xy (List[Union[int, float]]): distances in x-y plane for histogram of number of objects
        self.xylim_dict (Dict[str, List[float]]): xlim, ylim for each class used in hist2d_object_center_xy_for_each_class
        self.width_lim_dict (Dict[str, List[float]]): width_lim for each class used in hist2d_object_wl_for_each_class
        self.length_lim_dict (Dict[str, List[float]]): length_lim for each class used in hist2d_object_wl_for_each_class
        self.label_converter (LabelConverter): label converter
    """

    def __init__(
        self,
        root_path: str,
        class_names: List[str],
        ranges_xy: List[float],
        xylim_dict: Dict[str, List[float]],
        width_lim_dict: Dict[str, List[float]],
        length_lim_dict: Dict[str, List[float]],
        evaluation_task: EvaluationTask,
        merge_similar_labels: bool = False,
        label_prefix: str = "autoware",
        show: bool = False,
    ) -> None:
        """[summary]

        Args:
            root_path (str): root path for saving graphs
            class_names (List[str]): names of class you want to visualize
            ranges_xy (List[Union[int, float]]): distances in x-y plane for histogram of number of objects
            xylim_dict (Dict[str, List[float]]): xlim, ylim for each class for hist2d_object_center_xy_for_each_class
            width_lim_dict (Dict[str, List[float]]): width_lim for each class used in hist2d_object_wl_for_each_class
            length_lim_dict (Dict[str, List[float]]): length_lim for each class used in hist2d_object_wl_for_each_class
            merge_similar_labels (bool): Whether merge similar labels. Defaults to False.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE
            show (bool): Whether show visualized figures. Defaults to False.
        """
        self.root_path = root_path
        self.class_names = class_names
        self.ranges_xy = ranges_xy
        self.xylim_dict = xylim_dict
        self.width_lim_dict = width_lim_dict
        self.length_lim_dict = length_lim_dict
        self.label_converter = LabelConverter(  # todo accept as param
            evaluation_task=evaluation_task,
            merge_similar_labels=merge_similar_labels,
            label_prefix=label_prefix,
            count_label_number=True,
        )
        self.target_labels: List[LabelType] = [
            self.label_converter.convert_name(name) for name in self.class_names
        ]
        self.show = show
        self.results = []
        self.result_dicts = []

    # todo:refactor
    def set_result_dicts(self, result_dicts: List[Dict]) -> None:
        self.result_dicts = result_dicts

    # set_results for lists of objects
    def set_results(
        self, results: List[Dict[str, List[DynamicObjectWithPerceptionResult]]]
    ) -> None:
        # split to fps with and without gt
        for result_data in results:
            fp_objects = result_data["fp_objects"]
            fp_objects_with_gt = [obj for obj in fp_objects if obj.ground_truth_object is not None]
            fp_objects_without_gt = [obj for obj in fp_objects if obj.ground_truth_object is None]
            result_data["fp_objects_with_gt"] = fp_objects_with_gt
            result_data["fp_objects_without_gt"] = fp_objects_without_gt
        self.results = results

    def visualize_ground_truth_objects(
        self, ground_truth_object_dict: Dict[str, List[DynamicObject]]
    ) -> None:
        """[summary]
        visualize ground truth objects

        Args:
            ground_truth_object_dict (Dict[str, List[DynamicObject]]):
                    Key of dict is name of object. This is used in directory name for saving graphs.
                    Value is list of ground truth object.
        """
        # visualize ground truths
        for object_name, ground_truth_objects in ground_truth_object_dict.items():
            self.visualize(ground_truth_objects, object_name)

    def visualize_estimated_objects(
        self, estimated_object_dict: Dict[str, List[DynamicObjectWithPerceptionResult]]
    ) -> None:
        """[summary]
        visualize estimated objects

        Args:
            estimated_object_dict (Dict[str, List[DynamicObjectWithPerceptionResult]]]):
                    Key of dict is name of object. This is used in directory name for saving graphs.
                    Value is list of estimated object.
        """
        # visualize estimated objects
        for object_name, estimated_objects in estimated_object_dict.items():
            self.visualize(estimated_objects, object_name)

    def calculate_and_set_results(
        self,
        objects_source_name: str,
        object_results: List[DynamicObjectWithPerceptionResult],
        ground_truth_objects: List[DynamicObject],
        matching_mode: MatchingMode,
        matching_threshold: float,
        confidence_threshold: float,
    ):
        """[summary]
        visualize TP, FP, FN objects and FP objects with high confidence

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]):
                    list of estimated object
            ground_truth_objects (List[DynamicObject]):
                    list of ground truth object
            matching_mode (MatchingMode):
                    mode for matching (e.g. MatchingMode.CENTERDISTANCE)
            matching_threshold (float):
                    matching threshold (e.g. 0.5)
            confidence_threshold (float):
                    confidence threshold for visualization
        """
        # todo: check if the same issue with fewer class names
        tp_results, fp_results = divide_tp_fp_objects(
            object_results,
            self.target_labels,
            matching_mode,
            [matching_threshold] * len(self.target_labels),
        )
        # fp_results_with_high_confidence = filter_object_results(
        #     object_results=fp_results,
        #     target_labels=self.target_labels,
        #     confidence_threshold_list=[confidence_threshold] * len(self.target_labels),
        # )
        fn_gts: List[DynamicObject] = get_fn_objects(
            ground_truth_objects, object_results, tp_results
        )
        self.results.append(
            {
                "objects_source_name": objects_source_name,
                "tp_results": tp_results,
                "fp_results": fp_results,
                "fn_results": fn_gts,
            }
        )

    # todo: evaluate refers to mAP in ape, so change name, the only "evaluation" here is done in report rates, besides that its just splitting already estimated objects and visualization
    def visualize_evaluated_results(
        self,
    ) -> None:
        # todo check if calculated
        self.visualize(self.results[0]["tp_objects"], visualized_results_name="tp_results")
        self.visualize(self.results[0]["fp_objects"], visualized_results_name="fp_results")
        self.visualize(self.results[0]["fn_objects"], visualized_results_name="fn_results")

    def visualize(
        self,
        objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]],
        visualized_results_name: str,
    ) -> None:
        """[summary]
        visualize objects

        Args:
            objects (Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]):
                    estimated objects(List[DynamicObject]) or ground truth objects(List[DynamicObjectWithPerceptionResult]]) which you want to visualize
            visualized_results_name (str):
                    name for visualized results type, e.g. "tp_results". This is used in directory name for saving graphs and in plots.
        """
        visualizer = EDAVisualizer(
            objects,
            self.root_path + "/" + visualized_results_name,
            self.show,
            visualized_results_name,
            self.results[0]["objects_source_name"],
        )
        visualizer.visualize(
            class_names=self.class_names,
            ranges_xy=self.ranges_xy,
            xylim_dict=self.xylim_dict,
            width_lim_dict=self.width_lim_dict,
            length_lim_dict=self.length_lim_dict,
        )

    def visualize_all_evaluated_results_comparison(self, to_vis=["tps", "fps", "fns"]):
        # todo check if self.results calculated or set
        # we assume same gt objects from both results
        if type(to_vis) == str:
            to_vis = [to_vis]

        (
            ground_truth_objects,
            objects_source_name_1,
            tp_objects_1,
            fp_objects_1,
            fp_with_gt_objects_1,
            fp_without_gt_objects_1,
            fn_gts_1,
        ) = (
            self.results[0]["gt_objects"],
            self.results[0]["objects_source_name"],
            self.results[0]["tp_objects"],
            self.results[0]["fp_objects"],
            self.results[0]["fp_objects_with_gt"],
            self.results[0]["fp_objects_without_gt"],
            self.results[0]["fn_objects"],
        )
        (
            objects_source_name_2,
            tp_objects_2,
            fp_objects_2,
            fp_with_gt_objects_2,
            fp_without_gt_objects_2,
            fn_gts_2,
        ) = (
            self.results[1]["objects_source_name"],
            self.results[1]["tp_objects"],
            self.results[1]["fp_objects"],
            self.results[1]["fp_objects_with_gt"],
            self.results[1]["fp_objects_without_gt"],
            self.results[1]["fn_objects"],
        )

        if "tps" in to_vis:
            self.visualize_evaluated_results_comparison(
                objects_source_name_1,
                objects_source_name_2,
                tp_objects_1,
                tp_objects_2,
                ground_truth_objects,
                visualized_results_name="tps",
            )
        if "fps" in to_vis:
            self.visualize_evaluated_results_comparison(
                objects_source_name_1,
                objects_source_name_2,
                fp_objects_1,
                fp_objects_2,
                ground_truth_objects,
                visualized_results_name="fps_all",
                show_gt=False,
            )
            self.visualize_evaluated_results_comparison(
                objects_source_name_1,
                objects_source_name_2,
                fp_with_gt_objects_1,
                fp_with_gt_objects_2,
                ground_truth_objects,
                visualized_results_name="fps_with_gt",
                show_gt=False,
            )
            self.visualize_evaluated_results_comparison(
                objects_source_name_1,
                objects_source_name_2,
                fp_without_gt_objects_1,
                fp_without_gt_objects_2,
                ground_truth_objects,
                visualized_results_name="fps_without_gt",
                show_gt=False,
            )
        if "fns" in to_vis:
            self.visualize_evaluated_results_comparison(
                objects_source_name_1,
                objects_source_name_2,
                fn_gts_1,
                fn_gts_2,
                ground_truth_objects,
                visualized_results_name="fn_gts",
            )

    def visualize_all_evaluated_results_comparison_dfs(self, to_vis=["tps", "fps", "fns"]):
        # todo check if self.results calculated or set
        # we assume same gt objects from both results
        # todoL to vis doesnt work now
        if type(to_vis) == str:
            to_vis = [to_vis]

        # todo: refactor result to class namedtuple or sth
        for name in self.result_dicts[0]:
            if name == "source_name" or name == "gt":
                continue
            source_name_1 = self.result_dicts[0]["source_name"]
            source_name_2 = self.result_dicts[1]["source_name"]
            df_1 = self.result_dicts[0][name]
            df_2 = self.result_dicts[1][name]
            df_gt = self.result_dicts[0]["gt"]  # separately somehow?
            self.visualize_evaluated_results_comparison_dfs(
                source_name_1, source_name_2, df_1, df_2, df_gt, name, show_gt="without_gt" in name
            )

    def visualize_evaluated_results_comparison(
        self,
        objects_source_name_1: str,
        objects_source_name_2: str,
        objects_1,
        objects_2,
        ground_truth_objects,
        visualized_results_name,
        show_gt=True,
    ):
        eda_results_comparison_visualizer = EDAResultsComparisonVisualizer(
            objects_1,
            objects_2,
            ground_truth_objects,
            f"{self.root_path}/{visualized_results_name}",
            self.show,
            objects_source_name_1,
            objects_source_name_2,
            visualized_results_name,
            show_gt,
        )

        for cn in self.class_names:
            eda_results_comparison_visualizer.hist_object_count_for_each_distance_comparison(
                [cn], ranges_xy=self.ranges_xy
            )

        # eda_results_comparison_visualizer.hist_object_count_for_each_distance_comparison(
        #     self.class_names, ranges_xy=self.ranges_xy
        # )

        # # # makes separate plot for each class
        eda_results_comparison_visualizer.hist2d_object_center_xy_for_each_class(
            self.class_names, self.xylim_dict, self.xylim_dict
        )

    def visualize_evaluated_results_comparison_dfs(
        self,
        objects_source_name_1: str,
        objects_source_name_2: str,
        df_1,
        df_2,
        df_gt,
        visualized_results_name,
        show_gt=True,
    ):
        eda_results_comparison_visualizer = EDAResultsComparisonVisualizerDfs(
            df_1,
            df_2,
            df_gt,
            f"{self.root_path}/{visualized_results_name}",
            self.show,
            objects_source_name_1,
            objects_source_name_2,
            visualized_results_name,
            show_gt,
        )

        # for cn in self.class_names:
        #     eda_results_comparison_visualizer.hist_object_count_for_each_distance_comparison(
        #         [cn], ranges_xy=self.ranges_xy
        #     )

        eda_results_comparison_visualizer.hist_object_count_for_each_distance_comparison(
            self.class_names, ranges_xy=self.ranges_xy
        )

        # replace with px scatter, matplotlib hist works poorly
        # makes separate plot for each class
        # eda_results_comparison_visualizer.hist2d_object_center_xy_for_each_class(
        #     self.class_names, self.xylim_dict, self.xylim_dict
        # )

    def report_rates(
        self,
        tp_num: int,
        fp_num: int,
        estimated_objects_num: int,
        fn_num: int,
        ground_truth_num: int,
    ) -> None:
        """[summary]
        report TP, FP, FN rate.

        Args:
            tp_num, fp_num, estimated_objects_num, fn_num, ground_truth_num (int): number of TP, FP, estimated objects, FN, ground truths
        """
        tp_rate_precision = tp_num / estimated_objects_num
        fp_rate = fp_num / estimated_objects_num
        logger.info(f"TP rate (Precision): {tp_rate_precision}")
        logger.info(f"FP rate: {fp_rate}")

        tp_rate_recall = tp_num / ground_truth_num
        fn_rate = fn_num / ground_truth_num
        logger.info(f"TP rate (Recall): {tp_rate_recall}")
        logger.info(f"FN rate: {fn_rate}")
