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
from typing import List
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import LabelType
from perception_eval.matching import MatchingMode
from perception_eval.matching.objects_filter import divide_tp_fp_objects
from perception_eval.matching.objects_filter import filter_object_results
from perception_eval.matching.objects_filter import get_fn_objects
from perception_eval.object import DynamicObject
from perception_eval.result import DynamicObjectWithPerceptionResult
from plotly import graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

try:
    import pandas_profiling as pdp
except ImportError:
    import ydata_profiling as pdp

logger = getLogger(__name__)


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
    ) -> None:
        """[summary]

        Args:
            objects (Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]):
                    estimated objects(List[DynamicObject]) or ground truth objects(List[DynamicObjectWithPerceptionResult]]) which you want to visualize
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
            show (bool): Whether show visualized figures. Defaults to False.
        """
        self.visualize_df: pd.DataFrame = self.objects_to_df(objects)
        self.save_dir: str = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.show: show = show

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
        if isinstance(objects[0], DynamicObject):
            self.is_gt = True

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

        elif isinstance(objects[0], DynamicObjectWithPerceptionResult):
            self.is_gt = False

            names = np.stack(
                [estimated_object.estimated_object.semantic_label.label.value for estimated_object in objects]
            )
            xyz = np.stack([estimated_object.estimated_object.state.position for estimated_object in objects])
            wlh = np.stack([estimated_object.estimated_object.state.size for estimated_object in objects])
            pcd_nums = np.stack([estimated_object.estimated_object.pointcloud_num for estimated_object in objects])
            center_distances = np.stack([estimated_object.center_distance.value for estimated_object in objects])
            confidences = np.stack([estimated_object.estimated_object.semantic_score for estimated_object in objects])

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
                )
            )
            df["distance_2d"] = np.sqrt((df["x"]) ** 2 + (df["y"]) ** 2)

        return df

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

    def hist_object_count_for_each_distance(self, class_names: List[str], ranges_xy: List[Union[int, float]]) -> None:
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

        for i, range_xy in enumerate(ranges_xy):
            _df: pd.DataFrame = self.visualize_df[self.visualize_df.distance_2d < range_xy]

            fig.add_trace(
                go.Histogram(x=_df["name"], name=f"#objects: ~{range_xy}m", marker=dict(color="blue")),
                row=1,
                col=i + 1,
            )

        fig.update_yaxes(range=[0, len(self.visualize_df)])
        fig.update_xaxes(categoryorder="array", categoryarray=class_names)
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

            layout = go.Histogram(x=dist2d, name=f"{class_name}: #objects={len(_df_cls):,}", nbinsx=400)
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
                axes[cls_i].plot(w_mean, l_mean, marker="x", color="r", markersize=10, markeredgewidth=3)
                axes[cls_i].set_title(
                    f"{class_name}: (w, l, h)=({w_mean:.2f}±{w_std:.2f}, {l_mean:.2f}±{l_std:.2f}, {h_mean:.2f}±{h_std:.2f})"
                )
                axes[cls_i].set_xlabel("width")
                axes[cls_i].set_ylabel("length")
                if width_lim_dict:
                    axes[cls_i].set_xlim(width_lim_dict[class_name][0], width_lim_dict[class_name][1])
                if length_lim_dict:
                    axes[cls_i].set_ylim(length_lim_dict[class_name][0], length_lim_dict[class_name][1])
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
                axes[cls_i].plot(x_mean, y_mean, marker="x", color="r", markersize=10, markeredgewidth=3)
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

    def hist2d_object_num_points_for_each_class(self, class_names: List[str], max_pts: int = 500) -> None:
        """[summary]
        Show 2d-histogram of number of point clouds in each class.
        Ground truth objects only have the number of point cloud in bbox, so this method works only for ground truth objects.

        Args:
            class_names (List[str]):
                    names of class you want to visualize.
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
            try:
                report = pdp.ProfileReport(_df_cls)
                report.to_file(self.save_dir + "/" + file_name + f"_{class_name}.html")
            except ValueError:
                logger.warning("Empty DataFrame is detected, skip to save report")


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
        self.label_converter = LabelConverter(
            evaluation_task=evaluation_task,
            merge_similar_labels=merge_similar_labels,
            label_prefix=label_prefix,
            count_label_number=True,
        )
        self.show = show

    def visualize_ground_truth_objects(self, ground_truth_object_dict: Dict[str, List[DynamicObject]]) -> None:
        """[summary]
        visualize ground truth objects

        Args:
            ground_truth_object_dict (Dict[str, List[DynamicObject]]):
                    Key of dict is name of object. This is used in directory name for saving graphs.
                    Value is list of ground truth object.
        """
        # visualize ground truths
        for object_name, ground_truth_objects in ground_truth_object_dict.items():
            self.visualize(ground_truth_objects, object_name, is_gt=True)

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
            self.visualize(estimated_objects, object_name, is_gt=False)

    def visualize_evaluated_results(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ground_truth_objects: List[DynamicObject],
        matching_mode: MatchingMode,
        matching_threshold: float,
        confidence_threshold: float,
    ) -> None:
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
        target_labels: List[LabelType] = [self.label_converter.convert_name(name) for name in self.class_names]
        # visualize tp, fp in estimated objects
        tp_results, fp_results = divide_tp_fp_objects(
            object_results,
            target_labels,
            matching_mode,
            [matching_threshold] * len(target_labels),
        )
        if len(tp_results) == 0:
            logger.info("No TP results, so no graphs are generated for TP.")
        else:
            self.visualize(tp_results, "tp_results", is_gt=False)

        if len(fp_results) == 0:
            logger.info("No FP results, so no graphs are generated for FP.")
        else:
            self.visualize(fp_results, "fp_results", is_gt=False)
            # visualize fp with high confidence in estimated objects
            confidence_threshold_list: List[float] = [confidence_threshold] * len(target_labels)
            fp_results_with_high_confidence = filter_object_results(
                object_results=fp_results,
                target_labels=target_labels,
                confidence_threshold_list=confidence_threshold_list,
            )
            self.visualize(
                fp_results_with_high_confidence,
                "fp_results_with_high_confidence",
                is_gt=False,
            )

        # visualize fn in ground truth objects
        fn_gts: List[DynamicObject] = get_fn_objects(ground_truth_objects, object_results, tp_results)
        if len(fn_gts) == 0:
            logger.info("No FN ground truths, so no graphs are generated for FN.")
        else:
            self.visualize(fn_gts, "fn_ground_truths", is_gt=True)

        # report rates
        self.report_rates(
            tp_num=len(tp_results),
            fp_num=len(fp_results),
            estimated_objects_num=len(object_results),
            fn_num=len(fn_gts),
            ground_truth_num=len(ground_truth_objects),
        )

    def visualize(
        self,
        objects: Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]],
        objects_name: str,
        is_gt: bool,
    ) -> None:
        """[summary]
        visualize objects

        Args:
            objects (Union[List[DynamicObject], List[DynamicObjectWithPerceptionResult]]):
                    estimated objects(List[DynamicObject]) or ground truth objects(List[DynamicObjectWithPerceptionResult]]) which you want to visualize
            objects_name (str):
                    name of object. This is used in directory name for saving graphs.
            is_gt (bool):
                    ground truth objects or not
        """
        visualizer = EDAVisualizer(objects, self.root_path + "/" + objects_name, self.show)
        visualizer.hist_object_count_for_each_distance(self.class_names, ranges_xy=self.ranges_xy)
        visualizer.hist_object_dist2d_for_each_class(self.class_names)
        visualizer.hist2d_object_wl_for_each_class(
            self.class_names,
            width_lim_dict=self.width_lim_dict,
            length_lim_dict=self.length_lim_dict,
        )
        visualizer.hist2d_object_center_xy_for_each_class(
            self.class_names, xlim_dict=self.xylim_dict, ylim_dict=self.xylim_dict
        )
        if is_gt:
            visualizer.hist2d_object_num_points_for_each_class(self.class_names)
        visualizer.get_pandas_profiling(self.class_names, "profiling_" + objects_name)

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
