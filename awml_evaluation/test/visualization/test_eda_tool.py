import os
from secrets import token_hex
import tempfile
from test.util.dummy_object import make_dummy_data
from typing import Dict
from typing import List
from typing import Union

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.object_result import get_object_results
from awml_evaluation.util.debug import get_objects_with_difference
from awml_evaluation.visualization.eda_tool import EDAManager
from awml_evaluation.visualization.eda_tool import EDAVisualizer
from pyquaternion.quaternion import Quaternion
import pytest


class TestEDAVisualizer:
    """[summary]
    EDA Visualization test class.

    Attributes:
        self.dummy_estimated_objects (List[DynamicObject]): dummy estimated objects.
        self.dummy_ground_truth_objects (List[DynamicObject]): dummy ground truth objects.
        self.object_results (List[DynamicObjectWithPerceptionResult]): dummy object results.
        self.class_names (List[str]): names of class you want to visualize.
        self.ranges_xy (List[Union[int, float]]): distances in x-y plane.
        self.xylim_dict (Dict[str, List[float]]): xlim, ylim for visualization. e.g. xlim_dict['car'] is [xmin, xmax] for car
    """

    dummy_estimated_objects: List[DynamicObject] = []
    dummy_ground_truth_objects: List[DynamicObject] = []
    dummy_estimated_objects, dummy_ground_truth_objects = make_dummy_data()

    dummy_estimated_objects.append(
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
        ),
    )

    object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
        estimated_objects=dummy_estimated_objects,
        ground_truth_objects=dummy_ground_truth_objects,
    )

    class_names: List[str] = ["car", "pedestrian", "bicycle"]
    ranges_xy: List[Union[int, float]] = [250, 100, 50, 25]
    xylim_dict: Dict[str, List[float]] = {
        "car": [-100, 100],
        "bicycle": [-100, 100],
        "pedestrian": [-100, 100],
    }

    @pytest.fixture
    def save_dir(self):
        """[summary]
        Path for saving files.
        In test, make files in this directory.
        Then, remove this directory with files included in this directory.
        """
        tmpdir = tempfile.TemporaryDirectory()
        yield tmpdir.name
        tmpdir.cleanup()

    def test_objects_to_df_for_object_results(self, save_dir: str):
        """[summary]
        Check if fields exist in DataFrame when input is List[DynamicObjectWithPerceptionResult]]

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        visualizer.objects_to_df(self.object_results)
        assert hasattr(visualizer, "visualize_df")
        assert "center_distance" in visualizer.visualize_df.columns
        assert "confidence" in visualizer.visualize_df.columns

    def test_get_subplots(self, save_dir: str):
        """[summary]
        Check if length of axes equals length of class names

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        axes = visualizer.get_subplots(self.class_names)
        assert len(axes) == len(self.class_names)

    def test_hist_object_count_for_each_distance_for_object_results(self, save_dir: str):
        """[summary]
        Check if file of hist_object_count_for_each_distance is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        visualizer.hist_object_count_for_each_distance(self.class_names, ranges_xy=self.ranges_xy)
        assert os.path.exists(save_dir + "/hist_object_count_for_each_distance.html")

    def test_hist_object_dist2d_for_each_class_for_object_results(self, save_dir: str):
        """[summary]
        Check if file of hist_object_dist2d_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        visualizer.hist_object_dist2d_for_each_class(self.class_names)
        assert os.path.exists(save_dir + "/hist_object_dist2d_for_each_class.html")

    def test_hist2d_object_wl_for_each_class_for_object_results(self, save_dir: str):
        """[summary]
        Check if file of hist2d_object_wl_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        visualizer.hist2d_object_wl_for_each_class(self.class_names)
        assert os.path.exists(save_dir + "/hist2d_object_wl_for_each_class.svg")

    def test_hist2d_object_center_xy_for_each_class_for_object_results(self, save_dir: str):
        """[summary]
        Check if file of hist2d_object_center_xy_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        ylim_dict = self.xylim_dict
        xlim_dict = self.xylim_dict
        visualizer.hist2d_object_center_xy_for_each_class(self.class_names, xlim_dict, ylim_dict)
        assert os.path.exists(save_dir + "/hist2d_object_center_xy_for_each_class.svg")

    def test_get_pandas_profiling_for_object_results(self, save_dir: str):
        """[summary]
        Check if file of pandas_profiling is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        file_name = "profiling_object_results"
        visualizer.get_pandas_profiling(self.class_names, file_name)
        assert os.path.exists(save_dir + "/" + file_name + "_all.html")
        for class_name in self.class_names:
            assert os.path.exists(save_dir + "/" + file_name + f"_{class_name}.html")

    def test_hist2d_object_num_points_for_each_class_for_object_results(self, save_dir: str):
        """[summary]
        Check if error raises when hist2d_object_num_points_for_each_class is used for List[DynamicObjectWithPerceptionResult]

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.object_results, save_dir)
        with pytest.raises(ValueError) as e:
            visualizer.hist2d_object_num_points_for_each_class(self.class_names)

        assert str(e.value) == "You should use this method only for ground truth objects"

    def test_objects_to_df_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if fields exist in DataFrame when input is List[DynamicObject]]

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """

        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        visualizer.objects_to_df(self.dummy_ground_truth_objects)
        assert hasattr(visualizer, "visualize_df")
        assert visualizer.is_gt
        assert "num_points" in visualizer.visualize_df.columns

    def test_hist_object_count_for_each_distance_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of hist_object_count_for_each_distance is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        visualizer.hist_object_count_for_each_distance(self.class_names, ranges_xy=self.ranges_xy)
        assert os.path.exists(save_dir + "/hist_object_count_for_each_distance.html")

    def test_hist_object_dist2d_for_each_class_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of hist_object_dist2d_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        visualizer.hist_object_dist2d_for_each_class(self.class_names)
        assert os.path.exists(save_dir + "/hist_object_dist2d_for_each_class.html")

    def test_hist2d_object_wl_for_each_class_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of hist2d_object_wl_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        visualizer.hist2d_object_wl_for_each_class(self.class_names)
        assert os.path.exists(save_dir + "/hist2d_object_wl_for_each_class.svg")

    def test_hist2d_object_center_xy_for_each_class_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of hist2d_object_center_xy_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        ylim_dict = self.xylim_dict
        xlim_dict = self.xylim_dict
        visualizer.hist2d_object_center_xy_for_each_class(self.class_names, xlim_dict, ylim_dict)
        assert os.path.exists(save_dir + "/hist2d_object_center_xy_for_each_class.svg")

    def test_hist2d_object_num_points_for_each_class_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of hist2d_object_num_points_for_each_class is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        visualizer.hist2d_object_num_points_for_each_class(self.class_names)
        assert os.path.exists(save_dir + "/hist2d_object_num_points_for_each_class.svg")

    def test_get_pandas_profiling_for_gt_objects(self, save_dir: str):
        """[summary]
        Check if file of pandas_profiling is generated.

        Args:
            save_dir (str):
                    save directory for each graph. If there is no directory in save_dir, make directory.
        """
        visualizer = EDAVisualizer(self.dummy_ground_truth_objects, save_dir)
        file_name = "profiling_ground_truth_objects"
        visualizer.get_pandas_profiling(self.class_names, file_name)
        assert os.path.exists(save_dir + "/" + file_name + "_all.html")
        for class_name in self.class_names:
            assert os.path.exists(save_dir + "/" + file_name + f"_{class_name}.html")


class TestEDAManager:
    dummy_estimated_objects: List[DynamicObject] = []
    dummy_ground_truth_objects: List[DynamicObject] = []
    dummy_estimated_objects, dummy_ground_truth_objects = make_dummy_data()

    dummy_estimated_objects.append(
        DynamicObject(
            unix_time=100,
            position=(-1.0, 1.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
            uuid=token_hex(16),
        ),
    )

    dummy_estimated_objects += get_objects_with_difference(
        ground_truth_objects=dummy_ground_truth_objects,
        diff_distance=(2.3, 0.0, 0.2),
        diff_yaw=0.2,
        is_confidence_with_distance=True,
    )

    dummy_ground_truth_objects += [
        DynamicObject(
            unix_time=100,
            position=(-1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(2.0, 4.0, 2.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(2.0, 4.0, 2.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.CAR,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 2.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 2.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.BICYCLE,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
        DynamicObject(
            unix_time=100,
            position=(-1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
        DynamicObject(
            unix_time=100,
            position=(1.0, 10.0, 1.0),
            orientation=Quaternion([0.0, 0.0, 0.0, 1.0]),
            size=(1.0, 1.0, 1.0),
            semantic_score=0.9,
            semantic_label=AutowareLabel.PEDESTRIAN,
            velocity=(1.0, 1.0, 1.0),
            pointcloud_num=10,
            uuid=token_hex(16),
        ),
    ]

    object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
        estimated_objects=dummy_estimated_objects,
        ground_truth_objects=dummy_ground_truth_objects,
    )

    root_path: str = tempfile.TemporaryDirectory().name

    class_names: List[str] = ["car", "pedestrian", "bicycle"]
    ranges_xy: List[float] = [250, 100, 50, 25]
    xylim_dict: Dict[str, List[float]] = {
        "car": [-100, 100],
        "bicycle": [-100, 100],
        "pedestrian": [-100, 100],
    }
    width_lim_dict: Dict[str, List[float]] = {
        "car": [0, 4.0],
        "bicycle": [0, 2.5],
        "pedestrian": [0, 1.6],
    }
    length_lim_dict: Dict[str, List[float]] = {
        "car": [0, 18],
        "bicycle": [0, 2.5],
        "pedestrian": [0, 2.5],
    }
    eda_manager = EDAManager(
        root_path, class_names, ranges_xy, xylim_dict, width_lim_dict, length_lim_dict
    )

    def test_visualize_ground_truth_objects(self) -> None:
        """[summary]
        Check if visualized results of ground_truth_objects are generated.
        """
        ground_truth_object_dict: Dict[str, List[DynamicObject]] = {
            "dummy_ground_truths": self.dummy_ground_truth_objects
        }

        self.eda_manager.visualize_ground_truth_objects(ground_truth_object_dict)

        for object_name in ground_truth_object_dict.keys():
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_count_for_each_distance.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_dist2d_for_each_class.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_wl_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_center_xy_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_num_points_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/" + "profiling_" + object_name + "_all.html"
            )
            for class_name in self.class_names:
                assert os.path.exists(
                    self.root_path
                    + "/"
                    + object_name
                    + "/"
                    + "profiling_"
                    + object_name
                    + f"_{class_name}.html"
                )

    def test_visualize_estimated_objects(self) -> None:
        """[summary]
        Check if visualized results of estimated_objects are generated.
        """
        estimated_object_dict: Dict[str, List[DynamicObjectWithPerceptionResult]] = {
            "object_results": self.object_results,
        }

        self.eda_manager.visualize_estimated_objects(estimated_object_dict)

        for object_name in estimated_object_dict.keys():
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_count_for_each_distance.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_dist2d_for_each_class.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_wl_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_center_xy_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/" + "profiling_" + object_name + "_all.html"
            )
            for class_name in self.class_names:
                assert os.path.exists(
                    self.root_path
                    + "/"
                    + object_name
                    + "/"
                    + "profiling_"
                    + object_name
                    + f"_{class_name}.html"
                )

    def test_visualize_evaluated_results(self) -> None:
        """[summary]
        Check if visualized TP, FP results, FP results with high confidence and FN ground_truth_objects are generated.
        """

        self.eda_manager.visualize_evaluated_results(
            self.object_results,
            self.dummy_ground_truth_objects,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold=0.5,
            confidence_threshold=0.7,
        )

        for object_name in [
            "tp_results",
            "fp_results",
            "fp_results_with_high_confidence",
            "fn_ground_truths",
        ]:
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_count_for_each_distance.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist_object_dist2d_for_each_class.html"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_wl_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/hist2d_object_center_xy_for_each_class.svg"
            )
            assert os.path.exists(
                self.root_path + "/" + object_name + "/" + "profiling_" + object_name + "_all.html"
            )
            for class_name in self.class_names:
                assert os.path.exists(
                    self.root_path
                    + "/"
                    + object_name
                    + "/"
                    + "profiling_"
                    + object_name
                    + f"_{class_name}.html"
                )

    def test_report_rates(self) -> None:
        """[summary]
        Check if output of report_rates is correct.
        """
        self.eda_manager.report_rates(
            tp_num=8, fp_num=12, estimated_objects_num=20, fn_num=2, ground_truth_num=10
        )
