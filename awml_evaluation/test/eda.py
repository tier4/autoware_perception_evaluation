import argparse
import tempfile
from typing import Dict
from typing import List
from typing import Union

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.label import LabelConverter
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.visualization.eda_tool import EDAVisualizer


def visualize_ground_truth(dataset_paths: List[str], save_path: str) -> None:
    """[summary]
    Visualize ground truth objects

    Args:
        dataset_paths (List[str]): path to dataset for visualization
        save_path (str): save directory for each graph. If there is no directory in save_dir, make directory.

    """

    """
    Settings
    """
    all_ground_truths: List[DynamicObject] = get_all_ground_truths(dataset_paths)
    class_names: List[str] = ["car", "pedestrian", "bicycle"]
    ranges_xy: List[Union[int, float]] = [250, 100, 50, 25]
    xylim_dict: Dict[str, List[float]] = {
        "car": [-100, 100],
        "bicycle": [-100, 100],
        "pedestrian": [-100, 100],
    }
    width_lim_dict: Dict[str, List[float]] = {
        "car": [0, 6.0],
        "bicycle": [0, 2.5],
        "pedestrian": [0, 1.5],
    }
    length_lim_dict: Dict[str, List[float]] = {
        "car": [0, 2.5],
        "bicycle": [0, 2.5],
        "pedestrian": [0, 1.5],
    }
    objects_name: str = "all_ground_truths"

    """
    Example of EDAVisualizer
    """
    visualizer = EDAVisualizer(all_ground_truths, save_path + "/" + objects_name)

    # Show histogram of number of objects that are less than the certain distance in x-y plane
    visualizer.hist_object_count_for_each_distance(class_names, ranges_xy=ranges_xy)

    # Show histogram of distance in x-y plane of objects
    visualizer.hist_object_dist2d_for_each_class(class_names)

    # Show 2d-histogram of width and length in each class
    visualizer.hist2d_object_wl_for_each_class(
        class_names, width_lim_dict=width_lim_dict, length_lim_dict=length_lim_dict
    )

    # Show 2d-histogram of x and y in each class
    visualizer.hist2d_object_center_xy_for_each_class(
        class_names, xlim_dict=xylim_dict, ylim_dict=xylim_dict
    )

    # Show 2d-histogram of number of point clouds in each class
    visualizer.hist2d_object_num_points_for_each_class(class_names)

    # Get pandas profiling report
    visualizer.get_pandas_profiling(class_names, "profiling_" + objects_name)

    print(f"visualize results are saved in {save_path}")


def get_all_ground_truths(dataset_paths: List[str]) -> List[DynamicObject]:
    """[summary]
    Get all ground truth objects

    Args:
        dataset_paths (List[str]): path to dataset for visualization

    Returns:
        all_ground_truths (List[DynamicObject]): all ground truth objects

    """
    frame_results: List[FrameGroundTruth] = load_all_datasets(
        dataset_paths,
        does_use_pointcloud=False,
        evaluation_task="detection",
        label_converter=LabelConverter(),
        frame_id="base_link",
    )
    all_ground_truths: List[DynamicObject] = []
    for frame_result in frame_results:
        all_ground_truths += frame_result.objects

    return all_ground_truths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_paths", nargs="+", type=str, help="The path(s) of dataset")
    parser.add_argument(
        "--save-path",
        type=str,
        required=False,
        help="Path to the directory for saving results. If this arg is empty, /tmp is used.",
    )
    args = parser.parse_args()

    dataset_paths: List[str] = args.dataset_paths

    if args.save_path:
        save_path: str = args.save_path
    else:
        tmpdir = tempfile.TemporaryDirectory()
        save_path: str = tmpdir.name

    visualize_ground_truth(dataset_paths, save_path=save_path)
