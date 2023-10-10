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

import argparse
import tempfile
from typing import Dict, List, Union

from perception_eval.common.dataset import FrameGroundTruth, load_all_datasets
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelConverter
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.visualization.eda_tool import EDAVisualizer


def visualize_ground_truth(dataset_paths: List[str], save_path: str, show: bool) -> None:
    """[summary]
    Visualize ground truth objects.

    Args:
    ----
        dataset_paths (List[str]): path to dataset for visualization
        save_path (str): save directory for each graph. If there is no directory in save_dir, make directory.
        show (bool): Whether show visualized figures.
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
    visualizer = EDAVisualizer(all_ground_truths, save_path + "/" + objects_name, show)

    # Show histogram of number of objects that are less than the certain distance in x-y plane
    visualizer.hist_object_count_for_each_distance(class_names, ranges_xy=ranges_xy)

    # Show histogram of distance in x-y plane of objects
    visualizer.hist_object_dist2d_for_each_class(class_names)

    # Show 2d-histogram of width and length in each class
    visualizer.hist2d_object_wl_for_each_class(
        class_names,
        width_lim_dict=width_lim_dict,
        length_lim_dict=length_lim_dict,
    )

    # Show 2d-histogram of x and y in each class
    visualizer.hist2d_object_center_xy_for_each_class(class_names, xlim_dict=xylim_dict, ylim_dict=xylim_dict)

    # Show 2d-histogram of number of point clouds in each class
    visualizer.hist2d_object_num_points_for_each_class(class_names)

    # Get pandas profiling report
    visualizer.get_pandas_profiling(class_names, "profiling_" + objects_name)

    print(f"visualize results are saved in {save_path}")


def get_all_ground_truths(dataset_paths: List[str]) -> List[DynamicObject]:
    """[summary]
    Get all ground truth objects.

    Args:
    ----
        dataset_paths (List[str]): path to dataset for visualization

    Returns:
    -------
        all_ground_truths (List[DynamicObject]): all ground truth objects

    """
    evaluation_task = EvaluationTask.DETECTION
    frame_results: List[FrameGroundTruth] = load_all_datasets(
        dataset_paths,
        evaluation_task=evaluation_task,
        label_converter=LabelConverter(
            evaluation_task,
            merge_similar_labels=False,
            label_prefix="autoware",
        ),
        frame_id=FrameID.BASE_LINK,
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether show visualized result",
    )
    args = parser.parse_args()

    dataset_paths: List[str] = args.dataset_paths

    if args.save_path:
        save_path: str = args.save_path
    else:
        tmpdir = tempfile.TemporaryDirectory()
        save_path: str = tmpdir.name

    visualize_ground_truth(dataset_paths, save_path=save_path, show=args.show)
