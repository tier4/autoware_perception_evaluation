# Copyright 2023 TIER IV, Inc.

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
import logging
import tempfile
from typing import List

from perception_eval.common import ObjectType
from perception_eval.common.label import AutowareLabel
from perception_eval.common.status import get_scene_rates
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import get_object_status
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.util.debug import get_objects_with_difference
from perception_eval.util.logger_config import configure_logger


class FPValidationLsimMoc:
    def __init__(self, dataset_paths: List[int], result_root_directory: str) -> None:
        evaluation_config_dict = {
            "evaluation_task": "fp_validation",
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            "max_matchable_radii": [5.0],
            "merge_similar_labels": False,
            "label_prefix": "autoware",
            "load_raw_data": True,
            "allow_matching_unknown": True,
        }

        evaluation_config = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link",
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
        )

        configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = PerceptionEvaluationManager(evaluation_config)

    def callback(self, unix_time: int, estimated_objects: List[ObjectType]) -> None:
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)

        # Ideally, critical GT should be obtained in each frame.
        # In this mock, set it as a copy of `ground_truth_now_frame`.
        ros_critical_ground_truth_objects = ground_truth_now_frame.objects

        critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            max_x_position_list=[100.0, 100.0, 100.0, 100.0],
            max_y_position_list=[100.0, 100.0, 100.0, 100.0],
        )

        frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            matching_threshold_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        self.display(frame_result)

    def display(self, frame_result: PerceptionFrameResult) -> None:
        """Display TP/FP/FN information."""

        logging.info(
            f"{len(frame_result.pass_fail_result.tp_object_results)} TP objects, "
            f"{len(frame_result.pass_fail_result.fp_object_results)} FP objects, "
            f"{len(frame_result.pass_fail_result.tn_objects)} TN objects, "
            f"{len(frame_result.pass_fail_result.fn_objects)} FN objects",
        )

    def display_status_rates(self) -> None:
        status_list = get_object_status(self.evaluator.frame_results)
        for status_info in status_list:
            tp_rate, fp_rate, tn_rate, fn_rate = status_info.get_status_rates()
            # display
            logging.info(
                f"uuid: {status_info.uuid}, "
                # display TP/FP/TN/FN rates per frames
                f"TP: {tp_rate.rate:0.3f}, "
                f"FP: {fp_rate.rate:0.3f}, "
                f"TN: {tn_rate.rate:0.3f}, "
                f"FN: {fn_rate.rate:0.3f}\n"
                # display total or TP/FP/TN/FN frame numbers
                f"Total: {status_info.total_frame_nums}, "
                f"TP: {status_info.tp_frame_nums}, "
                f"FP: {status_info.fp_frame_nums}, "
                f"TN: {status_info.tn_frame_nums}, "
                f"FN: {status_info.fn_frame_nums}",
            )

        scene_tp_rate, scene_fp_rate, scene_tn_rate, scene_fn_rate = get_scene_rates(status_list)
        logging.info(
            "[scene]"
            f"TP: {scene_tp_rate}, "
            f"FP: {scene_fp_rate}, "
            f"TN: {scene_tn_rate}, "
            f"FN: {scene_fn_rate}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_paths", nargs="+", type=str, help="The path(s) of dataset")
    parser.add_argument(
        "--use_tmpdir",
        action="store_true",
        help="Whether save results to temporal directory",
    )
    args = parser.parse_args()

    dataset_paths = args.dataset_paths
    if args.use_tmpdir:
        tmpdir = tempfile.TemporaryDirectory()
        result_root_directory: str = tmpdir.name
    else:
        result_root_directory: str = "data/result/{TIME}/"

    # ========================================= FP validation =========================================
    print("=" * 50 + "Start FP validation" + "=" * 50)
    fp_validation_lsim = FPValidationLsimMoc(dataset_paths, result_root_directory)

    for ground_truth_frame in fp_validation_lsim.evaluator.ground_truth_frames:
        # Because FP label is contained in GT, updates them to the other labels
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(1.0, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
            ego2map=ground_truth_frame.ego2map,
            label_candidates=[
                AutowareLabel.CAR,
                AutowareLabel.BICYCLE,
                AutowareLabel.PEDESTRIAN,
                AutowareLabel.MOTORBIKE,
            ],
        )
        if len(objects_with_difference) > 0:
            objects_with_difference.pop(0)
        fp_validation_lsim.callback(ground_truth_frame.unix_time, objects_with_difference)

    fp_validation_lsim.display_status_rates()
