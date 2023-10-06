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
import logging
import tempfile
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.config import SensingEvaluationConfig
from perception_eval.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from perception_eval.evaluation.sensing.sensing_frame_result import SensingFrameResult
from perception_eval.manager import SensingEvaluationManager
from perception_eval.util.logger_config import configure_logger


class SensingLSimMoc:
    """Moc to evaluate sensing in driving_log_replayer.

    Args:
        dataset_paths: List[str]: The list of dataset paths.
        result_root_directory (str): The directory path to save results.
    """

    def __init__(self, dataset_paths: List[str], result_root_directory: str):
        # sensing
        evaluation_config_dict = {
            "evaluation_task": "sensing",
            # object uuids to be detected
            # "target_uuids": ["1b40c0876c746f96ac679a534e1037a2"],
            "target_uuids": None,
            # The scale factor for boxes at 0 and 100[m]
            "box_scale_0m": 1.0,
            "box_scale_100m": 1.0,
            "min_points_threshold": 1,
        }

        evaluation_config: SensingEvaluationConfig = SensingEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link",
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=True,
        )

        _ = configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = SensingEvaluationManager(evaluation_config=evaluation_config)

    def callback(
        self,
        unix_time: int,
        pointcloud: np.ndarray,
        non_detection_areas: List[List[Tuple[float, float, float]]],
    ) -> SensingFrameResult:
        """[summary]

        Args:
            unix_time (int): Unix time [us]
            pointcloud (numpy.ndarray): Array of pointcloud after removing ground
            non_detection_areas (List[List[[Tuple[float, float, float]]]):
                The list of 3D-polygon areas for non-detection.

        Returns:
            frame_result (SensingFrameResult): Result per frame.
        """
        ground_truth_now_frame: Optional[FrameGroundTruth] = self.evaluator.get_ground_truth_now_frame(
            unix_time=unix_time
        )

        # Evaluation config for one frame.
        # If not specified, params of SensingEvaluationConfig will be used.
        sensing_frame_config = SensingFrameConfig(
            target_uuids=None,
            box_scale_0m=1.0,
            box_scale_100m=1.0,
            min_points_threshold=1,
        )

        if ground_truth_now_frame is not None:
            frame_result: SensingFrameResult = self.evaluator.add_frame_result(
                unix_time=unix_time,
                ground_truth_now_frame=ground_truth_now_frame,
                pointcloud=pointcloud,
                non_detection_areas=non_detection_areas,
                sensing_frame_config=sensing_frame_config,
            )

            self.visualize(frame_result)

        return frame_result

    def get_final_result(self) -> None:
        """Output the evaluation results on the command line"""
        # use case fail object num
        num_use_case_fail: int = 0
        for frame_results in self.evaluator.frame_results:
            num_use_case_fail += len(frame_results.detection_fail_results)
        logging.warning(f"{num_use_case_fail} fail results.")

    @staticmethod
    def visualize(frame_result: SensingFrameResult) -> None:
        """Visualize results per frame
        Args:
            frame_result (SensingFrameResult)
        """
        if len(frame_result.detection_fail_results) > 0:
            logging.warning(f"Fail {len(frame_result.detection_fail_results)} detection.")
            for fail_result in frame_result.detection_fail_results:
                logging.info(
                    f"[FAIL] Inside points: {fail_result.inside_pointcloud_num}, Is detected: {fail_result.is_detected}"
                )
        elif len(frame_result.detection_warning_results) > 0:
            logging.warning(f"Warning {len(frame_result.detection_warning_results)} detection.")
            for warning_result in frame_result.detection_warning_results:
                logging.info(
                    f"[WARNING] Inside points: {warning_result.inside_pointcloud_num}, "
                    f"Is detected: {warning_result.is_detected}, "
                    f"Is occluded: {warning_result.is_occluded}"
                )
        else:
            logging.info("all detections were succeeded.")

        for success_result in frame_result.detection_success_results:
            logging.info(
                f"[SUCCESS] Inside points: {success_result.inside_pointcloud_num}, "
                f"Is detected: {success_result.is_detected}, "
                f"Nearest point: {success_result.nearest_point}\n"
            )

        if len(frame_result.pointcloud_failed_non_detection) > 0:
            logging.warning(
                f"The number of Failed non-detection pointcloud: {len(frame_result.pointcloud_failed_non_detection)}"
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

    sensing_lsim = SensingLSimMoc(dataset_paths, result_root_directory)

    non_detection_areas: List[List[Tuple[float, float, float]]] = [
        [
            # lower plane
            (10.0, 1.5, 0.0),
            (10, -1.5, 0.0),
            (0.0, -1.5, 0.0),
            (0.0, 1.5, 0.0),
            # upper plane
            (10.0, 1.5, 4.0),
            (10, -1.5, 4.0),
            (0.0, -1.5, 4.0),
            (0.0, 1.5, 4.0),
        ],
    ]
    num_frames = len(sensing_lsim.evaluator.ground_truth_frames)
    for ground_truth_frame in sensing_lsim.evaluator.ground_truth_frames:
        frame_result: SensingFrameResult = sensing_lsim.callback(
            ground_truth_frame.unix_time,
            ground_truth_frame.raw_data["lidar"],
            non_detection_areas,
        )

    # final result
    final_sensing_score = sensing_lsim.get_final_result()

    # Debug
    logging.info(
        "Frame result example (frame_results[0]: "
        f"{len(sensing_lsim.evaluator.frame_results[0].detection_success_results)} success, "
        f"{len(sensing_lsim.evaluator.frame_results[0].detection_fail_results)} fail"
    )

    logging.info(
        "Failed to be Non-detected pointcloud example (frame_results[0]): "
        f"{len(sensing_lsim.evaluator.frame_results[0].pointcloud_failed_non_detection)}"
    )

    # Visualize all frame results
    logging.info("Start visualizing sensing results")
    sensing_lsim.evaluator.visualize_all()

    # Clean up tmpdir
    if args.use_tmpdir:
        tmpdir.cleanup()
