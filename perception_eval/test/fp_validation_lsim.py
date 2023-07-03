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

import logging
from typing import List
from typing import Optional

from perception_eval.common import ObjectType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.util.logger_config import configure_logger


class FPValidationLsimMoc:
    def __init__(self, dataset_paths: List[int], result_root_directory: str) -> None:
        evaluation_config_dict = {
            "evaluation_task": "fp_validation",
            "target_labels": ["fp"],
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            "min_point_numbers": [0],
        }

        evaluation_config = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link",
            merge_similar_labels=False,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=True,
        )

        configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = PerceptionEvaluationManager(evaluation_config)

    def callback(self, unix_time: int, estimated_objects: ObjectType) -> None:
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)

        # Ideally, critical GT should be obtained in each frame.
        # In this mock, set it as a copy of `ground_truth_now_frame`.
        ros_critical_ground_truth_objects = ground_truth_now_frame

        critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["fp"],
            max_x_position_list=[100.0],
            max_y_position_list=[100.0],
        )

        frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["fp"],
            matching_threshold_list=[2.0],
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
            f"{len(frame_result.pass_fail_result.fn_objects)} FN objects",
        )

        if len(frame_result.pass_fail_result.fp_object_results) < 0:
            return

        for fp_result in frame_result.pass_fail_result.fp_object_results:
            est_uuid: str = fp_result.estimated_object.uuid
            gt_uuid: Optional[str] = (
                None
                if fp_result.ground_truth_object is None
                else fp_result.ground_truth_object.uuid
            )
            logging.info(f"Estimation: {est_uuid}, GT: {gt_uuid}")

    def display_final_result(self) -> None:
        """Display the number of FP result, the ratio of FP result for each object identified by uuid."""

        object_fp_results = {}
        for frame_result in self.evaluator.frame_results:
            for fp_result in frame_result.pass_fail_result.fp_object_results:
                fp_estimated_object = fp_result.estimated_object
                est_uuid = fp_estimated_object.uuid
                fp_ground_truth_object = fp_result.ground_truth_object
                gt_uuid = None if fp_ground_truth_object is None else fp_ground_truth_object.uuid
                pass

        for uuid, objects in object_fp_results.items():
            pass
