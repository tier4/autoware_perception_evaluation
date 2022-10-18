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

from perception_eval.common.object import DynamicObject
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager.perception_evaluation_manager import PerceptionEvaluationManager
from perception_eval.util.logger_config import configure_logger


class PerceptionLSimMoc:
    def __init__(
        self,
        dataset_paths: List[str],
        evaluation_task: str,
        result_root_directory: str,
    ):
        evaluation_config_dict = {
            # ラベル，max x/y，マッチング閾値 (detection/tracking/predictionで共通)
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            # objectごとにparamを設定
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            "iou_2d_thresholds": [0.5],
        }
        if evaluation_task == "detection2d":
            # detection
            frame_id: str = "base_link"  # objectのframe_id: base_link or map
            evaluation_config_dict.update({"evaluation_task": evaluation_task})
        else:
            raise ValueError(f"Unexpected evaluation task: {evaluation_task}")

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            merge_similar_labels=False,
            does_use_pointcloud=False,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
        )

        _ = configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

    def callback(
        self,
        unix_time: int,
        estimated_objects: List[DynamicObject],
    ) -> None:

        # 現frameに対応するGround truthを取得
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)

        # [Option] ROS側でやる（Map情報・Planning結果を用いる）UC評価objectを選別
        # ros_critical_ground_truth_objects : List[DynamicObject] = custom_critical_object_filter(
        #   ground_truth_now_frame.objects
        # )
        ros_critical_ground_truth_objects = ground_truth_now_frame.objects

        # 1 frameの評価
        # 距離などでUC評価objectを選別するためのインターフェイス（PerceptionEvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        # どれを注目物体とするかのparam
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            max_x_position_list=[30.0, 30.0, 30.0, 30.0],
            max_y_position_list=[30.0, 30.0, 30.0, 30.0],
        )
        # Pass fail を決めるパラメータ
        frame_pass_fail_config: PerceptionPassFailConfig = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            plane_distance_threshold_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        self.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """
        final_metric_score = self.evaluator.get_scene_result()

        # final result
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    def visualize(self, frame_result: PerceptionFrameResult):
        """
        Frameごとの可視化
        """
        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")


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

    # ========================================= Detection =========================================
    print("=" * 50 + "Start Detection" + "=" * 50)
    detection_lsim = PerceptionLSimMoc(dataset_paths, "detection2d", result_root_directory)

    for ground_truth_frame in detection_lsim.evaluator.ground_truth_frames:
        objects_with_difference = ground_truth_frame.objects
        # To avoid case of there is no object
        if len(objects_with_difference) > 0:
            objects_with_difference.pop(0)
        detection_lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )

    # final result
    detection_final_metric_score = detection_lsim.get_final_result()
