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

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.object import DynamicObject
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.tool import PerceptionAnalyzer3D
from perception_eval.util.debug import format_class_for_log
from perception_eval.util.debug import get_objects_with_difference
from perception_eval.util.logger_config import configure_logger


class PerceptionLSimMoc:
    def __init__(
        self,
        dataset_paths: List[str],
        evaluation_task: str,
        result_root_directory: str,
    ):
        evaluation_config_dict = {
            "evaluation_task": evaluation_task,
            # ラベル，max x/y，マッチング閾値 (detection/tracking/predictionで共通)
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            "ignore_attributes": ["cycle_state.without_rider"],
            # max x/y position or max/min distanceの指定が必要
            # # max x/y position
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            # max/min distance
            # "max_distance": 102.4,
            # "min_distance": 10.0,
            # # confidenceによるフィルタ (Optional)
            # "confidence_threshold": 0.5,
            # # GTのuuidによるフィルタ (Optional)
            # "target_uuids": ["foo", "bar"],
            # objectごとにparamを設定
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],  # = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
            "center_distance_bev_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],  # = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
            # objectごとに同じparamの場合はこのような指定が可能
            "plane_distance_thresholds": [
                2.0,
                3.0,
            ],  # = [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
            "iou_2d_thresholds": [0.5, 0.5, 0.5, 0.5],  # = [[0.5, 0.5, 0.5, 0.5]]
            "iou_3d_thresholds": [0.5],  # = [[0.5, 0.5, 0.5, 0.5]]
            "min_point_numbers": [0, 0, 0, 0],
            "max_matchable_radii": 5.0,  # = [5.0, 5.0, 5.0, 5.0]
            # label parameters
            "label_prefix": "autoware",
            "merge_similar_labels": False,
            "allow_matching_unknown": True,
        }

        evaluation_config = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link" if evaluation_task == "detection" else "map",
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=True,
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
        interpolate = not self.evaluator.evaluation_task == EvaluationTask.DETECTION
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(
            unix_time, interpolate_ground_truth=interpolate
        )

        # 1 frameの評価
        # 距離などでUC評価objectを選別するためのインターフェイス（PerceptionEvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        # どれを注目物体とするかのparam
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            ignore_attributes=["cycle_state.without_rider"],
            max_x_position_list=[30.0, 30.0, 30.0, 30.0],
            max_y_position_list=[30.0, 30.0, 30.0, 30.0],
        )
        # Pass fail を決めるパラメータ
        frame_pass_fail_config: PerceptionPassFailConfig = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            matching_threshold_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        self.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """

        # number of fails for critical objects
        num_critical_fail: int = sum(
            map(
                lambda frame_result: frame_result.pass_fail_result.get_num_fail(),
                self.evaluator.frame_results,
            )
        )
        logging.info(f"Number of fails for critical objects: {num_critical_fail}")

        # scene metrics score
        final_metric_score = self.evaluator.get_scene_result()
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    def visualize(self, frame_result: PerceptionFrameResult):
        """
        Frameごとの可視化
        """
        logging.info(
            f"{len(frame_result.pass_fail_result.tp_object_results)} TP objects, "
            f"{len(frame_result.pass_fail_result.fp_object_results)} FP objects, "
            f"{len(frame_result.pass_fail_result.fn_objects)} FN objects",
        )

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")

        # Visualize the latest frame result
        # self.evaluator.visualize_frame()


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
    detection_lsim = PerceptionLSimMoc(dataset_paths, "detection", result_root_directory)

    for ground_truth_frame in detection_lsim.evaluator.ground_truth_frames:
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(1.0, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
            label_to_unknown_rate=0.5,
            transforms=ground_truth_frame.transforms,
        )
        # To avoid case of there is no object
        if len(objects_with_difference) > 0:
            objects_with_difference.pop(0)
        detection_lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )

    # final result
    detection_final_metric_score = detection_lsim.get_final_result()

    # Debug
    if len(detection_lsim.evaluator.frame_results) > 0:
        logging.info(
            "Frame result example (frame_results[0]): "
            f"{format_class_for_log(detection_lsim.evaluator.frame_results[0], 1)}",
        )

        if len(detection_lsim.evaluator.frame_results[0].object_results) > 0:
            logging.info(
                "Object result example (frame_results[0].object_results[0]): "
                f"{format_class_for_log(detection_lsim.evaluator.frame_results[0].object_results[0])}",
            )

    # Metrics config
    logging.info(
        "Detection Metrics example (final_metric_score): "
        f"{format_class_for_log(detection_final_metric_score, len(detection_final_metric_score.detection_config.target_labels))}",
    )

    # Detection metrics score
    logging.info(
        "mAP result example (final_metric_score.maps[0].aps[0]): "
        f"{format_class_for_log(detection_final_metric_score.maps[0], 100)}",
    )

    if detection_lsim.evaluator.evaluator_config.load_raw_data:
        # Visualize all frame results.
        logging.info("Start visualizing detection results")
        detection_lsim.evaluator.visualize_all()

    # Detection performance report
    detection_analyzer = PerceptionAnalyzer3D(detection_lsim.evaluator.evaluator_config)
    detection_analyzer.add(detection_lsim.evaluator.frame_results)
    result = detection_analyzer.analyze()
    if result.score is not None:
        logging.info(result.score.to_string())
    if result.error is not None:
        logging.info(result.error.to_string())
    if result.confusion_matrix is not None:
        logging.info(result.confusion_matrix.to_string())

    # detection_analyzer.plot_state("4bae7e75c7de70be980ce20ce8cbb642", ["x", "y"])
    # detection_analyzer.plot_error(["x", "y"])
    # detection_analyzer.plot_num_object()
    # detection_analyzer.box_plot()

    # ========================================= Tracking =========================================
    print("=" * 50 + "Start Tracking" + "=" * 50)
    if args.use_tmpdir:
        tmpdir = tempfile.TemporaryDirectory()
        result_root_directory: str = tmpdir.name
    else:
        result_root_directory: str = "data/result/{TIME}/"
    tracking_lsim = PerceptionLSimMoc(dataset_paths, "tracking", result_root_directory)

    for ground_truth_frame in tracking_lsim.evaluator.ground_truth_frames:
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(1.0, 0.0, 0.0),
            diff_yaw=0.0,
            is_confidence_with_distance=True,
            label_to_unknown_rate=0.5,
            transforms=ground_truth_frame.transforms,
        )
        # To avoid case of there is no object
        if len(objects_with_difference) > 0:
            objects_with_difference.pop(0)
        tracking_lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )

    # final result
    tracking_final_metric_score = tracking_lsim.get_final_result()

    # Debug
    if len(tracking_lsim.evaluator.frame_results) > 0:
        logging.info(
            "Frame result example (frame_results[0]): "
            f"{format_class_for_log(tracking_lsim.evaluator.frame_results[0], 1)}",
        )

        if len(tracking_lsim.evaluator.frame_results[0].object_results) > 0:
            logging.info(
                "Object result example (frame_results[0].object_results[0]): "
                f"{format_class_for_log(tracking_lsim.evaluator.frame_results[0].object_results[0])}",
            )

    # Metrics config
    logging.info(
        "Tracking Metrics example (tracking_final_metric_score): "
        f"{format_class_for_log(tracking_final_metric_score, len(tracking_final_metric_score.detection_config.target_labels))}",
    )

    # Detection metrics score in Tracking
    logging.info(
        "mAP result example (tracking_final_metric_score.maps[0].aps[0]): "
        f"{format_class_for_log(tracking_final_metric_score.maps[0], 100)}",
    )

    # Tracking metrics score
    logging.info(
        "CLEAR result example (tracking_final_metric_score.tracking_scores[0].clears[0]): "
        f"{format_class_for_log(tracking_final_metric_score.tracking_scores[0], 100)}"
    )

    if tracking_lsim.evaluator.evaluator_config.load_raw_data:
        # Visualize all frame results
        logging.info("Start visualizing tracking results")
        tracking_lsim.evaluator.visualize_all()

    # Tracking performance report
    tracking_analyzer = PerceptionAnalyzer3D(tracking_lsim.evaluator.evaluator_config)
    tracking_analyzer.add(tracking_lsim.evaluator.frame_results)
    result = tracking_analyzer.analyze()
    if result.score is not None:
        logging.info(result.score.to_string())
    if result.error is not None:
        logging.info(result.error.to_string())
    if result.confusion_matrix is not None:
        logging.info(result.confusion_matrix.to_string())

    # Clean up tmpdir
    if args.use_tmpdir:
        tmpdir.cleanup()
