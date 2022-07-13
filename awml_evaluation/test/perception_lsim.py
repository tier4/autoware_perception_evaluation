import argparse
import logging
import tempfile
from typing import List

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.config.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.manager.perception_evaluation_manager import PerceptionEvaluationManager
from awml_evaluation.util.debug import format_class_for_log
from awml_evaluation.util.debug import get_objects_with_difference
from awml_evaluation.util.logger_config import configure_logger

# tmp: logger-handlerが複数できないようにする変数
has_logger: bool = False


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
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            # objectごとにparamを設定
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            # objectごとに同じparamの場合はこのような指定が可能
            "plane_distance_thresholds": [2.0, 3.0],
            "iou_bev_thresholds": [0.5],
            "iou_3d_thresholds": [0.5],
        }
        if evaluation_task == "detection":
            # detection
            frame_id: str = "base_link"  # objectのframe_id: base_link or map
            # evaluation_task指定 + 今後各taskで異なるパラメータが入るかも
            evaluation_config_dict.update({"evaluation_task": evaluation_task})
            evaluation_config_dict.update({"min_point_numbers": [0, 0, 0, 0]})
        elif evaluation_task == "tracking":
            # tracking
            frame_id: str = "map"
            evaluation_config_dict.update({"evaluation_task": evaluation_task})
        else:
            raise ValueError(f"Unexpected evaluation task: {evaluation_task}")
        # TODO prediction

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            does_use_pointcloud=False,
            result_root_directory=result_root_directory,
            log_directory="",
            visualization_directory="visualization/",
            evaluation_config_dict=evaluation_config_dict,
        )

        global has_logger
        if has_logger is False:
            _ = configure_logger(
                log_file_directory=evaluation_config.get_result_log_directory(),
                console_log_level=logging.INFO,
                file_log_level=logging.INFO,
            )
            has_logger = True

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
            threshold_plane_distance_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        PerceptionLSimMoc.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """

        # use case fail object num
        number_use_case_fail_object: int = 0
        for frame_results in self.evaluator.frame_results:
            number_use_case_fail_object += frame_results.pass_fail_result.get_fail_object_num()
        logging.info(f"final use case fail object: {number_use_case_fail_object}")
        final_metric_score = self.evaluator.get_scene_result()

        # final result
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    @staticmethod
    def visualize(frame_result: PerceptionFrameResult):
        """
        Frameごとの可視化
        """
        if frame_result.pass_fail_result.get_fail_object_num() > 0:
            logging.warning(
                f"{len(frame_result.pass_fail_result.tp_objects)} TP objects, "
                f"{len(frame_result.pass_fail_result.fp_objects_result)} FP objects, "
                f"{len(frame_result.pass_fail_result.fn_objects)} FN objects,",
            )
            # logging.debug(f"frame result {format_class_for_log(frame_result.pass_fail_result)}")
        else:
            logging.info("No TP/FP/FN objects")

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")


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
            diff_distance=(2.3, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
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
            f"{format_class_for_log(detection_lsim.evaluator.frame_results[0], 5)}",
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

    # ========================================= Tracking =========================================
    print("=" * 50 + "Start Tracking" + "=" * 50)
    tracking_lsim = PerceptionLSimMoc(dataset_paths, "tracking", result_root_directory)

    for ground_truth_frame in tracking_lsim.evaluator.ground_truth_frames:
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(2.3, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
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
            f"{format_class_for_log(tracking_lsim.evaluator.frame_results[0], 5)}",
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

    # Clean up tmpdir
    if args.use_tmpdir:
        tmpdir.cleanup()
