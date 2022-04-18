import logging
from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
from awml_evaluation.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.perception_evaluation_manager import PerceptionEvaluationManager
from awml_evaluation.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.sensing_evaluation_manager import SensingEvaluationManager
from awml_evaluation.util.debug import format_class_for_log
from awml_evaluation.util.debug import get_objects_with_difference
from awml_evaluation.util.logger_config import configure_logger
import numpy as np


class PerceptionLSimMoc:
    def __init__(self, dataset_paths: List[str]):
        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            does_use_pointcloud=False,
            result_root_directory="data/result/{TIME}/",
            log_directory="",
            visualization_directory="visualization/",
            evaluation_tasks=["detection"],
            # target_labels=["car", "truck", "bicycle", "pedestrian", "motorbike"],
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            max_x_position=102.4,
            max_y_position=102.4,
            # objectごとにparamを設定
            map_thresholds_center_distance=[
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            # objectごとに同じparamの場合はこのような指定が可能
            map_thresholds_plane_distance=[2.0, 3.0],
            map_thresholds_iou_bev=[0.5],
            map_thresholds_iou_3d=[0.5],
        )
        _ = configure_logger(
            log_file_directory=evaluation_config.get_result_log_directory(),
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )
        self.evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

    def callback(
        self,
        unix_time: int,
        predicted_objects: List[DynamicObject],
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

        frame_result = self.evaluator.add_perception_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            predicted_objects=predicted_objects,
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
                f"{len(frame_result.pass_fail_result.fp_objects_result)} FP objects, "
                f"{len(frame_result.pass_fail_result.fn_objects)} FN objects,",
            )
            # logging.debug(f"frame result {format_class_for_log(frame_result.pass_fail_result)}")

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")


class SensingLSimMoc:
    """Moc to evaluate sensing in LogSim.

    Args:
        dataset_paths: List[str]: The list of dataset paths.
    """

    def __init__(self, dataset_paths: List[str]):
        evaluation_config: SensingEvaluationConfig = SensingEvaluationConfig(
            dataset_paths=dataset_paths,
            does_use_pointcloud=False,
            result_root_directory="data/result/{TIME}/",
            log_directory="",
            visualization_directory="visualization/",
            # object uuids to be detected
            target_uuids=["1b40c0876c746f96ac679a534e1037a2"],
            # The scale factor for boxes at 0 and 100[m]
            box_scale_0m=1.0,
            box_scale_100m=1.0,
            min_points_threshold=1,
        )

        # NOTE: If confiure_logger() is called multi time, same components are logged in multi line.
        # _ = configure_logger(
        #     log_file_directory=evaluation_config.get_result_log_directory(),
        #     console_log_level=logging.INFO,
        #     file_log_level=logging.INFO,
        # )

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
        ground_truth_now_frame: FrameGroundTruth = self.evaluator.get_ground_truth_now_frame(
            unix_time=unix_time,
        )

        pointcloud_for_non_detection: List[np.ndarray] = self.evaluator.crop_pointcloud(
            pointcloud=pointcloud,
            non_detection_areas=non_detection_areas,
        )

        frame_result: SensingFrameResult = self.evaluator.add_sensing_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            pointcloud_for_detection=pointcloud,
            pointcloud_for_non_detection=pointcloud_for_non_detection,
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
        else:
            logging.info("all detections were succeeded.")

        for success_result in frame_result.detection_success_results:
            logging.info(
                f"[SUCCESS] Inside points: {success_result.inside_pointcloud_num}, Is detected: {success_result.is_detected}"
            )

        if len(frame_result.pointcloud_failed_non_detection) > 0:
            logging.warn(
                f"The number of Failed non-detection pointcloud: {len(frame_result.pointcloud_failed_non_detection)}"
            )


if __name__ == "__main__":
    # dataset_paths = [
    #     "../../dataset_3d/tier4/202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475",
    #     "../../dataset_3d/tier4/202108_3d_cuboid_v1_1_1_nishishinjuku_mini/5f772b2ca6ace800391c3e74",
    # ]
    dataset_paths = [
        "../../dataset_3d/tier4/sensing_lsim_data",
    ]
    perception_lsim = PerceptionLSimMoc(dataset_paths)

    for ground_truth_frame in perception_lsim.evaluator.ground_truth_frames:
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(2.3, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
        )
        # To avoid case of there is no object
        if len(objects_with_difference) > 0:
            objects_with_difference.pop(0)
        perception_lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )

    # final result
    final_metric_score = perception_lsim.get_final_result()

    # Debug
    if len(perception_lsim.evaluator.frame_results) > 0:
        logging.info(
            "Frame result example (frame_results[0]): "
            f"{format_class_for_log(perception_lsim.evaluator.frame_results[0], 5)}",
        )

        if len(perception_lsim.evaluator.frame_results[0].object_results) > 0:
            logging.info(
                "Object result example (frame_results[0].object_results[0]): "
                f"{format_class_for_log(perception_lsim.evaluator.frame_results[0].object_results[0])}",
            )

    logging.info(
        "Metrics example (final_metric_score): "
        f"{format_class_for_log(final_metric_score, len(final_metric_score.config.target_labels))}",
    )

    logging.info(
        "mAP result example (final_metric_score.maps[0].aps[0]): "
        f"{format_class_for_log(final_metric_score.maps[0], 100)}",
    )

    sensing_lsim = SensingLSimMoc(dataset_paths)

    non_detection_areas: List[List[Tuple[float, float, float]]] = [
        [
            # lower plane
            (1.0, 1.0, 0.5),
            (100.0, 1.0, 0.5),
            (100.0, -1.0, 0.5),
            (1.0, -1.0, 0.5),
            # upper plane
            (1.0, 1.0, 2.0),
            (100.0, 1.0, 2.0),
            (100.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
        ],
    ]
    num_frames = len(sensing_lsim.evaluator.ground_truth_frames)
    pointcloud_frames = np.random.rand(num_frames, 100, 3) * 10
    for ground_truth_frame, pointcloud in zip(
        sensing_lsim.evaluator.ground_truth_frames,
        pointcloud_frames,
    ):
        frame_result: SensingFrameResult = sensing_lsim.callback(
            ground_truth_frame.unix_time,
            pointcloud,
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
        "Failed to be Non-detected pointclouds example (frame_results[0]): "
        f"{len(sensing_lsim.evaluator.frame_results[0].pointcloud_failed_non_detection)}"
    )
