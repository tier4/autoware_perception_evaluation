import logging
from typing import List

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.perception_evaluation_manager import PerceptionEvaluationManager
from awml_evaluation.util.debug import format_class_for_log
from awml_evaluation.util.debug import get_objects_with_difference
from awml_evaluation.util.logger_config import configure_logger


class LSimMoc:
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
        LSimMoc.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """

        # use case fail object num
        number_use_case_fail_object: int = 0
        for frame_results in self.evaluator.frame_results:
            number_use_case_fail_object += frame_results.pass_fail_result.get_fail_object_num()
        logging.info(f"final use case fail object: {number_use_case_fail_object}")
        final_metric_score = self.evaluator.get_scenario_result()

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
                f"{len(frame_result.pass_fail_result.fp_objects_result)} FP objects, {len(frame_result.pass_fail_result.fn_objects)} FN objects,"
            )
            # logging.debug(f"frame result {format_class_for_log(frame_result.pass_fail_result)}")

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")


if __name__ == "__main__":
    # dataset_paths = [
    #     "../../dataset_3d/tier4/202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475",
    #     "../../dataset_3d/tier4/202108_3d_cuboid_v1_1_1_nishishinjuku_mini/5f772b2ca6ace800391c3e74",
    # ]
    dataset_paths = [
        "../../dataset_3d/tier4/202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475",
    ]
    lsim = LSimMoc(dataset_paths)

    for ground_truth_frame in lsim.evaluator.ground_truth_frames:
        objects_with_difference = get_objects_with_difference(
            ground_truth_objects=ground_truth_frame.objects,
            diff_distance=(2.3, 0.0, 0.2),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
        )
        objects_with_difference.pop(0)
        lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )

    # final result
    final_metric_score = lsim.get_final_result()

    # Debug
    logging.info(
        f"Frame result example (frame_results[0]): {format_class_for_log(lsim.evaluator.frame_results[0], 5)}"
    )

    logging.info(
        f"Object result example (frame_results[0].object_results[0]): {format_class_for_log(lsim.evaluator.frame_results[0].object_results[0])}"
    )

    logging.info(
        f"Metrics example (final_metric_score): {format_class_for_log(final_metric_score, len(final_metric_score.config.target_labels))}"
    )

    logging.info(
        f"mAP result example (final_metric_score.maps[0].aps[0]): {format_class_for_log(final_metric_score.maps[0], 100)}"
    )
