import logging
from test.util.logger_config import configure_logger
from typing import List

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.frame_result import FrameResult
from awml_evaluation.evaluation.result.pass_fail_result import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.pass_fail_result import FramePassFailConfig
from awml_evaluation.evaluation_config import EvaluationConfig
from awml_evaluation.evaluation_manager import EvaluationManager
from awml_evaluation.util.debug import get_objects_with_difference

# from awml_evaluation.util.debug import format_class_for_log


class LSimMoc:
    def __init__(self, dataset_paths: List[str]):
        evaluation_config: EvaluationConfig = EvaluationConfig(
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
        self.evaluator = EvaluationManager(evaluation_config=evaluation_config)

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
        # 距離などでUC評価objectを選別するためのインターフェイス（EvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        # どれを注目物体とするかのparam
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            max_x_position_list=[30.0, 30.0, 30.0, 30.0],
            max_y_position_list=[30.0, 30.0, 30.0, 30.0],
        )
        # Pass fail を決めるパラメータ
        frame_pass_fail_config: FramePassFailConfig = FramePassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            threshold_plane_distance_list=[2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
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
            number_use_case_fail_object += len(frame_results.pass_fail_result.uc_fail_objects)
        logging.info(f"final use case fail object: {number_use_case_fail_object}")
        final_metric_score = self.evaluator.get_scenario_result()
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    @staticmethod
    def visualize(frame_result: FrameResult):
        """
        可視化
        """
        if len(frame_result.pass_fail_result.uc_fail_objects) > 0:
            logging.warning(f"{len(frame_result.pass_fail_result.uc_fail_objects)} fail objects")
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
            diff_distance=(0.3, 0.0, 0.0),
            diff_yaw=0.2,
            is_confidence_with_distance=True,
        )
        lsim.callback(
            ground_truth_frame.unix_time,
            objects_with_difference,
        )
    final_metric_score = lsim.get_final_result()
