# import rospy

import datetime
import logging
import os
from typing import List

from pyquaternion.quaternion import Quaternion

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.frame_result import FrameResult
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation_manager import EvaluationManager
from awml_evaluation.util.debug import format_class_for_log


class LSim:
    def __init__(self, dataset_path: str):
        self.evaluator: EvaluationManager = EvaluationManager(
            dataset_path=dataset_path,
            does_use_pointcloud=False,
            result_root_directory="data/result/{TIME}/",
            log_directory="",
            visualization_directory="visualization/",
            evaluation_tasks=["detection"],
            # target_labels=["car", "truck", "bicycle", "pedestrian", "motorbike"],
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            map_thresholds_center_distance=[0.5, 1.0, 2.0],
            map_thresholds_plane_distance=[0.5, 1.0, 2.0],
            map_thresholds_iou=[],
        )

        _ = configure_logger(
            log_file_directory=self.evaluator.evaluator_config.result_log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

    # rospy.init_node('LSim for perception', anonymous=True)
    # self.detection_sub = rospy.Subscriber("/perception/detection", DynamicObjectWithArray, callback_detection)
    # self.ground_truth_sub = rospy.Subscriber("/perception/detection", DynamicObjectWithArray, callback_ground_truth)
    # rospy.Timer(rospy.Duration(1.0), self.timerCallback)

    def callback_detection(self, data):
        self.predicted_objects: List[DynamicObject] = []
        for d in data:
            # will_collide_within_5s = self._will_collide_within_5s(d, )
            will_collide_within_5s = True
            self.predicted_objects.append(
                DynamicObject(
                    d.unix_time,
                    d.semantic_score,
                    d.semantic_label,
                    d.pose,
                    d.shape,
                    d.twist,
                    will_collide_within_5s,
                )
            )

    def timer_callback(self, event):
        # adjust for ROS interface
        unix_time = 10000000  # dummy
        self.evaluate_one_frame(unix_time, self.predicted_objects)

    def evaluate_one_frame(
        self,
        unix_time: int,
        predicted_objects: List[DynamicObject],
    ) -> None:
        frame_result = self.evaluator.add_frame_result(
            unix_time,
            predicted_objects,
        )
        self.visualize(frame_result)

    def evaluate_one_frame_with_diff(
        self,
        unix_time: int,
        predicted_objects: List[DynamicObject],
        diff_distance: float = 0.0,
        diff_yaw: float = 0.0,
    ) -> None:
        test_objects_ = []
        for predicted_object in predicted_objects:
            position = (
                predicted_object.state.position[0] + diff_distance,
                predicted_object.state.position[1],
                predicted_object.state.position[2],
            )
            orientation = Quaternion(
                axis=predicted_object.state.orientation.axis,
                radians=predicted_object.state.orientation.radians + diff_yaw,
            )
            test_object_ = DynamicObject(
                predicted_object.unix_time,
                position,
                orientation,
                predicted_object.state.size,
                predicted_object.state.velocity,
                predicted_object.semantic_score,
                predicted_object.semantic_label,
            )
            test_objects_.append(test_object_)
        frame_result = self.evaluator.add_frame_result(
            unix_time,
            test_objects_,
        )
        self.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """
        final_metric_score = self.evaluator.get_scenario_result()
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    @staticmethod
    def visualize(frame_result: FrameResult):
        """
        可視化
        """
        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug(f"mAP is low")
            logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")


def CustomTextFormatter():
    """[summary]
    Custom Formatter
    """
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [func]  [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
    )


class SensitiveWordFilter(logging.Filter):
    """[summary]
    The class to filer sensitive words like password
    """

    def filter(self, record):
        sensitive_words = [
            "password",
            "auth_token",
            "secret",
        ]
        log_message = record.getMessage()
        for word in sensitive_words:
            if word in log_message:
                return False
        return True


def configure_logger(
    log_file_directory: str,
    console_log_level=logging.INFO,
    file_log_level=logging.INFO,
) -> None:
    """[summary]
    The function to make logger

    Args:
        log_file_directory (str): The directory path to save log
        console_log_level ([type], optional): Log level for console. Defaults to logging.INFO.
        file_log_level ([type], optional): Log level for log file. Defaults to logging.INFO.
        modname ([type], optional): Modname for logger. Defaults to __name__.
    """
    # make directory
    log_directory = os.path.dirname(log_file_directory)
    os.makedirs(log_directory, exist_ok=True)

    formatter = CustomTextFormatter()

    logger = logging.getLogger("")
    logger.addFilter(SensitiveWordFilter())
    logger.setLevel(console_log_level)

    # handler for console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # handler for file
    time = "{0:%Y%m%d_%H%M%S}.txt".format(datetime.datetime.now())
    log_file_path = os.path.join(log_directory, time)
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setLevel(file_log_level)
    file_formatter = CustomTextFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # dataset_path = "../../dataset_3d/tier4/202108_3d_cuboid_v1_1_1_nishishinjuku_mini/5f97e3d5a3c9d30032763ab5"
    # dataset_path = "../../dataset_3d/tier4/20210819_3d_cuboid_v1_2_sample/602625664232f4002ce37cef"
    dataset_path = "../../dataset_3d/tier4/202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475"
    lsim = LSim(dataset_path)
    # rospy.spin()
    # dummy data

    # example code for mAP 1.0
    # for ground_truth_frame in lsim.evaluator.ground_truth_frames:
    #     lsim.evaluate_one_frame(
    #         ground_truth_frame.unix_time,
    #         ground_truth_frame.objects,
    #     )

    for ground_truth_frame in lsim.evaluator.ground_truth_frames:
        lsim.evaluate_one_frame_with_diff(
            ground_truth_frame.unix_time,
            ground_truth_frame.objects,
            diff_distance=1.9,
            diff_yaw=0.5,
            # 1.4,
        )

    # example code for 1 frame difference
    # for i in range(1, len(lsim.evaluator.ground_truth_frames)):
    #     lsim.evaluate_one_frame(
    #         lsim.evaluator.ground_truth_frames[i].unix_time,
    #         lsim.evaluator.ground_truth_frames[i - 1].objects,
    #     )

    final_metric_score = lsim.get_final_result()
