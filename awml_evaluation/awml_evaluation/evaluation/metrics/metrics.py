from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.map import Map
from awml_evaluation.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from awml_evaluation.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class MetricsScore:
    """[summary]
    Metrics score class.

    Attributes:
        self.detection_config (Optional[DetectionMetricsConfig])
        self.tracking_config (Optional[TrackingMetricsConfig])
        self.prediction_config (Optional[PredictionMetricsConfig]): TBD
        self.maps (List[Map]): The list of mAP class object. Each mAP is different from threshold
                               for matching (ex. IoU 0.3).
        self.tracking_scores (List[TrackingMetricsScore])
        self.prediction_scores (List[TODO]): TBD
    """

    def __init__(
        self,
        config: MetricsScoreConfig,
    ) -> None:
        """[summary]
        Args:
            metrics_config (MetricsScoreConfig): A config for metrics
        """
        self.detection_config = config.detection_config
        self.tracking_config = config.tracking_config
        self.prediction_config = config.prediction_config
        # detection metrics scores for each matching method
        self.maps: List[Map] = []
        # tracking metrics scores for each matching method
        self.tracking_scores: List[TrackingMetricsScore] = []
        # TODO: prediction metrics scores for each matching method
        self.prediction_scores: List = []

    def __str__(self) -> str:
        """[summary]
        Str method
        """
        str_: str = "\n\n"
        # object num
        object_num: int = 0
        for ap_ in self.maps[0].aps:
            object_num += ap_.ground_truth_objects_num
        str_ += f"Ground Truth Num: {object_num}\n"

        # detection
        for map_ in self.maps:
            # whole result
            str_ += str(map_)

        # tracking
        for track_score in self.tracking_scores:
            str_ += str(track_score)

        # TODO: prediction

        return str_

    def evaluate_detection(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
    ) -> None:
        """[summary]
        Calculate detection metrics

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object result
            frame_ground_truths (List[FrameGroundTruth]): The list ground truth for each frame.
        """
        for distance_threshold_ in self.detection_config.center_distance_thresholds:
            map_ = Map(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.detection_config.target_labels,
                max_x_position_list=self.detection_config.max_x_position_list,
                max_y_position_list=self.detection_config.max_y_position_list,
                matching_mode=MatchingMode.CENTERDISTANCE,
                matching_threshold_list=distance_threshold_,
            )
            self.maps.append(map_)
        for iou_threshold_bev_ in self.detection_config.iou_bev_thresholds:
            map_ = Map(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.detection_config.target_labels,
                max_x_position_list=self.detection_config.max_x_position_list,
                max_y_position_list=self.detection_config.max_y_position_list,
                matching_mode=MatchingMode.IOUBEV,
                matching_threshold_list=iou_threshold_bev_,
            )
            self.maps.append(map_)
        for iou_threshold_3d_ in self.detection_config.iou_3d_thresholds:
            map_ = Map(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.detection_config.target_labels,
                max_x_position_list=self.detection_config.max_x_position_list,
                max_y_position_list=self.detection_config.max_y_position_list,
                matching_mode=MatchingMode.IOU3D,
                matching_threshold_list=iou_threshold_3d_,
            )
            self.maps.append(map_)
        for plane_distance_threshold_ in self.detection_config.plane_distance_thresholds:
            map_ = Map(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.detection_config.target_labels,
                max_x_position_list=self.detection_config.max_x_position_list,
                max_y_position_list=self.detection_config.max_y_position_list,
                matching_mode=MatchingMode.PLANEDISTANCE,
                matching_threshold_list=plane_distance_threshold_,
            )
            self.maps.append(map_)

    def evaluate_tracking(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
    ) -> None:
        """[summary]
        Calculate tracking metrics.

        NOTE:
            object_results and ground_truth_objects must be nested list.
            In case of evaluating single frame, [[previous], [current]].
            In case of evaluating multi frame, [[], [t1], [t2], ..., [tn]]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object result for each frame.
            ground_truth_objects (List[List[DynamicObject]]): The list of ground truth object for each frame.
        """
        for distance_threshold_ in self.tracking_config.center_distance_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.tracking_config.target_labels,
                max_x_position_list=self.tracking_config.max_x_position_list,
                max_y_position_list=self.tracking_config.max_y_position_list,
                matching_mode=MatchingMode.CENTERDISTANCE,
                matching_threshold_list=distance_threshold_,
            )
            self.tracking_scores.append(tracking_score_)
        for iou_threshold_bev_ in self.tracking_config.iou_bev_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.tracking_config.target_labels,
                max_x_position_list=self.tracking_config.max_x_position_list,
                max_y_position_list=self.tracking_config.max_y_position_list,
                matching_mode=MatchingMode.IOUBEV,
                matching_threshold_list=iou_threshold_bev_,
            )
            self.tracking_scores.append(tracking_score_)
        for iou_threshold_3d_ in self.tracking_config.iou_3d_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.tracking_config.target_labels,
                max_x_position_list=self.tracking_config.max_x_position_list,
                max_y_position_list=self.tracking_config.max_y_position_list,
                matching_mode=MatchingMode.IOU3D,
                matching_threshold_list=iou_threshold_3d_,
            )
            self.tracking_scores.append(tracking_score_)
        for plane_distance_threshold_ in self.tracking_config.plane_distance_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=self.tracking_config.target_labels,
                max_x_position_list=self.tracking_config.max_x_position_list,
                max_y_position_list=self.tracking_config.max_y_position_list,
                matching_mode=MatchingMode.PLANEDISTANCE,
                matching_threshold_list=plane_distance_threshold_,
            )
            self.tracking_scores.append(tracking_score_)

    def evaluate_prediction(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
    ) -> None:
        """[summary]
        Calculate prediction metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
        """
        pass
