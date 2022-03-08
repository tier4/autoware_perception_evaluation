from typing import List

from awml_evaluation.common.dataset import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.map import Map
from awml_evaluation.evaluation.metrics.metrics_config import MetricsScoreConfig
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class MetricsScore:
    """[summary]
    Metrics score class.

    Attributes:
        self.config (MetricsScoreConfig): The config for metrics calculation.
        self.maps (List[Map]): The list of mAP class object. Each mAP is different from threshold
                               for matching (ex. IoU 0.3).
    """

    def __init__(
        self,
        config: MetricsScoreConfig,
    ) -> None:
        """[summary]
        Args:
            metrics_config (MetricsScoreConfig): A config for metrics
        """
        self.config: MetricsScoreConfig = config
        # for detection metrics
        self.maps: List[Map] = []

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

        for map_ in self.maps:
            # whole result
            str_ += "\n"
            str_ += f"mAP: {map_.map:.3f}, mAPH: {map_.maph:.3f} "
            str_ += f"({map_.map_config.matching_mode.value})\n"
            # Table
            str_ += "\n"
            # label
            str_ += "|      Label |"
            for ap_ in map_.aps:
                target_str: str = ""
                for target in ap_.target_labels:
                    target_str += target.value
                str_ += f" {target_str}({ap_.matching_threshold_list}) | "
            str_ += "\n"
            str_ += "| :--------: |"
            for ap_ in map_.aps:
                str_ += " :---: |"
            str_ += "\n"
            str_ += "| Predict_num |"
            for ap_ in map_.aps:
                str_ += f" {ap_.objects_results_num} |"
            # Each label result
            str_ += "\n"
            str_ += "|         AP |"
            for ap_ in map_.aps:
                str_ += f" {ap_.ap:.3f} | "
            str_ += "\n"
            str_ += "|        APH |"
            for aph_ in map_.aphs:
                target_str: str = ""
                for target in aph_.target_labels:
                    target_str += target.value
                str_ += f" {aph_.ap:.3f} | "
            str_ += "\n"
        return str_

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate API

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
            ground_truth_objects (List[DynamicObject]): The ground truth objects
        """
        self._evaluation_detection(object_results, ground_truth_objects)
        self._evaluation_tracking(object_results)
        self._evaluation_prediction(object_results)

    def _evaluation_detection(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Calculate detection metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
            ground_truth_objects (List[DynamicObject]): The ground truth objects
        """
        for distance_threshold_ in self.config.map_thresholds_center_distance:
            if distance_threshold_:
                map_ = Map(
                    object_results=object_results,
                    ground_truth_objects=ground_truth_objects,
                    target_labels=self.config.target_labels,
                    max_x_position_list=self.config.max_x_position_list,
                    max_y_position_list=self.config.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=distance_threshold_,
                )
                self.maps.append(map_)
        for iou_threshold_bev_ in self.config.map_thresholds_iou_bev:
            if iou_threshold_bev_:
                map_ = Map(
                    object_results=object_results,
                    ground_truth_objects=ground_truth_objects,
                    target_labels=self.config.target_labels,
                    max_x_position_list=self.config.max_x_position_list,
                    max_y_position_list=self.config.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=iou_threshold_bev_,
                )
                self.maps.append(map_)
        for iou_threshold_3d_ in self.config.map_thresholds_iou_3d:
            if iou_threshold_3d_:
                map_ = Map(
                    object_results=object_results,
                    ground_truth_objects=ground_truth_objects,
                    target_labels=self.config.target_labels,
                    max_x_position_list=self.config.max_x_position_list,
                    max_y_position_list=self.config.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=iou_threshold_3d_,
                )
                self.maps.append(map_)
        for plane_distance_threshold_ in self.config.map_thresholds_plane_distance:
            if plane_distance_threshold_:
                map_ = Map(
                    object_results=object_results,
                    ground_truth_objects=ground_truth_objects,
                    target_labels=self.config.target_labels,
                    max_x_position_list=self.config.max_x_position_list,
                    max_y_position_list=self.config.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=plane_distance_threshold_,
                )
                self.maps.append(map_)

    def _evaluation_tracking(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> None:
        """[summary]
        Calculate tracking metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
        """
        pass

    def _evaluation_prediction(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> None:
        """[summary]
        Calculate prediction metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
        """
        pass
