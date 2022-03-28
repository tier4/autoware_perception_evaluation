import os
from typing import List
from typing import Optional

import numpy as np

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import divide_tp_fp_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_object_results
from awml_evaluation.evaluation.matching.objects_filter import get_fn_objects
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.visualization.visualization_config import Color
from awml_evaluation.visualization.visualization_config import VisualizationAppearanceConfig
from awml_evaluation.visualization.visualization_config import VisualizationConfig


class VisualizationBEV:
    def __init__(
        self,
        visualization_directory_path: str,
        height: int = 640,
        width: int = 640,
        is_tracking_visualization: bool = False,
        is_prediction_visualization: bool = False,
    ) -> None:
        self.visualization_config = VisualizationConfig(
            visualization_directory_path,
            height,
            width,
            is_tracking_visualization,
            is_prediction_visualization,
        )

    def visualize_bev(
        self,
        file_name: str,
        object_results: Optional[List[DynamicObjectWithPerceptionResult]] = None,
        ground_truth_objects: Optional[List[DynamicObject]] = None,
        pointcloud: Optional[np.ndarray] = None,
        matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
        matching_threshold: float = 1.0,
        pointcloud_color: Color = Color.WHITE,
        objects_list: List[List[DynamicObject]] = None,
        color_list: List[Color] = [Color.GREEN, Color.YELLOW, Color.RED, Color.WHITE],
        line_width_list: List[int] = [4, 4, 4, 1],
    ):
        """[summary]
        Visualize the frame result from bird eye view

        Args:
            file_name (Optional[str]): File name. Defaults to None.
        """

        # set file name
        file_name_: str
        if file_name is None and object_results is not None:
            unix_time = object_results[0].predicted_object.unix_time
            file_name_ = f"{unix_time}.png"
        else:
            file_name_ = file_name
        file_path: str = os.path.join("bev_pictures", file_name_)
        full_path: str = os.path.join(
            self.visualization_config.visualization_directory_path, file_path
        )

        # set default config
        if object_results is not None:
            appearance_config = VisualizationAppearanceConfig(
                len(object_results),
                color_list,
                line_width_list,
                pointcloud_color,
            )
        else:
            appearance_config = None

        # set object
        # filtered_predicted_objects: List[DynamicObjectWithPerceptionResult] = filter_tp_objects(
        #     object_results=object_results,
        #     matching_mode=matching_mode,
        #     matching_threshold=matching_threshold,
        # )
        # tp_objects, fp_objects = divide_tp_fp_objects(filtered_predicted_objects)
        # fn_objects = get_fn_objects(tp_objects, self.ground_truth_objects)

        # image_data = []

        # if pointcloud:
        #     self._add_pointcloud(pointcloud)

        # if objects_list:
        #     for objects_, color_, line_width_ in (
        #         objects_list,
        #         appearance_config.color_list,
        #         appearance_config.line_width_list,
        #     ):
        #         for ob in objects_:
        #             self._add_bbox(ob, color_, line_width_)

        # save png
        # self.save_png_file()
        raise NotImplementedError()

    def _add_pointcloud(pointcloud: np.ndarray, color: Color):
        """
        pointcloudの描画
        """
        raise NotImplementedError()

    def _add_bbox(
        self,
        object: DynamicObject,
        color: Color,
        width: int,
    ):
        """
        objectの描画
        """
        raise NotImplementedError()
