from enum import Enum
import os
from typing import List

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import divide_tp_fp_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_tp_objects
from awml_evaluation.evaluation.matching.objects_filter import get_fn_objects
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult


class Color(Enum):
    GREEN = [0, 255, 0]
    YELLOW = [255, 255, 0]
    RED = [255, 0, 0]
    WHITE = [255, 255, 255]


class VisualizationConfig:
    def __init__(
        self,
        visualization_directory_path: str,
        height: int = 640,
        width: int = 640,
        is_tracking_visualization: bool = False,
        is_prediction_visualization: bool = False,
    ) -> None:
        self.visualization_directory_path: str = visualization_directory_path
        self.height: int = height
        self.width: int = width
        self.is_tracking_visualization: bool = is_tracking_visualization
        self.is_prediction_visualization: bool = is_prediction_visualization


class VisualizationAppearanceConfig:
    def __init__(
        self,
        objects_list_num: int,
        color_list: List[Color],
        line_width_list: List[int],
        pointcloud_color: Color = Color.WHITE,
    ) -> None:

        # object setting check
        try:
            len(color_list) != objects_list_num or len(line_width_list) != objects_list_num
        except:
            print("VisualizationObjectConfig Error")

        self.color_list: List[Color] = self._set_color(color_list, objects_list_num)
        self.line_width: List[str] = self._set_line_width(line_width_list, objects_list_num)
        self.pointcloud_color: Color = pointcloud_color

    def _set_line_width(line_width_list: List[str], object_num: int):
        if line_width_list:
            return line_width_list
        else:
            return [2.0 for i in range(object_num)]

    def _set_color(color_list: List[str], object_num: int):
        if color_list:
            return color_list
        else:
            return [Color.WHITE for i in range(object_num)]


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
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
        pointcloud: List[List[float]] = None,
        matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
        matching_threshold: float = 1.0,
        pointcloud_color: Color = Color.WHITE,
        objects_list: List[List[DynamicObject]] = None,
        color_list: List[Color] = [Color.GREEN, Color.YELLOW, Color.RED, Color.WHITE],
        line_width_list: List[int] = [4.0, 4.0, 4.0, 1.0],
    ):
        """[summary]
        Visualize the frame result from bird eye view

        Args:
            file_name (Optional[str]): File name. Defaults to None.
        """

        # set file name
        if file_name is None:
            file_name_: str = f"{self.unix_time}_{self.frame_name}.png"
        else:
            file_name_: str = file_name
        file_path: str = os.join("bev_pictures", file_name_)
        full_path: str = os.join(self.visualization_directory_path, file_path)

        # set default config
        appearance_config = VisualizationAppearanceConfig(
            len(objects_list),
            color_list,
            line_width_list,
            pointcloud_color,
        )

        # set object
        filtered_predicted_objects: List[DynamicObjectWithResult] = filter_tp_objects(
            object_results=object_results,
            matching_mode=matching_mode,
            matching_threshold=matching_threshold,
        )
        tp_objects, fp_objects = divide_tp_fp_objects(filtered_predicted_objects)
        fn_objects = get_fn_objects(tp_objects, self.ground_truth_objects)

        image_data = []

        if pointcloud:
            self._add_pointcloud(pointcloud)

        if objects_list:
            for objects_, color_, line_width_ in (
                objects_list,
                appearance_config.color_list,
                appearance_config.line_width_list,
            ):
                for ob in objects_:
                    self._add_bbox(ob, color_, line_width_)

        # save png
        # self.save_png_file()
        raise NotImplementedError()

    def _add_pointcloud(pointcloud: List[List[float]], color: Color):
        """
        pointcloudの描画
        """
        print("visualize pointcloud")
        raise NotImplementedError()

    def _add_bbox(
        object: DynamicObject,
        color: Color,
        width: int,
    ):
        """
        objectの描画
        """
        raise NotImplementedError()
