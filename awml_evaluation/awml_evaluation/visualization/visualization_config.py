from enum import Enum
from typing import List


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
        if len(color_list) != objects_list_num or len(line_width_list) != objects_list_num:
            RuntimeError("VisualizationObjectConfig Error")

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
