from enum import Enum
from enum import unique
from typing import Dict
from typing import List
from typing import Tuple

from awml_evaluation.common.object import DynamicObject


class Color(Enum):
    GREEN = [0, 255, 0]
    YELLOW = [255, 255, 0]
    RED = [255, 0, 0]
    WHITE = [255, 255, 255]


class VisualizationConfig:
    def __init__(
        self,
        height: int = 640,
        width: int = 640,
        is_tracking_visualization: bool = False,
        is_prediction_visualization: bool = False,
    ) -> None:
        height: int = 640
        width: int = 640
        is_tracking_visualization: bool = False
        is_prediction_visualization: bool = False


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
        height: int = 640,
        width: int = 640,
        is_tracking_visualization: bool = False,
        is_prediction_visualization: bool = False,
    ) -> None:
        self.height: int = height
        self.width: int = width
        self.is_tracking_visualization: bool = is_tracking_visualization
        self.is_prediction_visualization: bool = is_prediction_visualization

    def visualize_bev(
        self,
        file_path: str,
        pointcloud: List[List[float]] = None,
        pointcloud_color: Color = Color.WHITE,
        objects_list: List[List[DynamicObject]] = None,
        color_list: List[Color] = None,
        line_width_list: List[int] = None,
    ):
        """
        BEV可視化したpngを吐く
        """

        # set default config
        appearance_config = VisualizationAppearanceConfig(
            len(objects_list),
            color_list,
            line_width_list,
            pointcloud_color,
        )

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
