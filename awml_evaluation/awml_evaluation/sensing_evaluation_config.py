import datetime
import os.path as osp
from typing import List

from awml_evaluation.common.label import LabelConverter
from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig


class SensingEvaluationConfig:
    """The class of config for sensing evaluation.

    Attributes:
        self.dataset_paths (List[str]): The path(s) of dataset(s).
        self.does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
        self.result_root_directory (str): The directory path to save result.
        self.log_directory (str): The directory path to save log.
        self.visualization_directory (str): The directory path to save visualization result.
        self.target_uuids (List[str]): The list of objects' uuids should be detected.
        self.label_converter (LabelConverter): The instance to convert label names.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        does_use_pointcloud: bool,
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        target_uuids: List[str],
        box_scale_0m: float,
        box_scale_100m: float,
        min_points_threshold: int,
    ):
        """
        Args:
            dataset_paths (List[str]): The list of dataset paths.
            does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
            result_root_directory (str): The directory path to save result.
            log_directory (str): The directory path to save log.
            visualization_directory (str): The directory path to save visualization result.
            target_uuids (List[str]): The list of uuids to be selected.
            box_scale_0m (float): The scale factor for bounding box at 0m.
            box_scale_100m (float): The scale factor for bounding box at 100m.
            min_points_threshold (int): The minimum number of points should be detected in box. Defaults to 1.
        """
        # dataset
        self.dataset_paths: List[str] = dataset_paths
        self.does_use_pointcloud: bool = does_use_pointcloud

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.log_directory: str = log_directory
        self.visualization_directory: str = visualization_directory

        # target uuids
        self.target_uuids: List[str] = target_uuids

        # labels
        self.label_converter: LabelConverter = LabelConverter()

        # config per frame
        self.sensing_frame_config: SensingFrameConfig = SensingFrameConfig(
            box_scale_0m=box_scale_0m,
            box_scale_100m=box_scale_100m,
            min_points_threshold=min_points_threshold,
        )

    def get_result_log_directory(self) -> str:
        """[summary]
        Get the full path to put logs

        Returns:
            str: The full path to put logs
        """
        return osp.join(self.result_root_directory, self.log_directory)

    def get_result_visualization_directory(self) -> str:
        """[summary]
        Get the full path to put the visualization images

        Returns:
            str: The full path to put the visualization images
        """
        return osp.join(self.result_root_directory, self.visualization_directory)
