from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from perception_eval.object import ObjectType


class FrameGroundTruth:
    """
    Ground truth data per frame

    Attributes:
        unix_time (float): The unix time for the frame [us].
        frame_name (str): The file name for the frame.
        objects (List[DynamicObject]): Objects data.
        pointcloud (Optional[numpy.ndarray], optional):
                Pointcloud data. Defaults to None, but if you want to visualize dataset,
                you should load pointcloud data.
        transform_matrix (Optional[np.ndarray]): The numpy array to transform position.
        objects (List[ObjectType]): Objects data.
        ego2map (Optional[np.ndarray]): The numpy array to transform position.
        raw_data (Optional[Dict[str, numpy.ndarray]]): Raw data for each sensor modality.

    Args:
        unix_time (int): The unix time for the frame [us]
        frame_name (str): The file name for the frame
        objects (List[DynamicObject]): Objects data.
        ego2map (Optional[np.ndarray]): The array of 4x4 matrix.
            Transform position with respect to vehicle coord system to map one.
        raw_data (Optional[Dict[str, numpy.ndarray]]): Raw data for each sensor modality.
    """

    def __init__(
        self,
        unix_time: int,
        frame_number: int,
        objects: List[ObjectType],
        ego2map: Optional[np.ndarray] = None,
        raw_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.unix_time = unix_time
        self.frame_number = frame_number
        self.objects = objects
        self.ego2map = ego2map
        self.raw_data = raw_data
