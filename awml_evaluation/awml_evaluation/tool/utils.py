# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
import numpy as np


def generate_area_points(
    num_area_division: int,
    max_x: float,
    max_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]
    Generate (x,y) pairs of upper right and bottom left of each separate area.
    They are arranged in numerical order as shown in below.

    num_area_division:
        1:                            3:                  9:
                    max_x_position
                    +--------+          +--------+          +--------+
                    |    0   |          |____0___|          |_0|_1|_2|
    max_y_position  |    +   |          |____1___|          |_3|_4|_5|
                    |   ego  |          |    2   |          | 6| 7| 8|
                    +--------+          +--------+          +--------+

    Args:
        num_area_division (int)
        max_x (float)
        max_y (float)

    Returns:
        upper_rights (np.ndarray)
        bottom_lefts (np.ndarray)
    """
    if num_area_division == 1:
        upper_rights: np.ndarray = np.array((max_x, -max_y)).reshape(1, -1)
        bottom_lefts: np.ndarray = np.array((-max_x, max_y)).reshape(1, -1)
    elif num_area_division == 3:
        right_x: np.ndarray = np.arange(max_x, -max_x, -2 * max_x / 3)
        left_x: np.ndarray = np.arange(-max_x, max_x, 2 * max_x / 3)[::-1]
        right_y: np.ndarray = np.repeat(-max_y, 3)
        left_y: np.ndarray = np.repeat(max_y, 3)
        upper_rights: np.ndarray = np.stack([right_x, right_y], axis=1)
        bottom_lefts: np.ndarray = np.stack([left_x, left_y], axis=1)
    elif num_area_division == 9:
        right_x: np.ndarray = np.arange(max_x, -max_x, -2 * max_x / 3)
        left_x: np.ndarray = np.arange(-max_x, max_x, 2 * max_x / 3)[::-1]
        right_y: np.ndarray = np.arange(-max_y, max_y, 2 * max_y / 3)[::-1]
        left_y: np.ndarray = np.arange(max_y, -max_y, -2 * max_y / 3)
        r_xx, r_yy = np.meshgrid(right_x, right_y)
        l_xx, l_yy = np.meshgrid(left_x, left_y)
        upper_rights: np.ndarray = np.stack([r_xx, r_yy], axis=-1).reshape(-1, 2)
        bottom_lefts: np.ndarray = np.stack([l_xx, l_yy], axis=-1).reshape(-1, 2)
    else:
        raise ValueError(
            f"The number of area division must be 1, 3 or 9, but got {num_area_division}"
        )

    return upper_rights, bottom_lefts


def get_area_idx(
    frame_id: str,
    object_result: Union[DynamicObject, DynamicObjectWithPerceptionResult],
    upper_rights: np.ndarray,
    bottom_lefts: np.ndarray,
    ego2map: Optional[np.ndarray] = None,
) -> Optional[int]:
    """[summary]
    Returns the index of area.

    Args:
        object_result (Union[DynamicObject, DynamicObjectWithPerceptionResult])
        upper_rights (np.ndarray): in shape (N, 2), N is number of area division.
        bottom_lefts (np.ndarray): in shape (N, 2), N is number of area division.
        ego2map (Optional[np.ndarray]): in shape (4, 4)

    Returns:
        area_idx (Optional[int]): If the position is out of range, returns None.
    """
    if isinstance(object_result, DynamicObject):
        obj_xyz: np.ndarray = np.array(object_result.state.position)
    elif isinstance(object_result, DynamicObjectWithPerceptionResult):
        obj_xyz: np.ndarray = np.array(object_result.estimated_object.state.position)
    else:
        raise TypeError(f"Unexpected object type: {type(object_result)}")

    if frame_id == "map":
        if ego2map is None:
            raise ValueError("When frame id is map, ego2map must be specified.")
        obj_xyz: np.ndarray = np.append(obj_xyz, 1.0)
        obj_xy = np.linalg.inv(ego2map).dot(obj_xyz)[:2]
    elif frame_id == "base_link":
        obj_xy = obj_xyz[:2]
    else:
        raise ValueError(f"Unexpected frame_id: {frame_id}")

    is_x_inside: np.ndarray = (obj_xy[0] < upper_rights[:, 0]) * (obj_xy[0] > bottom_lefts[:, 0])
    is_y_inside: np.ndarray = (obj_xy[1] > upper_rights[:, 1]) * (obj_xy[1] < bottom_lefts[:, 1])
    if any(is_x_inside * is_y_inside) is False:
        return None
    area_idx: int = np.where(is_x_inside * is_y_inside)[0].item()
    return area_idx


def extract_area_results(
    frame_results: List[PerceptionFrameResult],
    area: Union[int, List[int]],
    upper_rights: np.ndarray,
    bottom_lefts: np.ndarray,
) -> List[PerceptionFrameResult]:
    """[summary]
    Extract object results and ground truth of PerceptionFrameResult in area.
    Args:
        frame_results (List[PerceptionFrameResult])
        area (Union[int, List[int]])
        upper_rights (np.ndarray)
        bottom_lefts (np.ndarray)
    Returns:
        List[PerceptionFrameResult]
    """
    out_frame_results: List[PerceptionFrameResult] = deepcopy(frame_results)
    if isinstance(area, int):
        area = [area]

    for frame_result in out_frame_results:
        out_object_results: List[DynamicObjectWithPerceptionResult] = []
        out_ground_truths: List[DynamicObject] = []
        frame_id: str = frame_result.frame_ground_truth.frame_id
        ego2map: Optional[np.ndarray] = frame_result.frame_ground_truth.ego2map
        for object_result in frame_result.object_results:
            object_result_area: int = get_area_idx(
                frame_id,
                object_result,
                upper_rights,
                bottom_lefts,
                ego2map,
            )
            if object_result_area in area:
                out_object_results.append(object_result)
        for ground_truth in frame_result.frame_ground_truth:
            ground_truth_area: int = get_area_idx(
                frame_id,
                ground_truth,
                upper_rights,
                bottom_lefts,
                ego2map,
            )
            if ground_truth_area in area:
                out_ground_truths.append(ground_truth)

        frame_result.object_results = out_object_results
        frame_result.frame_ground_truth.objects = out_ground_truths

    return out_frame_results
