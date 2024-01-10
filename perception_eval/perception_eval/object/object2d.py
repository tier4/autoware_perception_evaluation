# Copyright 2022-2024 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from perception_eval.common.label import SemanticLabel
    from perception_eval.common.schema import FrameID
    from perception_eval.common.schema import Visibility


class Roi:
    """Region of Interest; ROI class.

    TODO:
    - Support multi roi input for Blinker and BrakeLamp objects.

    Args:
    -----
        roi (Tuple[int, int, int, int]): (xmin, ymin, width, height) of ROI.
    """

    def __init__(self, roi: Tuple[int, int, int, int]) -> None:
        if len(roi) != 4:
            raise ValueError("`roi` must be 4 length int array.")

        self.__offset: Tuple[int, int] = roi[:2]
        self.__size: Tuple[int, int] = roi[2:]

        self.__center: Tuple[int, int] = (
            self.offset[0] + self.width // 2,
            self.offset[1] + self.height // 2,
        )
        self.__area: int = self.size[0] * self.size[1]
        # corners
        top_left: Tuple[int, int] = self.offset
        top_right: Tuple[int, int] = (
            self.offset[0] + self.width,
            self.offset[1],
        )
        bottom_right: Tuple[int, int] = (
            self.offset[0] + self.width,
            self.offset[1] + self.height,
        )
        bottom_left: Tuple[int, int] = (
            self.offset[0],
            self.offset[1] + self.height,
        )
        self.__corners: np.ndarray = np.array([top_left, top_right, bottom_right, bottom_left])

    @property
    def offset(self) -> Tuple[int, int]:
        return self.__offset

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @property
    def center(self) -> Tuple[int, int]:
        return self.__center

    @property
    def width(self) -> int:
        return self.__size[0]

    @property
    def height(self) -> int:
        return self.__size[1]

    @property
    def area(self) -> int:
        return self.__area

    @property
    def corners(self) -> np.ndarray:
        return self.__corners


class DynamicObject2D:
    """Dynamic object class for 2D object.

    Args:
    -----
        unix_time (int): Unix time [us].
        frame_id (FrameID): FrameID instance, where 2D objects are with respect, related to CAM_**.
        semantic_score (float): Object's confidence [0, 1].
        semantic_label (Label): Object's Label.
        roi (Optional[Tuple[int, int, int, int]]): (xmin, ymin, width, height) of ROI.
            For classification, None is OK. Defaults to None.
        uuid (Optional[str]): Unique ID. For traffic light objects, set lane ID. Defaults to None.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_id: FrameID,
        semantic_score: float,
        semantic_label: SemanticLabel,
        roi: Optional[Tuple[int, int, int, int]] = None,
        uuid: Optional[str] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        super().__init__()
        self.unix_time = unix_time
        self.frame_id = frame_id
        self.semantic_score = semantic_score
        self.semantic_label = semantic_label
        self.roi = Roi(roi) if roi is not None else None
        self.uuid = uuid
        self.visibility = visibility

    def get_corners(self) -> np.ndarray:
        """Returns the corners of bounding box in pixel.

        Returns:
        --------
            np.ndarray: (top_left, top_right, bottom_right, bottom_left), in shape (4, 2).
        """
        assert self.roi is not None, "self.roi is None."
        return self.roi.corners

    def get_area(self) -> int:
        """Returns the area of bounding box in pixel.

        Returns:
        --------
            int: Area of bounding box[px].
        """
        assert self.roi is not None, "self.roi is None."
        return self.roi.area

    def get_polygon(self) -> Polygon:
        """Returns the corners as polygon.

        Returns:
        --------
            Polygon: Corners as Polygon. ((x0, y0), ..., (x0, y0))
        """
        assert self.roi is not None, "self.roi is None."
        corners: List[List[float]] = self.get_corners().tolist()
        corners.append(corners[0])
        return Polygon(corners)
