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

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.common.status import Visibility
from shapely.geometry import Polygon


class Roi:
    """Region of Interest; ROI class.
    Attributes:
        self.offset (Tuple[int, int]): (x, y) offset from (0, 0).
        self.size (Tuple[int, int]): (height, width) of bounding box.
    """

    def __init__(
        self,
        offset: Tuple[int, int],
        size: Tuple[int, int],
    ) -> None:
        """
        Args:
            offset (Tuple[int, int]): (x, y) offset from (0, 0).
            size (Tuple[int, int]): (height, width) of bounding box.
        """
        self.offset: Tuple[int, int] = offset
        self.size: Tuple[int, int] = size

        self.__center: Tuple[int, int] = (
            self.offset[0] + self.size[1] // 2,
            self.offset[1] + self.size[0] // 2,
        )
        self.__height: int = self.size[0]
        self.__width: int = self.size[1]
        self.__area: int = self.size[0] * self.size[1]

    @property
    def center(self) -> Tuple[int, int]:
        return self.__center

    @property
    def height(self) -> int:
        return self.__height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def area(self) -> int:
        return self.__area


class DynamicObject2D:
    def __init__(
        self,
        unix_time: int,
        semantic_score: float,
        semantic_label: LabelType,
        roi: Optional[Roi] = None,
        uuid: Optional[str] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        super().__init__()
        self.unix_time: int = unix_time
        self.semantic_score: float = semantic_score
        self.semantic_label: LabelType = semantic_label
        self.roi: Optional[Roi] = roi
        self.uuid: Optional[str] = uuid
        self.visibility: Optional[Visibility] = visibility

    def get_corners(self) -> np.ndarray:
        """[summary]
        Returns the corners of bounding box in pixel.

        Returns:
            numpy.ndarray: (top_left, top_right, bottom_right, bottom_left), in shape (4, 2).
        """
        if self.roi is None:
            raise RuntimeError("self.roi is None.")

        top_left: Tuple[int, int] = self.roi.offset
        top_right: Tuple[int, int] = (
            self.roi.offset[0] + self.roi.width,
            self.roi.offset[1],
        )
        bottom_right: Tuple[int, int] = (
            self.roi.offset[0] + self.roi.width,
            self.roi.offset[1] + self.roi.height,
        )
        bottom_left: Tuple[int, int] = (
            self.roi.offset[0],
            self.roi.offset[1] + self.roi.height,
        )
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_area(self) -> int:
        """[summary]
        Returns the area of bounding box in pixel.

        Returns:
            int: Area of bounding box[px].
        """
        if self.roi is None:
            raise RuntimeError("self.roi is None.")
        return self.roi.area

    def get_polygon(self) -> Polygon:
        """[summary]
        Returns the corners as polygon.

        Returns:
            Polygon: Corners as Polygon. ((x0, y0), ..., (x0, y0))
        """
        corners: List[List[float]] = self.get_corners().tolist()
        corners.append(corners[0])
        return Polygon(corners)
