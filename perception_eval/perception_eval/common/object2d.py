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

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from perception_eval.common.label import Label
    from perception_eval.common.schema import FrameID, Visibility


class Roi:
    """Region of Interest; ROI class.

    Todo:
    ----
    - Support multi roi input for Blinker and BrakeLamp objects.

    Attributes:
    ----------
        offset (Tuple[int, int]): Top-left pixels from (0, 0), (x, y) order.
        size (Tuple[int, int]): Size of ROI, (width, height) order.
        center (Tuple[int, int]): Center of ROI, (x, y) order.
        height (int): Height of ROI.
        width (int): Width  of ROI.
        area (int): Area of ROI.

    Args:
    ----
        roi (Tuple[int, int, int, int]): (xmin, ymin, width, height) of ROI.
    """

    def __init__(
        self,
        roi: tuple[int, int, int, int],
    ) -> None:
        if len(roi) != 4:
            msg = "`roi` must be 4 length int array."
            raise ValueError(msg)

        self.__offset: tuple[int, int] = roi[:2]
        self.__size: tuple[int, int] = roi[2:]

        self.__center: tuple[int, int] = (
            self.offset[0] + self.width // 2,
            self.offset[1] + self.height // 2,
        )
        self.__area: int = self.size[0] * self.size[1]
        # corners
        top_left: tuple[int, int] = self.offset
        top_right: tuple[int, int] = (
            self.offset[0] + self.width,
            self.offset[1],
        )
        bottom_right: tuple[int, int] = (
            self.offset[0] + self.width,
            self.offset[1] + self.height,
        )
        bottom_left: tuple[int, int] = (
            self.offset[0],
            self.offset[1] + self.height,
        )
        self.__corners: np.ndarray = np.array([top_left, top_right, bottom_right, bottom_left])

    @property
    def offset(self) -> tuple[int, int]:
        return self.__offset

    @property
    def size(self) -> tuple[int, int]:
        return self.__size

    @property
    def center(self) -> tuple[int, int]:
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

    Attributes:
    ----------
        unix_time (int): Unix time[us].
        frame_id (FrameID): FrameID instance, where 2D objects are with respect, related to CAM_**.
        semantic_score (float): Object's confidence [0, 1].
        semantic_label (Label): Object's Label.
        roi (Optional[Roi]): ROI in image. For classification, None is OK. Defaults to None.
        uuid (Optional[str]): Unique ID. For traffic light objects, set lane ID. Defaults to None.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.

    Args:
    ----
        unix_time (int): Unix time[us].
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
        semantic_label: Label,
        roi: tuple[int, int, int, int] | None = None,
        uuid: str | None = None,
        visibility: Visibility | None = None,
    ) -> None:
        super().__init__()
        self.unix_time: int = unix_time
        self.frame_id: FrameID = frame_id
        self.semantic_score: float = semantic_score
        self.semantic_label: Label = semantic_label
        self.roi: Roi | None = Roi(roi) if roi is not None else None
        self.uuid: str | None = uuid
        self.visibility: Visibility | None = visibility

    def get_corners(self) -> np.ndarray:
        """Returns the corners of bounding box in pixel.

        Returns:
        -------
            numpy.ndarray: (top_left, top_right, bottom_right, bottom_left), in shape (4, 2).
        """
        if self.roi is None:
            msg = "self.roi is None."
            raise RuntimeError(msg)
        return self.roi.corners

    def get_area(self) -> int:
        """Returns the area of bounding box in pixel.

        Returns:
        -------
            int: Area of bounding box[px].
        """
        if self.roi is None:
            msg = "self.roi is None."
            raise RuntimeError(msg)
        return self.roi.area

    def get_polygon(self) -> Polygon:
        """Returns the corners as polygon.

        Returns:
        -------
            Polygon: Corners as Polygon. ((x0, y0), ..., (x0, y0))
        """
        if self.roi is None:
            msg = "self.roi is None."
            raise RuntimeError(msg)
        corners: list[list[float]] = self.get_corners().tolist()
        corners.append(corners[0])
        return Polygon(corners)
