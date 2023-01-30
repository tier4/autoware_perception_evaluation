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

from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from shapely.geometry import Polygon


class ShapeType(Enum):
    """Type of shape.

    BOUNDING_BOX
    CYLINDER
    POLYGON
    """

    BOUNDING_BOX = "bounding_box"
    CYLINDER = "cylinder"
    POLYGON = "polygon"

    @classmethod
    def from_value(cls, name: str) -> ShapeType:
        """Returns ShapeType instance from string.

        Args:
            name (str): ShapeTYpe name in string.

        Returns:
            ShapeType: ShapeType instance.

        Raises:
            ValueError: When unexpected name is specified.

        Examples:
            >>> ShapeType.from_value("bounding_box")
            ShapeType.BOUNDING_BOX
        """
        for k, v in cls.__members__.items():
            if v == name:
                return k
        raise ValueError(f"Unexpected name: {name}, choose from {list(cls.__members__.keys())}")


class Shape:
    """Shape class.

    NOTE: Footprint of cylinder is rectangle.

    Attributes:
        type (ShapeType): Type of shape.
        size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
            - BOUNDING_BOX:
                (width, length, height)
            - CYLINDER:
                (diameter, 0.0, height)
            - POLYGON:
                (0.0, 0.0, height)
        footprint (Optional[Polygon]): When shape is BOUNDING_BOX, it is allowed to be None.
            Footprint should be with respect to each object's coordinate system.

    Args:
        type (Union[str, ShapeType]): Type of shape, BOUNDING_BOX, CYLINDER or POLYGON.
        size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
            - BOUNDING_BOX:
                (width, length, height)
            - CYLINDER:
                (diameter, 0.0, height)
            - POLYGON:
                (0.0, 0.0, height)
        footprint (Optional[Polygon]): When shape type is BOUNDING_BOX or CYLINDER, this needs not to be set.
            However, when POLYGON, this must be set. Defaults to None.
            Footprint should be with respect to each object's coordinate system.
    """

    def __init__(
        self,
        type: Union[str, ShapeType],
        size: Tuple[float, float, float],
        footprint: Optional[Polygon] = None,
    ) -> None:
        if isinstance(type, str):
            type = ShapeType.from_value(type)

        if type == ShapeType.POLYGON and footprint is None:
            raise RuntimeError("For POLYGON shape objects, footprint must be set")

        self.type: ShapeType = type
        self.size = (size[0], size[0], size[2]) if type == ShapeType.CYLINDER else size
        self.footprint: Optional[Polygon] = footprint if footprint else self.get_footprint()

    def get_footprint(self) -> Polygon:
        """Calculate footprint for type of BOUNDING_BOX or CYLINDER with respect to each object's coordinate system.

        Returns:
            footprint (Polygon): Object's footprint.
        """
        if self.type == ShapeType.POLYGON:
            raise RuntimeError("Expected BOUNDING_BOX or CYLINDER")

        corners: List[np.ndarray] = [
            np.array([self.size[1], self.size[0], 0.0]) / 2.0,
            np.array([-self.size[1], self.size[0], 0.0]) / 2.0,
            np.array([-self.size[1], -self.size[0], 0.0]) / 2.0,
            np.array([self.size[1], -self.size[0], 0.0]) / 2.0,
        ]

        footprint: Polygon = Polygon(
            [
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                corners[0],
            ]
        )

        return footprint
