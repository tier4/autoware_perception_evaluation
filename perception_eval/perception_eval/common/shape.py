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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.geometry import shape


class ShapeType(Enum):
    """Type of shape.

    BOUNDING_BOX
    POLYGON
    """

    BOUNDING_BOX = "bounding_box"
    POLYGON = "polygon"

    @classmethod
    def from_value(cls, name: str) -> ShapeType:
        """Returns ShapeType instance from string.

        Args:
            name (str): ShapeType name in string.

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

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[str, ShapeType]) -> bool:
        return self.value == other if isinstance(other, str) else super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class Shape:
    """Class for object's shape.

    Attributes:
        type (ShapeType): Type of shape.
        size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
            - BOUNDING_BOX:
                (width, length, height)
            - POLYGON:
                (0.0, 0.0, height)
        footprint (Optional[Polygon]): When shape is BOUNDING_BOX, it is allowed to be None.
            Footprint should be with respect to each object's coordinate system.

    Args:
        shape_type (Union[str, ShapeType]): Type of shape, BOUNDING_BOX or POLYGON.
        size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
            - BOUNDING_BOX:
                (width, length, height)
            - POLYGON:
                (0.0, 0.0, height)
        footprint (Optional[Polygon]): When shape type is BOUNDING_BOX, this does not need to be set.
            However, when POLYGON, this must be set. Defaults to None.
            Footprint should be with respect to each object's coordinate system.
    """

    def __init__(
        self,
        shape_type: Union[str, ShapeType],
        size: Tuple[float, float, float],
        footprint: Optional[Polygon] = None,
    ) -> None:
        if isinstance(shape_type, str):
            shape_type = ShapeType.from_value(shape_type)

        self.type: ShapeType = shape_type
        self.size: Tuple[float, float, float] = size
        self.footprint: Optional[Polygon] = footprint if footprint else self.__calculate_corners(shape_type, size)

    def __reduce__(self) -> Tuple[Shape, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (self.__class__, (self.type, self.size, self.footprint))

    @staticmethod
    def __calculate_corners(shape_type: ShapeType, size: Tuple[float, float, float]) -> Polygon:
        """Calculate footprint for type of BOUNDING_BOX with respect to each object's coordinate system.
        This method is only called if input shape type is `BOUNDING_BOX` to set footprint from .

        Args:
            shape_type (ShapeType): Shape type of target shape.
            size (Tuple[float, float, float]): (width, length, height).

        Returns:
            corner_polygon (Polygon): Object's corners as polygon.
        """
        if shape_type != ShapeType.BOUNDING_BOX:
            raise ValueError(f"Expected BOUNDING_BOX, but got {shape_type}, which should have footprint.")

        corners: List[np.ndarray] = [
            np.array([size[1], size[0], 0.0]) / 2.0,
            np.array([-size[1], size[0], 0.0]) / 2.0,
            np.array([-size[1], -size[0], 0.0]) / 2.0,
            np.array([size[1], -size[0], 0.0]) / 2.0,
        ]

        corner_polygon: Polygon = Polygon(
            [
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                corners[0],
            ]
        )

        return corner_polygon

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "shape_type": self.type.value,
            "size": self.size,
            "footprint": mapping(self.footprint) if self.footprint else None,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> shape:
        """Deserialize data to Shape."""
        return Shape(
            shape_type=data["shape_type"],
            size=data["size"],
            footprint=shape(data["footprint"]) if data["footprint"] else None,
        )
