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

from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class ShapeType(Enum):
    """[summary]
    Type of shape
        - BOUNDING_BOX
        - CYLINDER
        - POLYGON
    """

    BOUNDING_BOX = "bounding_box"
    CYLINDER = "cylinder"
    POLYGON = "polygon"


class Shape:
    """[summary]
    Shape class.

    NOTE: Footprint of cylinder is rectangle.

    Attributes:
        self.type (ShapeType): Type of shape.
        self.size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
            - BOUNDING_BOX:
                (width, length, height)
            - CYLINDER:
                (diameter, 0.0, height)
            - POLYGON:
                (0.0, 0.0, height)
        self.footprint (Optional[Polygon]): When shape is BOUNDING_BOX, it is allowed to be None.
    """

    def __init__(
        self,
        type: ShapeType,
        size: Tuple[float, float, float],
        footprint: Optional[Polygon] = None,
    ) -> None:
        """
        Args:
            type (ShapeType): Type of shape, BOUNDING_BOX, CYLINDER or POLYGON
            size (Tuple[float, float, float]): Size of bbox, (width, length, height) order.
                - BOUNDING_BOX:
                    (width, length, height)
                - CYLINDER:
                    (diameter, 0.0, height)
                - POLYGON:
                    (0.0, 0.0, height)
            footprint (Optional[Polygon]): When shape type is BOUNDING_BOX or CYLINDER, this needs not to be set.
                However, when POLYGON, this must be set. Defaults to None.
        """
        if type == ShapeType.POLYGON and footprint is None:
            raise RuntimeError("For POLYGON, footprint must be set")

        self.type: ShapeType = type
        self.size = (size[0], size[0], size[2]) if type == ShapeType.CYLINDER else size
        self.footprint: Optional[Polygon] = footprint


def set_footprint(
    position: Tuple[float, float, float],
    orientation: Quaternion,
    shape: Shape,
) -> Shape:
    """[summary]
    Calculate footprint for type of BOUNDING_BOX or CYLINDER.
    If footprint has been already set, do nothing.
    Returns:
        shape (Shape)
    """
    if shape.type == ShapeType.POLYGON:
        raise RuntimeError("Expected BOUNDING_BOX or CYLINDER")

    corners: List[np.ndarray] = [
        np.array([shape.size[1], shape.size[0], 0.0]) / 2.0,
        np.array([-shape.size[1], shape.size[0], 0.0]) / 2.0,
        np.array([-shape.size[1], -shape.size[0], 0.0]) / 2.0,
        np.array([shape.size[1], -shape.size[0], 0.0]) / 2.0,
    ]

    # rotate vector_center_to_corners
    rotated_corners: List[Tuple[float, float, float]] = []
    for vertex in corners:
        rotated_vertex: np.ndarray = orientation.rotate(vertex)
        rotated_vertex[:2]: np.ndarray = rotated_vertex[:2] + position[:2]
        rotated_corners.append(rotated_vertex.tolist())
    # corner point to footprint
    shape.footprint: Polygon = Polygon(
        [
            rotated_corners[0],
            rotated_corners[1],
            rotated_corners[2],
            rotated_corners[3],
            rotated_corners[0],
        ]
    )

    return shape
