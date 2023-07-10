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

from typing import Dict
from typing import List
from typing import Tuple

import numpy as np


class ColorMap:
    """The color map class.

    Attributes:
        self.rgb (bool): Whether use RGB order.
    """

    def __init__(self, rgb: bool = True) -> None:
        """[summary]

        Args:
            rgb (bool): Whether use RGB order. Defaults to True.
        """
        self.__rgb: bool = rgb
        self.__cmap: np.ndarray = _colormap(self.__rgb)
        self.__simple_cmap: Dict[str, Tuple[float, float, float]] = {
            "red": np.array((255, 0, 0)),
            "green": np.array((0, 255, 0)),
            "blue": np.array((0, 0, 255)),
            "yellow": np.array((255, 215, 0)),
            "cyan": np.array((0, 255, 255)),
            "purple": np.array((125, 0, 205)),
            "orange": np.array((255, 165, 0)),
            "black": np.array((0, 0, 0)),
        }
        self.__ids: List[str] = []

    def get(self, uuid: str, normalize: bool = True) -> np.ndarray:
        """[summary]
        Returns unique color has not been used yet.
        If the cached uuid is specified it returns same color.

        Args:
            uuid (str): The string id.
            normalize (bool): Whether normalize color code. Defaults to True.

        Returns:
            np.ndarray: The 3D array. If self.rgb is True, return RGB order, else BGR.
        """
        if self.is_unique(uuid):
            index: int = len(self.__ids) % 79
            self.__ids.append(uuid)
        index: int = self.__ids.index(uuid) % 79
        return self[index] / 255.0 if normalize else self[index]

    def get_simple(self, key: str, normalize: bool = True) -> np.ndarray:
        """[summary]
        Returns simple color, [red, green, blue, cyan, orange, black]

        Args:
            key (str): The name of color.
            normalize (bool): Whether normalize color code.

        Returns:
            np.ndarray: The 3D array. If self.rgb is True, return RGB order, else BGR.
        """
        if key not in self.__simple_cmap:
            raise KeyError(f"Unexpected color: {key}\n Usage: {list(self.__simple_cmap.keys())}")
        color: np.ndarray = self.__simple_cmap[key]

        if not self.__rgb:
            color = color[::-1]

        if normalize:
            color = color / 255.0

        return color

    def is_unique(self, uuid: str) -> bool:
        """Check whether input uuid has not been specified yet."""
        return uuid not in self.__ids

    def __len__(self) -> int:
        return len(self.__cmap)

    def __getitem__(self, index: int) -> np.ndarray:
        index = index % 79
        return self.__cmap[index]


def _colormap(rgb: bool) -> np.ndarray:
    """[summary]
    Create color map that has 79 RGB colors.

    Args:
        rgb (bool): Whether use RGB order.

    Returns:
        color_list (np.ndarray): The 3D array. If self.rgb is True, return RGB order, else BGR.
    """
    color_list: np.ndarray = np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list
