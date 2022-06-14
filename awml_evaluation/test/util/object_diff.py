import math
from typing import Tuple


class DiffTranslation:
    """Differences of translation class for estimated and ground truth object.

    Attributes:
        self.diff_estimated (Tuple[float, float, float]): The translation difference of estimated object.
        self.diff_ground_truth (Tuple[float, float, float]): The translation difference of ground truth object.
    """

    def __init__(
        self,
        diff_estimated: Tuple[float, float, float],
        diff_ground_truth: Tuple[float, float, float],
    ) -> None:
        """[summary]
        Args:
            diff_estimated (Tuple[float, float, float]): The translation difference of estimated object.
            diff_ground_truth (Tuple[float, float, float]): The translation difference of ground truth object.
        """
        self.diff_estimated: Tuple[float, float, float] = diff_estimated
        self.diff_ground_truth: Tuple[float, float, float] = diff_ground_truth


class DiffYaw:
    """Differences of yaw class for estimated and ground truth object.

    Attributes:
        self.diff_estimated (float): The yaw difference of estimated object.
        self.diff_ground_truth (float): The yaw difference of ground truth object.
    """

    def __init__(
        self,
        diff_estimated: float,
        diff_ground_truth: float,
        deg2rad: bool = False,
    ) -> None:
        """[summary]
        Args:
            diff_estimated (float): The yaw difference of estimated object.
            diff_ground_truth (float): The yaw difference of ground truth object.
            deg2rad (bool): Whether convert degrees to radians. Defaults to False.
        """
        self.diff_estimated: float = math.radians(diff_estimated) if deg2rad else diff_estimated
        self.diff_ground_truth: float = (
            math.radians(diff_ground_truth) if deg2rad else diff_ground_truth
        )
