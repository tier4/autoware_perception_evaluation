class SensingFrameConfig:
    """[summary]
    Config class for sensing detection evaluation per frame.

    Attributes:
        self.box_scale_0m (float): Scale factor for bounding box at 0m.
        self.box_scale_100m (float): Scale factor for bounding box at 100m.
        self.min_points_threshold (int): The minimum number of points should be detected in bounding box.
    """

    def __init__(
        self,
        box_scale_0m: float,
        box_scale_100m: float,
        min_points_threshold: int,
    ) -> None:
        """[summary]
        Args:
            box_scale_0m (float): Scale factor for bounding box at 0m.
            box_scale_100m (float): Scale factor for bounding box at 100m.
            min_points_threshold (int): The minimum number of points should be detected in bounding box.
        """
        self.box_scale_0m: float = box_scale_0m
        self.box_scale_100m: float = box_scale_100m
        self.min_points_threshold: int = min_points_threshold

        self.scale_slope_: float = 0.01 * (box_scale_100m - box_scale_0m)

    def get_scale_factor(self, distance: float) -> float:
        """Calculate scale factor linearly for bounding box at specified distance.

        Note:
            scale = ((box_scale_100m - box_scale_0m) / (100 - 0)) * (distance - 0) + box_scale_0m

        Args:
            distance (float): The distance from vehicle to target bounding box.

        Returns:
            float: Calculated scale factor.
        """
        return self.scale_slope_ * distance + self.box_scale_0m
