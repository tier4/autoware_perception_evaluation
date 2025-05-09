from typing import NamedTuple

from .object_matching import MatchingMode


class MatchingConfig(NamedTuple):
    """
    Configuration class representing the matching strategy and its associated threshold.

    Attributes:
        mode (MatchingMode): The matching mode used for evaluation (e.g., center distance, IoU).
        threshold (float): The threshold value used in the matching criterion (e.g., distance threshold or IoU threshold).

    This class provides a string representation in the format "{mode_name}_{threshold}", which is useful for logging and dictionary keys.
    """

    mode: MatchingMode
    threshold: float

    def __str__(self) -> str:
        return f"{self.mode.name}_{self.threshold}"
