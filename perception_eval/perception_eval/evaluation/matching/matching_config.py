from typing import NamedTuple

from .object_matching import MatchingMode


class MatchingConfig(NamedTuple):
    mode: MatchingMode
    threshold: float

    def __str__(self) -> str:
        return f"{self.mode.name}_{self.threshold}"
