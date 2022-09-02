from __future__ import annotations

from enum import Enum
from typing import Dict


class Visibility(Enum):
    """[summary]"""

    FULL = "full"
    MOST = "most"
    PARTIAL = "partial"
    NONE = "none"

    @staticmethod
    def from_alias(name: str) -> Dict[str, Visibility]:
        if name == "v0-40":
            return Visibility.NONE
        elif name == "v40-60":
            return Visibility.PARTIAL
        elif name == "v60-80":
            return Visibility.MOST
        elif name == "v80-100":
            return Visibility.FULL
        else:
            raise ValueError(f"{Visibility.__class__.__name__}has not {name}")

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.value == __o
        return super().__eq__(__o)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, name: str) -> Visibility:
        for k, v in cls.__members__.items():
            if v == name:
                return k
        return cls.from_alias(name)
