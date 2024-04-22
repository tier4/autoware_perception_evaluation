from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from typing import Any
from typing import Dict


class BaseParam(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, cfg: Dict[str, Any], *args, **kwargs) -> BaseParam:
        pass

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
