from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Dict

__all__ = ("LabelParam",)


@dataclass
class LabelParam:
    label_prefix: str
    merge_similar_labels: bool = False
    count_label_number: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> LabelParam:
        label_prefix: str = cfg.get("label_prefix", "autoware")
        merge_similar_labels: bool = cfg.get("merge_similar_labels", False)
        count_label_number: bool = cfg.get("count_label_number", True)
        return cls(label_prefix, merge_similar_labels, count_label_number)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
