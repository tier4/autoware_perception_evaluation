from __future__ import annotations

from enum import Enum
from numbers import Number
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from perception_eval.common.label import is_same_label
from perception_eval.common.schema import is_same_frame_id
from perception_eval.common.threshold import get_label_threshold

from .object_matching import CenterDistanceMatching
from .object_matching import IOU2dMatching
from .object_matching import IOU3dMatching
from .object_matching import MatchingMode
from .object_matching import PlaneDistanceMatching

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.object import ObjectType

    from .object_matching import MatchingMethod


class MatchingLabelPolicy(Enum):
    STRICT = "STRICT"
    ALLOW_UNKNOWN = "ALLOW_UNKNOWN"
    ALLOW_ANY = "ALLOW_ANY"

    @classmethod
    def from_str(cls, name: str) -> MatchingLabelPolicy:
        name = name.upper()
        assert name in cls.__members__, f"{name} is not enum member"
        return cls.__members__[name]

    def is_matchable(self, estimation: ObjectType, ground_truth: ObjectType) -> bool:
        if ground_truth.semantic_label.is_fp() or self == MatchingLabelPolicy.ALLOW_ANY:
            return True
        elif self == MatchingLabelPolicy.ALLOW_UNKNOWN:
            return is_same_label(estimation, ground_truth) or estimation.semantic_label.is_unknown()
        else:  # STRICT
            return is_same_label(estimation, ground_truth)


class MatchingPolicy:
    def __init__(
        self,
        matching_mode: Optional[Union[str, MatchingMode]] = None,
        label_policy: Optional[Union[str, MatchingLabelPolicy]] = None,
        matchable_thresholds: Optional[List[Number]] = None,
    ) -> None:
        if matching_mode is None:
            self.matching_mode = MatchingMode.CENTERDISTANCE
        elif isinstance(matching_mode, str):
            self.matching_mode = MatchingMode.from_str(matching_mode)
        else:
            self.matching_mode = matching_mode

        self.matching_module, self.maximize = self.get_matching_module(self.matching_mode)

        if label_policy is None:
            self.label_policy = MatchingLabelPolicy.STRICT
        elif isinstance(label_policy, str):
            self.label_policy = MatchingLabelPolicy.from_str(label_policy)
        else:
            self.label_policy = label_policy

        self.matchable_thresholds = matchable_thresholds

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> MatchingPolicy:
        matching_mode = cfg.get("matching_mode")
        label_policy = cfg.get("matching_label_policy")
        matchable_thresholds = cfg.get("matchable_thresholds")
        return cls(matching_mode=matching_mode, label_policy=label_policy, matchable_thresholds=matchable_thresholds)

    @staticmethod
    def get_matching_module(matching_mode: MatchingMode) -> Tuple[Callable, bool]:
        if matching_mode == MatchingMode.CENTERDISTANCE:
            matching_method_module: CenterDistanceMatching = CenterDistanceMatching
            maximize: bool = False
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            matching_method_module: PlaneDistanceMatching = PlaneDistanceMatching
            maximize: bool = False
        elif matching_mode == MatchingMode.IOU2D:
            matching_method_module: IOU2dMatching = IOU2dMatching
            maximize: bool = True
        elif matching_mode == MatchingMode.IOU3D:
            matching_method_module: IOU3dMatching = IOU3dMatching
            maximize: bool = True
        else:
            raise ValueError(f"Unsupported matching mode: {matching_mode}")

        return matching_method_module, maximize

    def is_matchable(self, estimation: ObjectType, ground_truth: ObjectType) -> bool:
        return self.label_policy.is_matchable(estimation, ground_truth) and is_same_frame_id(estimation, ground_truth)

    def get_matching_score(
        self,
        estimation: ObjectType,
        ground_truth: ObjectType,
        target_labels: List[LabelType],
    ) -> Optional[float]:
        threshold: Optional[float] = get_label_threshold(
            ground_truth.semantic_label,
            target_labels,
            self.matchable_thresholds,
        )

        matching_method: MatchingMethod = self.matching_module(estimation, ground_truth)

        if threshold is None or (threshold is not None and matching_method.is_better_than(threshold)):
            return matching_method.value
        else:
            return None
