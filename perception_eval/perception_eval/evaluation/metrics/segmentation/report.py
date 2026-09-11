# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scene-level result of the segmentation metric suite."""

from __future__ import annotations

import json
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.class_groups import fold_confusion
from perception_eval.evaluation.metrics.component import MetricReport
from perception_eval.evaluation.metrics.segmentation.state import RAW_TAXONOMY


class SegmentationMetricsReport:
    """Metric values plus the raw confusion matrices of one segmentation scene.

    Attributes:
        report (MetricReport): Slash-keyed values, coverage and warnings.
        confusion (NDArray): ``(F+1, R+1, C, C)`` int64 confusion matrices (trained taxonomy).
        class_names (tuple[str, ...]): Trained class names.
        group_names (tuple[str, ...] | None): Behaviour-group names when configured.
        group_lut (NDArray | None): Trained-index to group-index LUT when configured.
        filter_names (tuple[str, ...]): Filter names, ``""`` first for the identity filter.
        range_names (tuple[str, ...]): Range names, ``""`` first for the whole scene.
        num_frames (int): Frames accumulated.
    """

    def __init__(
        self,
        report: MetricReport,
        confusion: NDArray,
        class_names: Sequence[str],
        group_names: Optional[Sequence[str]],
        group_lut: Optional[NDArray],
        filter_names: Sequence[str],
        range_names: Sequence[str],
        num_frames: int,
    ) -> None:
        self.report = report
        self.confusion = np.asarray(confusion, dtype=np.int64)
        self.class_names = tuple(class_names)
        self.group_names = None if group_names is None else tuple(group_names)
        self.group_lut = None if group_lut is None else np.asarray(group_lut, dtype=np.int64)
        self.filter_names = tuple(filter_names)
        self.range_names = tuple(range_names)
        self.num_frames = int(num_frames)

    # ------------------------------------------------------------------------------------------
    @property
    def values(self) -> Dict[str, float]:
        return self.report.values

    @property
    def coverage(self) -> Dict[str, Tuple[int, int]]:
        return self.report.coverage

    @property
    def warnings(self) -> List[str]:
        return self.report.warnings

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.report.get(key, default)

    def to_flat_keys(self, separator: str = "_") -> Dict[str, float]:
        return self.report.to_flat_keys(separator)

    def confusion_matrix(
        self, taxonomy: str = RAW_TAXONOMY, filter_name: str = "", range_name: str = ""
    ) -> NDArray[np.int64]:
        """The ``(C, C)`` matrix of one view (folded lazily for the grouped taxonomy)."""
        if filter_name not in self.filter_names:
            raise KeyError(f"unknown filter {filter_name!r}; known: {list(self.filter_names)}")
        if range_name not in self.range_names:
            raise KeyError(f"unknown range {range_name!r}; known: {list(self.range_names)}")
        matrix = self.confusion[self.filter_names.index(filter_name), self.range_names.index(range_name)]
        if taxonomy == RAW_TAXONOMY:
            return matrix
        if self.group_lut is None or self.group_names is None:
            raise KeyError("no class groups configured, only the raw taxonomy is available.")
        return fold_confusion(matrix, self.group_lut, len(self.group_names))

    # ------------------------------------------------------------------------------------------
    def __str__(self) -> str:
        lines = [f"Segmentation metrics over {self.num_frames} frame(s), classes: {list(self.class_names)}"]
        if self.group_names is not None:
            lines.append(f"Groups: {list(self.group_names)}")
        lines.append(self.report.summary(max_rows=200))
        return "\n".join(lines)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.report,
                self.confusion,
                self.class_names,
                self.group_names,
                self.group_lut,
                self.filter_names,
                self.range_names,
                self.num_frames,
            ),
        )

    def serialization(self) -> Dict[str, Any]:
        return {
            "report": self.report.serialization(),
            "confusion": self.confusion.tolist(),
            "class_names": list(self.class_names),
            "group_names": None if self.group_names is None else list(self.group_names),
            "group_lut": None if self.group_lut is None else self.group_lut.tolist(),
            "filter_names": list(self.filter_names),
            "range_names": list(self.range_names),
            "num_frames": self.num_frames,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> SegmentationMetricsReport:
        return cls(
            report=MetricReport.deserialization(data["report"]),
            confusion=np.asarray(data["confusion"], dtype=np.int64),
            class_names=data["class_names"],
            group_names=data.get("group_names"),
            group_lut=data.get("group_lut"),
            filter_names=data["filter_names"],
            range_names=data["range_names"],
            num_frames=data["num_frames"],
        )

    def to_json(self, path: str, nan_as_null: bool = False) -> None:
        """Write :meth:`serialization` as JSON (NaN tokens unless ``nan_as_null``)."""
        data = self.serialization()
        if nan_as_null:
            data["report"]["values"] = {k: (None if math.isnan(v) else v) for k, v in data["report"]["values"].items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, allow_nan=not nan_as_null)
