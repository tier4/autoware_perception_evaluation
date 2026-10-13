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

"""Prepared samples: the map-free intermediate that can be pooled across scenes of the advanced detection metrics.

``AdvancedDetectionSuite.prepare()`` converts a scene's frames into :class:`PreparedSamples`
(base_link box rows, scores, labels, optional reachability TTC and every filter mask). Preparing
is the only step that needs the lanelet map, so a prepared scene can be written to one ``.npz``
file, read back anywhere and pooled with other scenes:
``evaluate_prepared(config, [a, b, c])`` is bit-identical to ``suite.evaluate(frames_a + frames_b
+ frames_c)``.

The ``.npz`` layout (schema version :data:`PREPARED_SCHEMA_VERSION`) holds no object arrays and is
read with ``allow_pickle=False``. See ``docs/en/perception/advanced_metrics_files.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.detection.state import DetectionSample

if TYPE_CHECKING:  # pragma: no cover - typing only
    from perception_eval.evaluation.metrics.component import MetricReport
    from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig

PREPARED_SCHEMA_VERSION: int = 1
"""Schema version written into the ``.npz`` header; readers reject any other version."""

BOX_COLUMNS: Tuple[str, ...] = ("cx", "cy", "cz", "dx", "dy", "dz", "yaw", "vx", "vy")
"""Column order of the ``pred_boxes`` / ``gt_boxes`` rows (base_link)."""

COUNT_KEYS: Tuple[str, ...] = (
    "input_frames",
    "skipped_frames",
    "polygon_as_box",
    "non_finite_score",
    "pred_total",
    "pred_kept",
    "zero_score",
)
"""Counter names of :attr:`PreparedSamples.counts`; every key is always present."""

FRAME_NAME_DTYPE = "<U64"


class PoolingError(ValueError):
    """Raised when prepared samples cannot be pooled or evaluated with the given configuration."""


def config_digest(config: AdvancedDetectionMetricsConfig) -> str:
    """``sha256:<hex>`` of the canonical JSON of the configuration that shapes prepared samples.

    The digest covers ``class_names`` and ``config.serialization()`` without ``map.data_root`` and
    ``map.mapping`` (where the maps live does not change the numbers), plus the schema version.
    Canonical JSON means ``sort_keys=True`` and ``separators=(",", ":")``, so the digest is
    independent of dict insertion order.
    """
    advanced = config.serialization()
    map_section = advanced.get("map")
    if isinstance(map_section, Mapping):
        advanced["map"] = {key: value for key, value in map_section.items() if key not in ("data_root", "mapping")}
    payload = {
        "class_names": list(config.class_names),
        "advanced": advanced,
        "prepared_schema_version": PREPARED_SCHEMA_VERSION,
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Canonical JSON text (sorted keys, no whitespace) used by every digest in this module."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PreparedFrame:
    """One frame converted to arrays plus its per-filter masks and availability.

    Attributes:
        sample (DetectionSample): Base_link box rows, scores, labels and optional TTC.
        available (Mapping[str, bool]): Per non-identity filter name, whether the filter could
            evaluate this frame (ego pose present, lanelet map resolved).
        pred_masks (Mapping[str, NDArray[bool]]): Per filter name, the predictions the filter keeps;
            all-False when the filter is unavailable for the frame.
        gt_masks (Mapping[str, NDArray[bool]]): Same for the ground truths.
        unix_time (int): Frame timestamp, carried for the ``.npz`` file only.
    """

    sample: DetectionSample
    available: Mapping[str, bool]
    pred_masks: Mapping[str, NDArray[np.bool_]]
    gt_masks: Mapping[str, NDArray[np.bool_]]
    unix_time: int = 0


@dataclass(frozen=True)
class PreparedSamples:
    """Every prepared frame of one scene (or of several pooled scenes) plus its provenance.

    Attributes:
        schema_version (int): :data:`PREPARED_SCHEMA_VERSION` of the producer.
        config_digest (str): :func:`config_digest` of the configuration used to prepare.
        class_names (tuple[str, ...]): Label index -> class name.
        filter_names (tuple[str, ...]): Non-identity filter names in configuration order.
        has_ttc (bool): Whether the samples carry reachability TTC (collision provider ran).
        frames (tuple[PreparedFrame, ...]): Prepared frames in input order (skipped frames absent).
        counts (Mapping[str, int]): Counters named in :data:`COUNT_KEYS`.
        warnings (tuple[str, ...]): Preparation warnings (skipped frames, dropped predictions, ...).
    """

    schema_version: int
    config_digest: str
    class_names: Tuple[str, ...]
    filter_names: Tuple[str, ...]
    has_ttc: bool
    frames: Tuple[PreparedFrame, ...]
    counts: Mapping[str, int]
    warnings: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_names", tuple(str(name) for name in self.class_names))
        object.__setattr__(self, "filter_names", tuple(str(name) for name in self.filter_names))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        counts = {key: 0 for key in COUNT_KEYS}
        counts.update({str(key): int(value) for key, value in dict(self.counts).items()})
        object.__setattr__(self, "counts", counts)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    # ------------------------------------------------------------------------------------------
    # npz
    # ------------------------------------------------------------------------------------------

    def to_npz(self, path: Any, producer: Optional[Mapping[str, str]] = None) -> str:
        """Write the samples as ``np.savez_compressed`` arrays (no object arrays).

        Args:
            path (str | PathLike): Output file; written exactly as given (no ``.npz`` appended).
            producer (Mapping[str, str] | None): Free-form producer identification for the header.

        Returns:
            str: sha256 hex digest of the written file.
        """
        arrays = self._to_arrays(producer)
        with open(path, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _to_arrays(self, producer: Optional[Mapping[str, str]]) -> Dict[str, NDArray]:
        num_filters = len(self.filter_names)
        frames = self.frames
        pred_counts = [frame.sample.num_predictions for frame in frames]
        gt_counts = [frame.sample.num_ground_truths for frame in frames]
        header = {
            "schema_version": int(self.schema_version),
            "config_digest": self.config_digest,
            "class_names": list(self.class_names),
            "filter_names": list(self.filter_names),
            "has_ttc": bool(self.has_ttc),
            "box_columns": list(BOX_COLUMNS),
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
            "producer": {str(key): str(value) for key, value in dict(producer or {}).items()},
        }
        arrays: Dict[str, NDArray] = {
            "header": np.frombuffer(json.dumps(header, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
            "frame_name": np.asarray([frame.sample.frame_name for frame in frames], dtype=FRAME_NAME_DTYPE),
            "unix_time": np.asarray([int(frame.unix_time) for frame in frames], dtype=np.int64),
            "ttc_covered": np.asarray([bool(frame.sample.ttc_covered) for frame in frames], dtype=bool),
            "available": np.asarray(
                [[bool(frame.available.get(name, False)) for name in self.filter_names] for frame in frames],
                dtype=bool,
            ).reshape(len(frames), num_filters),
            "pred_offsets": _offsets(pred_counts),
            "gt_offsets": _offsets(gt_counts),
            "pred_boxes": _concat([frame.sample.pred_boxes for frame in frames], (0, len(BOX_COLUMNS)), np.float64),
            "gt_boxes": _concat([frame.sample.gt_boxes for frame in frames], (0, len(BOX_COLUMNS)), np.float64),
            "pred_scores": _concat([frame.sample.pred_scores for frame in frames], (0,), np.float64),
            "pred_labels": _concat([frame.sample.pred_labels for frame in frames], (0,), np.int64),
            "gt_labels": _concat([frame.sample.gt_labels for frame in frames], (0,), np.int64),
            "pred_masks": _concat(
                [_stack_masks(frame.pred_masks, self.filter_names, count) for frame, count in zip(frames, pred_counts)],
                (0, num_filters),
                bool,
            ),
            "gt_masks": _concat(
                [_stack_masks(frame.gt_masks, self.filter_names, count) for frame, count in zip(frames, gt_counts)],
                (0, num_filters),
                bool,
            ),
        }
        if self.has_ttc:
            arrays["pred_ttc"] = _concat(
                [_ttc_or_inf(frame.sample.pred_ttc, count) for frame, count in zip(frames, pred_counts)],
                (0,),
                np.float64,
            )
            arrays["gt_ttc"] = _concat(
                [_ttc_or_inf(frame.sample.gt_ttc, count) for frame, count in zip(frames, gt_counts)],
                (0,),
                np.float64,
            )
        return arrays

    @classmethod
    def from_npz(cls, path: Any) -> PreparedSamples:
        """Read a file written by :meth:`to_npz` (``allow_pickle=False``).

        Raises:
            PoolingError: the header is missing or carries an unknown ``schema_version``.
        """
        with np.load(path, allow_pickle=False) as data:
            if "header" not in data.files:
                raise PoolingError(f"{path}: not a prepared-samples file (no header).")
            header = json.loads(bytes(np.asarray(data["header"], dtype=np.uint8)).decode("utf-8"))
            version = header.get("schema_version")
            if version != PREPARED_SCHEMA_VERSION:
                raise PoolingError(
                    f"{path}: prepared samples schema version {version!r} is not supported "
                    f"(this reader handles {PREPARED_SCHEMA_VERSION})."
                )
            filter_names = tuple(str(name) for name in header["filter_names"])
            has_ttc = bool(header["has_ttc"])
            frame_names = np.asarray(data["frame_name"]).tolist()
            unix_times = np.asarray(data["unix_time"], dtype=np.int64).tolist()
            ttc_covered = np.asarray(data["ttc_covered"], dtype=bool).tolist()
            available = np.asarray(data["available"], dtype=bool).reshape(len(frame_names), len(filter_names))
            pred_offsets = np.asarray(data["pred_offsets"], dtype=np.int64)
            gt_offsets = np.asarray(data["gt_offsets"], dtype=np.int64)
            pred_boxes = np.asarray(data["pred_boxes"], dtype=np.float64)
            gt_boxes = np.asarray(data["gt_boxes"], dtype=np.float64)
            pred_scores = np.asarray(data["pred_scores"], dtype=np.float64)
            pred_labels = np.asarray(data["pred_labels"], dtype=np.int64)
            gt_labels = np.asarray(data["gt_labels"], dtype=np.int64)
            pred_masks = np.asarray(data["pred_masks"], dtype=bool).reshape(pred_boxes.shape[0], len(filter_names))
            gt_masks = np.asarray(data["gt_masks"], dtype=bool).reshape(gt_boxes.shape[0], len(filter_names))
            pred_ttc = np.asarray(data["pred_ttc"], dtype=np.float64) if has_ttc else None
            gt_ttc = np.asarray(data["gt_ttc"], dtype=np.float64) if has_ttc else None

        frames: List[PreparedFrame] = []
        for index, frame_name in enumerate(frame_names):
            pred_slice = slice(int(pred_offsets[index]), int(pred_offsets[index + 1]))
            gt_slice = slice(int(gt_offsets[index]), int(gt_offsets[index + 1]))
            sample = DetectionSample(
                pred_boxes=pred_boxes[pred_slice],
                pred_scores=pred_scores[pred_slice],
                pred_labels=pred_labels[pred_slice],
                gt_boxes=gt_boxes[gt_slice],
                gt_labels=gt_labels[gt_slice],
                gt_ttc=None if gt_ttc is None else gt_ttc[gt_slice],
                pred_ttc=None if pred_ttc is None else pred_ttc[pred_slice],
                ttc_covered=bool(ttc_covered[index]),
                frame_name=str(frame_name),
            )
            frames.append(
                PreparedFrame(
                    sample=sample,
                    available={name: bool(available[index, column]) for column, name in enumerate(filter_names)},
                    pred_masks={
                        name: np.ascontiguousarray(pred_masks[pred_slice, column])
                        for column, name in enumerate(filter_names)
                    },
                    gt_masks={
                        name: np.ascontiguousarray(gt_masks[gt_slice, column])
                        for column, name in enumerate(filter_names)
                    },
                    unix_time=int(unix_times[index]),
                )
            )
        return cls(
            schema_version=int(version),
            config_digest=str(header["config_digest"]),
            class_names=tuple(str(name) for name in header["class_names"]),
            filter_names=filter_names,
            has_ttc=has_ttc,
            frames=tuple(frames),
            counts=dict(header.get("counts", {})),
            warnings=tuple(header.get("warnings", ())),
        )

    # ------------------------------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------------------------------

    @classmethod
    def concat(cls, parts: Sequence[PreparedSamples]) -> PreparedSamples:
        """Pool several prepared scenes in the given order.

        Frames are concatenated, counts summed and warnings deduplicated (first occurrence wins).

        Raises:
            PoolingError: ``parts`` is empty, or a part differs in ``schema_version``,
                ``config_digest``, ``class_names``, ``filter_names`` or ``has_ttc``.
        """
        parts = list(parts)
        if not parts:
            raise PoolingError("nothing to pool: no prepared samples given.")
        reference = parts[0]
        for index, part in enumerate(parts):
            if not isinstance(part, PreparedSamples):
                raise PoolingError(f"parts[{index}] is {type(part).__name__}, expected PreparedSamples.")
            if part.schema_version != PREPARED_SCHEMA_VERSION:
                raise PoolingError(
                    f"parts[{index}]: schema version {part.schema_version!r} is not supported "
                    f"(this reader handles {PREPARED_SCHEMA_VERSION})."
                )
            for attribute in ("config_digest", "class_names", "filter_names", "has_ttc"):
                if getattr(part, attribute) != getattr(reference, attribute):
                    raise PoolingError(
                        f"parts[{index}]: {attribute} {getattr(part, attribute)!r} differs from parts[0] "
                        f"{getattr(reference, attribute)!r}; scenes prepared with different configurations "
                        "cannot be pooled."
                    )
        counts: Dict[str, int] = {key: 0 for key in COUNT_KEYS}
        warnings: List[str] = []
        frames: List[PreparedFrame] = []
        for part in parts:
            frames.extend(part.frames)
            for key, value in part.counts.items():
                counts[key] = counts.get(key, 0) + int(value)
            for warning in part.warnings:
                if warning not in warnings:
                    warnings.append(warning)
        return cls(
            schema_version=PREPARED_SCHEMA_VERSION,
            config_digest=reference.config_digest,
            class_names=reference.class_names,
            filter_names=reference.filter_names,
            has_ttc=reference.has_ttc,
            frames=tuple(frames),
            counts=counts,
            warnings=tuple(warnings),
        )


def evaluate_prepared(
    config: AdvancedDetectionMetricsConfig,
    prepared: Union[PreparedSamples, Sequence[PreparedSamples]],
) -> MetricReport:
    """Run the configured components over prepared samples without touching any lanelet map.

    Args:
        config (AdvancedDetectionMetricsConfig): The configuration the samples were prepared with.
        prepared (PreparedSamples | Sequence[PreparedSamples]): One scene, or several scenes pooled
            in the given order (see :meth:`PreparedSamples.concat`).

    Raises:
        PoolingError: ``config_digest(config)`` differs from the samples' digest, or the parts
            cannot be pooled.
    """
    from perception_eval.evaluation.metrics.detection.suite import AdvancedDetectionSuite

    if not isinstance(prepared, PreparedSamples):
        prepared = PreparedSamples.concat(list(prepared))
    return AdvancedDetectionSuite(config).evaluate_prepared(prepared)


# ----------------------------------------------------------------------------------------------
# Array helpers
# ----------------------------------------------------------------------------------------------


def _offsets(counts: Sequence[int]) -> NDArray[np.int64]:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    if counts:
        offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    return offsets


def _concat(arrays: Sequence[NDArray], empty_shape: Tuple[int, ...], dtype: Any) -> NDArray:
    if not arrays:
        return np.zeros(empty_shape, dtype=dtype)
    return np.ascontiguousarray(np.concatenate([np.asarray(array, dtype=dtype) for array in arrays], axis=0))


def _stack_masks(masks: Mapping[str, NDArray], filter_names: Sequence[str], count: int) -> NDArray[np.bool_]:
    stacked = np.zeros((count, len(filter_names)), dtype=bool)
    for column, name in enumerate(filter_names):
        mask = masks.get(name)
        if mask is not None:
            stacked[:, column] = np.asarray(mask, dtype=bool).reshape(count)
    return stacked


def _ttc_or_inf(ttc: Optional[NDArray], count: int) -> NDArray[np.float64]:
    if ttc is None:
        return np.full(count, np.inf, dtype=np.float64)
    return np.asarray(ttc, dtype=np.float64).reshape(count)
