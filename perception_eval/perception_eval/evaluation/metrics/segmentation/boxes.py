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

"""Detection ground-truth boxes of a segmentation frame, in the box-row convention.

The partial-detection metric groups points inside detection boxes. Boxes come from
``SegmentationFrame.gt_objects`` and are converted through the shared detection box adapter (so
``dx`` is the length and ``dy`` the width), then mapped from their ``AutowareLabel`` to the
segmentation class index the interior points should carry.
"""

from __future__ import annotations

from typing import Mapping
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.common.object import DynamicObject
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.detection.adapter import objects_to_box_rows
from perception_eval.evaluation.metrics.detection.geometry import BOX_DIM


def gt_boxes_from_objects(
    objects: Sequence[DynamicObject],
    transforms: TransformDict,
    box_label_to_seg_class: Mapping[str, str],
    class_names: Sequence[str],
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Box rows ``(M, 9)``, segmentation class index and detection label index per object.

    Objects whose label is not in ``box_label_to_seg_class`` get ``-1`` in both index arrays and
    are ignored by the partial-detection metric. The detection label index is the position of the
    label name in ``box_label_to_seg_class`` (definition order), which names the ``pd_score_<name>`` keys.
    """
    if len(objects) == 0:
        return np.zeros((0, BOX_DIM), dtype=np.float64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    det_names = list(box_label_to_seg_class.keys())
    class_index = {name: index for index, name in enumerate(class_names)}
    rows = objects_to_box_rows(objects, transforms)
    seg_class = np.full(len(objects), -1, dtype=np.int64)
    det_label = np.full(len(objects), -1, dtype=np.int64)
    for index, obj in enumerate(objects):
        label_name = str(obj.semantic_label.label.value)
        if label_name not in box_label_to_seg_class:
            continue
        det_label[index] = det_names.index(label_name)
        seg_class[index] = class_index[box_label_to_seg_class[label_name]]
    return rows, seg_class, det_label
