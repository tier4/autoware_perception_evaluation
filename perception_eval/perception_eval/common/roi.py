# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

from perception_eval.common.label import AutowareLabel
from perception_eval.common.object_base import Object2DBase
from perception_eval.common.object_base import Roi
from perception_eval.common.status import Visibility


class RoiObject(Object2DBase):
    """ROI object class.

    Attributes:
        self.unix_time (int): Unix time.
        self.roi (Roi): Roi instance.
        self.semantic_score (float): Semantic score.
        self.semantic_label (AutowareLabel): Label name.
        self.uuid (Optional[str]): Unique object ID.
        self.visibility (Optional[Visibility]): Object's visibility.
    """

    def __init__(
        self,
        unix_time: int,
        roi: Roi,
        semantic_score: float,
        semantic_label: AutowareLabel,
        uuid: Optional[str] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        """[summary]
        Args:
            unix_time (int): Unix time.
            roi (Roi): Roi instance.
            semantic_score (float): Semantic score.
            semantic_label (AutowareLabel): Label name.
            uuid (Optional[str]): Unique object ID. Defaults to None.
            visibility (Optional[Visibility]): Object's visibility. Defaults None.
        """
        super().__init__(
            unix_time=unix_time,
            semantic_score=semantic_score,
            roi=roi,
            uuid=uuid,
            visibility=visibility,
        )
        self.__semantic_label: AutowareLabel = semantic_label

    @property
    def semantic_label(self) -> AutowareLabel:
        return self.__semantic_label
