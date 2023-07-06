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

from typing import Union

from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_result import get_object_status  # noqa
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.evaluation.sensing.sensing_frame_result import SensingFrameResult
from perception_eval.evaluation.sensing.sensing_result import DynamicObjectWithSensingResult

# type aliases
ObjectResultType = Union[DynamicObjectWithPerceptionResult, DynamicObjectWithSensingResult]
FrameResultType = Union[PerceptionFrameResult, SensingFrameResult]
