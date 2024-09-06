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

from copy import deepcopy
from typing import Tuple

from perception_eval.common.object import DynamicObject
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


def prepare_path(
    object_result: DynamicObjectWithPerceptionResult,
    top_k: int,
) -> Tuple[DynamicObject, DynamicObject]:
    """Extract top-K modes.

    Args:
        object_result (DynamicObjectResult)
        top_k (int)

    Returns:
        estimated_object_ (DynamicObject)
        ground_truth_object_ (DynamicObject)
    """
    if object_result.ground_truth_object is None:
        raise RuntimeError("Object result's ground truth object must be set")

    estimated_object_ = deepcopy(object_result.estimated_object)
    ground_truth_object_ = deepcopy(object_result.ground_truth_object)

    estimated_object_.predicted_paths.sort(key=lambda x: x.confidence, reverse=True)
    estimated_object_.predicted_paths = estimated_object_.predicted_paths[:top_k]

    num_stack: int = min(top_k, len(estimated_object_.predicted_paths))
    for _ in range(num_stack - 1):
        ground_truth_object_.predicted_paths.append(ground_truth_object_.predicted_paths)

    return estimated_object_, ground_truth_object_
