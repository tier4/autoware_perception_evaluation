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
        object_result (DynamicObjectResult): Object result.
        top_k (int): The number of top-K paths to be evaluated.

    Returns:
        tuple[DynamicObject, DynamicObject]: Estimated object and GT object.
    """
    if object_result.ground_truth_object is None:
        raise RuntimeError("Object result's ground truth object must be set")

    estimated_object_ = deepcopy(object_result.estimated_object)
    ground_truth_object_ = deepcopy(object_result.ground_truth_object)

    num_gt_future = len(object_result.ground_truth_object.predicted_paths[0])

    estimated_object_.predicted_paths.sort(key=lambda x: x.confidence, reverse=True)
    for i, path in enumerate(estimated_object_.predicted_paths):
        num_estimated_future =len(path) 
        if num_estimated_future < num_gt_future:
            # pad by the last state
            pad_states = path.states + [path.states[-1] for _ in range(num_gt_future - num_estimated_future)]
        else:
            pad_states = path.states[:num_gt_future]

        estimated_object_.predicted_paths[i].states = pad_states

    estimated_object_.predicted_paths = estimated_object_.predicted_paths[:top_k]

    num_stack: int = min(top_k, len(estimated_object_.predicted_paths))
    ground_truth_object_.predicted_paths = ground_truth_object_.predicted_paths * num_stack

    return estimated_object_, ground_truth_object_
