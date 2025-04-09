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

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Union


class EvaluationTask(Enum):
    """Evaluation tasks enum class.

    # 3D
    - DETECTION
    - TRACKING
    - PREDICTION
    - SENSING

    # 2D
    - DETECTION2D
    - TRACKING2D
    - CLASSIFICATION2D

    ## False Positive validation
    - FP_VALIDATION
    - FP_VALIDATION2D
    """

    # 3D
    DETECTION = "detection"
    TRACKING = "tracking"
    PREDICTION = "prediction"
    SENSING = "sensing"

    # 2D
    DETECTION2D = "detection2d"
    TRACKING2D = "tracking2d"
    CLASSIFICATION2D = "classification2d"

    # False Positive validation
    FP_VALIDATION = "fp_validation"
    FP_VALIDATION2D = "fp_validation2d"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[EvaluationTask, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def is_3d(self) -> bool:
        return self in (
            EvaluationTask.DETECTION,
            EvaluationTask.TRACKING,
            EvaluationTask.PREDICTION,
            EvaluationTask.SENSING,
            EvaluationTask.FP_VALIDATION,
        )

    def is_2d(self) -> bool:
        return not self.is_3d()

    def is_fp_validation(self) -> bool:
        """Indicates whether evaluation task is FP validation.

        Returns:
            bool: Return `True` if `FP_VALIDATION` of `FP_VALIDATION2D`.
        """
        return self in (EvaluationTask.FP_VALIDATION, EvaluationTask.FP_VALIDATION2D)

    @classmethod
    def from_value(cls, name: str) -> EvaluationTask:
        for _, v in cls.__members__.items():
            if v == name:
                return v
        raise ValueError(f"Unexpected value: {name}")


def set_task_lists(evaluation_tasks_str: List[str]) -> List[EvaluationTask]:
    """Convert str to EvaluationTask instances list.

    Args:
        evaluation_tasks_str (List[str]): Task names list in string.

    Returns:
        List[EvaluationTask]: Tasks list to be evaluated.

    Examples:
        >>> set_task_lists(["detection", "tracking"])
        [<EvaluationTask.DETECTION: 'detection'>, <EvaluationTask.TRACKING: 'tracking'>]
    """
    task_lists: List[EvaluationTask] = []
    for evaluation_task_str in evaluation_tasks_str:
        for task in EvaluationTask:
            if evaluation_task_str == task.value:
                task_lists.append(task)
    return task_lists


def set_task_dict(evaluation_tasks_dict: Dict[str, Dict[str, Any]]) -> Dict[EvaluationTask, Dict[str, Any]]:
    """Convert str key to EvaluationTask instance dict.

    Args:
        evaluation_tasks_str (Dict[str, Dict[str, Any]]): Dict object keyed by task name.

    Returns:
        Dict[EvaluationTask, Dict[str, Any]]: Dict object keyed by EvaluationTask instance.

    Examples:
        >>> data = {"foo": 1, "bar": 2}
        >>> set_task_dict(dict(detection=data))
        {<EvaluationTask.DETECTION: 'detection'>: {"foo": 1, "bar": 2}}
    """
    task_dict: Dict[EvaluationTask, Dict[str, Any]] = {}
    for key, item in evaluation_tasks_dict.items():
        for task in EvaluationTask:
            if key == task.value:
                task_dict[task] = item
    return task_dict


def set_task(task_name: str) -> EvaluationTask:
    """Convert string task name to EvaluationTask instance.

    Args:
        task_name (str): Task name in string.

    Returns:
        EvaluationTask: EvaluationTask instance.

    Examples:
        >>> set_task("detection")
        <EvaluationTask.DETECTION: 'detection'>
    """
    for task in EvaluationTask:
        if task_name == task.value:
            return task
