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
    """[summary]
    Evaluation tasks enum class
    """

    DETECTION = "detection"
    TRACKING = "tracking"
    PREDICTION = "prediction"
    DETECTION2D = "detection2d"
    SENSING = "sensing"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[EvaluationTask, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


def set_task_lists(evaluation_tasks_str: List[str]) -> List[EvaluationTask]:
    """[summary]
    Convert str to EvaluationTask class

    Args:
        evaluation_tasks (List[str]): The tasks to evaluate

    Returns:
        List[EvaluationTask]: The tasks to evaluate
    """
    task_lists: List[EvaluationTask] = []
    for evaluation_task_str in evaluation_tasks_str:
        for task in EvaluationTask:
            if evaluation_task_str == task.value:
                task_lists.append(task)
    return task_lists


def set_task_dict(
    evaluation_tasks_dict: Dict[str, Dict[str, Any]]
) -> Dict[EvaluationTask, Dict[str, Any]]:
    """[summary]
    Convert str key to EvaluationTask class

    Args:
        evaluation_tasks_str (Dict[str, Dict[str, Any]]): The tasks to evaluate

    Returns:
        Dict[EvaluationTask, Dict[str, Any]]: The tasks to evaluate
    """
    task_dict: Dict[EvaluationTask, Dict[str, Any]] = {}
    for key, item in evaluation_tasks_dict.items():
        for task in EvaluationTask:
            if key == task.value:
                task_dict[task] = item
    return task_dict


def set_task(task_name: str) -> EvaluationTask:
    """[summary]
    Convert str task name to EvaluationTask class

    Args:
        task_name (str): The task to evaluate

    Returns:
        EvaluationTask: The
    """
    for task in EvaluationTask:
        if task_name == task.value:
            return task
