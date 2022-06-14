from enum import Enum
from typing import Any
from typing import Dict
from typing import List


class EvaluationTask(Enum):
    """[summary]
    Evaluation tasks enum class
    """

    DETECTION = "detection"
    TRACKING = "tracking"
    PREDICTION = "prediction"
    SENSING = "sensing"


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
