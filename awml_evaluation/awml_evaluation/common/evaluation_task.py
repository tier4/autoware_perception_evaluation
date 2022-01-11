from enum import Enum
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
