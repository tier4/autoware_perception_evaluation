from enum import Enum


class EvaluationTask(Enum):
    """[summary]
    Evaluation tasks enum class
    """

    DETECTION = "detection"
    TRACKING = "tracking"
    PREDICTION = "prediction"
    SENSING = "sensing"
