from typing import Dict
from typing import List
from typing import Optional

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


def accumulate_nuscene_results(
    accumulated_results: Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]],
    frame_results: Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]],
) -> None:
    """
    Accumulate detection results from a single frame into the accumulated results dictionary.

    This function updates `accumulated_results` in-place by extending its inner lists with
    detection results from `frame_results`, preserving the nested structure of:
    MatchingMode → LabelType → threshold → List[DynamicObjectWithPerceptionResult].

    Args:
        accumulated_results (dict): Dictionary accumulating detection results across frames.
        frame_results (dict): Dictionary containing detection results for a single frame.
    """
    for matching_mode, label_result in frame_results.items():
        for label, threshold_result in label_result.items():
            for threshold, object_results in threshold_result.items():
                accumulated_results[matching_mode][label][threshold].extend(object_results)


def accumulate_nuscene_tracking_results(
    accumulated_results: Dict[
        MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]
    ],
    frame_results: Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]],
) -> None:
    """
    Accumulate tracking results from a single frame into the accumulated results dictionary.

    This function updates `accumulated_results` in-place by extending its inner lists with
    detection results from `frame_results`, preserving the nested structure of:
    MatchingMode → LabelType → threshold → List[List[DynamicObjectWithPerceptionResult]].

        The difference between this function and `accumulate_nuscene_results` is that this function appends a list of object results to the accumulated results,
        while another function extends the inner list. But this function is used for tracking results, so it appends a list of object results to the accumulated results.
    Args:
        accumulated_results (dict): Dictionary accumulating detection results across frames.
        frame_results (dict): Dictionary containing detection results for a single frame.
    """
    for matching_mode, label_result in frame_results.items():
        for label, threshold_result in label_result.items():
            for threshold, object_results in threshold_result.items():
                accumulated_results[matching_mode][label][threshold].append(object_results)


def initialize_nuscene_object_results(
    accumulated_results: Dict[
        MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]
    ],
    frame_results: Optional[Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]],
) -> None:
    """
    Initialize the accumulated_results dictionary from frame results.

    Args:
        accumulated_results (dict): Dictionary accumulating detection results across frames.
        frame_results (dict): Dictionary containing detection results for a single frame. If None, initialize the dictionary with empty lists.
    """
    for matching_mode, label_result in frame_results.items():
        for label, threshold_result in label_result.items():
            for threshold, _ in threshold_result.items():
                accumulated_results[matching_mode][label][threshold].append([])
