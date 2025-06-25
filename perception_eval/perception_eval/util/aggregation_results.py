from typing import Dict
from typing import List

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
    for mode, label_map in frame_results.items():
        for label, threshold_map in label_map.items():
            for threshold, detection_list in threshold_map.items():
                accumulated_results[mode][label][threshold].extend(detection_list)
