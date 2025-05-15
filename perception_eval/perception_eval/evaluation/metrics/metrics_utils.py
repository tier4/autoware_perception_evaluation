from collections import defaultdict
from itertools import chain
from typing import Dict
from typing import List
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching.matching_config import MatchingConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


def flatten_and_group_object_results_by_match_config(
    object_results: Union[
        Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]],
        Dict[LabelType, Dict[MatchingConfig, List[List[DynamicObjectWithPerceptionResult]]]],
    ],
) -> Dict[MatchingConfig, Dict[LabelType, List[DynamicObjectWithPerceptionResult]]]:
    """
    Flatten nested detection results and regroup them by matching configuration and label.

    This method supports both single-frame and multi-frame object results.
    In the case of multi-frame results (i.e., lists of lists), the method flattens the
    nested lists into a single list per (label, MatchingConfig) pair.
    It then reorganizes the data into a structure grouped first by MatchingConfig
    and then by LabelType, which simplifies downstream metric computations.

    Args:
        object_results: A dictionary where the first key is a LabelType, the second key is a MatchingConfig,
            and the value is either a list of DynamicObjectWithPerceptionResult (for single-frame input) or
            a list of lists of them (for multi-frame input).

    Returns:
        A dictionary where each key is a MatchingConfig, and each value is another dictionary
        mapping LabelType to a flattened list of DynamicObjectWithPerceptionResult.
    """
    results_by_match_config: Dict[
        MatchingConfig, Dict[LabelType, List[DynamicObjectWithPerceptionResult]]
    ] = defaultdict(dict)

    for label, method_results in object_results.items():
        for matching_config, result_list in method_results.items():
            if all(isinstance(r, list) for r in result_list):
                result_list_flat = list(chain.from_iterable(result_list))
            else:
                result_list_flat = result_list

            results_by_match_config[matching_config][label] = result_list_flat

    return results_by_match_config
