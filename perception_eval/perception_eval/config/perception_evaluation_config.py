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

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.matching import MatchingPolicy
from perception_eval.metrics import MetricsScoreConfig

from .evaluation_config_base import EvaluationConfigBase
from .params import PerceptionFilterParam
from .params import PerceptionMetricsParam


class PerceptionEvaluationConfig(EvaluationConfigBase):
    """Configuration class for perception evaluation.

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (Union[str, Sequence[str]]): FrameID(s) in string, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: Union[str, Sequence[str]],
        result_root_directory: str,
        config_dict: Dict[str, Any],
        load_raw_data: bool = False,
    ) -> None:
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            result_root_directory=result_root_directory,
            config_dict=config_dict,
            load_raw_data=load_raw_data,
        )

        self.matching_policy = MatchingPolicy.from_dict(config_dict)

        self.metrics_config = MetricsScoreConfig(self.metrics_param)

    def _validate_task(self, task: str | EvaluationTask) -> bool:
        if isinstance(task, str):
            task = EvaluationTask.from_value(task)
        return task.is_perception()

    def _build_params(self, cfg: Dict[str, Any]) -> Tuple[PerceptionFilterParam, PerceptionMetricsParam]:
        filter_param = PerceptionFilterParam.from_dict(cfg)
        metrics_param = PerceptionMetricsParam.from_dict(cfg)
        return filter_param, metrics_param
