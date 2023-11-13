# Copyright 2023 TIER IV, Inc.

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
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
import logging
import numpy as np
import pandas as pd

from perception_eval.common.status import MatchingStatus
from perception_eval.config import PerceptionEvaluationConfig
from .perception_analyzer3d import PerceptionAnalyzer3D
from .utils import extract_area_results
from .utils import get_metrics_info



# Perception Field
class GridConfig:
    def __init__(self) -> None:
        self.max_x = 100.0
        self.min_x = -100.0
        self.interval_x = 10.0
        self.offset_x = 5.0
        self.num_division_x = int((self.max_x - self.min_x) / self.interval_x)
        self.max_y = 100.0
        self.min_y = -100.0
        self.interval_y = 10.0
        self.offset_y = 5.0
        self.num_division_y = int((self.max_y - self.min_y) / self.interval_y)

class PerceptionFieldXY:
    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        self.__config = evaluation_config

        # set grid
        self.__config.grid = GridConfig()
        
        self.grid_x = np.linspace(
            self.__config.grid.min_x + self.__config.grid.offset_x,
            self.__config.grid.max_x + self.__config.grid.offset_x,
            self.__config.grid.num_division_x + 1,
        )
        self.grid_y = np.linspace(
            self.__config.grid.min_y + self.__config.grid.offset_y,
            self.__config.grid.max_y + self.__config.grid.offset_y,
            self.__config.grid.num_division_y + 1,
        )

        # arrays represent surface, including outside of grid points
        # therefore, the size of array is (number of grid points + 1)
        self.nx = self.grid_x.size + 1 
        self.ny = self.grid_y.size + 1 
        print("grid size", self.nx, self.ny)

        # define layers
        self.x = np.zeros((self.nx, self.ny))
        self.y = np.zeros((self.nx, self.ny))
        self.yaw = np.zeros((self.nx, self.ny))
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))

        self.num_total = np.zeros((self.nx, self.ny))
        self.num_tp = np.zeros((self.nx, self.ny))
        self.num_tn = np.zeros((self.nx, self.ny))
        self.num_fp = np.zeros((self.nx, self.ny))
        self.num_fn = np.zeros((self.nx, self.ny))

        self.ratio_tp = np.zeros((self.nx, self.ny))
        self.ratio_tn = np.zeros((self.nx, self.ny))
        self.ratio_fp = np.zeros((self.nx, self.ny))
        self.ratio_fn = np.zeros((self.nx, self.ny))

        self.error_x_mean = np.zeros((self.nx, self.ny))
        self.error_x_std = np.zeros((self.nx, self.ny))
        self.error_y_mean = np.zeros((self.nx, self.ny))
        self.error_y_std = np.zeros((self.nx, self.ny))
        self.error_yaw_mean = np.zeros((self.nx, self.ny))
        self.error_yaw_std = np.zeros((self.nx, self.ny))


# Analyzer
class PerceptionAnalyzer3DField(PerceptionAnalyzer3D):
    """An analyzer class for 3D perception field evaluation results.

    Attributes:

    Args:

    """

    def analyze(self, **kwargs) -> Tuple[Optional[PerceptionFieldXY], Optional[PerceptionFieldXY]]:
        """Analyze TP/FP/FN ratio, metrics score, error. If there is no DataFrame to be able to analyze returns None.

        Args:
            **kwargs: Specify scene, frame, area or uuid.

        Returns:
            error_field (Optional[PerceptionFieldXY]): Field of errors.
            uncertainty_field (Optional[PerceptionFieldXY]): Field of uncertainties.
        """
        df: pd.DataFrame = self.get(**kwargs)

        # initialize
        error_field : PerceptionFieldXY = PerceptionFieldXY(self.config)
        uncertainty_field : PerceptionFieldXY = PerceptionFieldXY(self.config)

        # set condition
        

        # loop for each frame

            # fill layers


        # process statistics

            # ratio

            # error score

            
        return error_field, uncertainty_field


    # get index


    # fill layers (with class filter)


    # process statistics

    

# Visualizer



