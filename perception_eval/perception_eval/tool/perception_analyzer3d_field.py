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
        # print("grid size", self.nx, self.ny)

        # set statistics parameters
        self.config_statistic_min_numb = 4 # minimum number of pairs to calculate statistics

        # define layers
        self.x = np.zeros((self.nx, self.ny))
        self.y = np.zeros((self.nx, self.ny))
        self.yaw = np.zeros((self.nx, self.ny))
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))

        self.num = np.zeros((self.nx, self.ny))
        self.ratio_valid = np.ones((self.nx, self.ny))
        self.num_tp = np.zeros((self.nx, self.ny))
        self.num_tn = np.zeros((self.nx, self.ny))
        self.num_fp = np.zeros((self.nx, self.ny))
        self.num_fn = np.zeros((self.nx, self.ny))
        self.ratio_tp = np.zeros((self.nx, self.ny))
        self.ratio_tn = np.zeros((self.nx, self.ny))
        self.ratio_fp = np.zeros((self.nx, self.ny))
        self.ratio_fn = np.zeros((self.nx, self.ny))

        self.num_pair = np.zeros((self.nx, self.ny))
        self.pair_valid = np.ones((self.nx, self.ny))
        self.error_x_mean = np.zeros((self.nx, self.ny))
        self.error_x_std = np.zeros((self.nx, self.ny))
        self.error_y_mean = np.zeros((self.nx, self.ny))
        self.error_y_std = np.zeros((self.nx, self.ny))
        self.error_delta_mean = np.zeros((self.nx, self.ny))
        self.error_delta_std = np.zeros((self.nx, self.ny))
        self.error_yaw_mean = np.zeros((self.nx, self.ny))
        self.error_yaw_std = np.zeros((self.nx, self.ny))

    def processMeans(self)->None:
        self.ratio_valid = self.num > 0
        self.x[self.ratio_valid] = np.divide(self.x[self.ratio_valid], self.num[self.ratio_valid])
        self.y[self.ratio_valid] = np.divide(self.y[self.ratio_valid], self.num[self.ratio_valid])
        self.yaw[self.ratio_valid] = np.divide(self.yaw[self.ratio_valid], self.num[self.ratio_valid])
        self.vx[self.ratio_valid] = np.divide(self.vx[self.ratio_valid], self.num[self.ratio_valid])
        self.vy[self.ratio_valid] = np.divide(self.vy[self.ratio_valid], self.num[self.ratio_valid])
        self.x[~self.ratio_valid] = np.nan
        self.y[~self.ratio_valid] = np.nan
        self.yaw[~self.ratio_valid] = np.nan
        self.vx[~self.ratio_valid] = np.nan
        self.vy[~self.ratio_valid] = np.nan

    def processRatios(self)->None:
        self.ratio_valid = self.num > 0
        self.ratio_tp[self.ratio_valid] = np.divide(self.num_tp[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_tn[self.ratio_valid] = np.divide(self.num_tn[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_fp[self.ratio_valid] = np.divide(self.num_fp[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_fn[self.ratio_valid] = np.divide(self.num_fn[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_tp[~self.ratio_valid] = np.nan
        self.ratio_tn[~self.ratio_valid] = np.nan
        self.ratio_fp[~self.ratio_valid] = np.nan
        self.ratio_fn[~self.ratio_valid] = np.nan

    def processError(self)->None:
        self.pair_valid = self.num_pair > self.config_statistic_min_numb
        self.error_x_mean[self.pair_valid] = np.divide(self.error_x_mean[self.pair_valid], self.num_pair[self.pair_valid])
        self.error_x_std[self.pair_valid] = np.sqrt(np.divide(self.error_x_std[self.pair_valid], self.num_pair[self.pair_valid]))
        self.error_y_mean[self.pair_valid] = np.divide(self.error_y_mean[self.pair_valid], self.num_pair[self.pair_valid])
        self.error_y_std[self.pair_valid] = np.sqrt(np.divide(self.error_y_std[self.pair_valid], self.num_pair[self.pair_valid]))
        self.error_yaw_mean[self.pair_valid] = np.divide(self.error_yaw_mean[self.pair_valid], self.num_pair[self.pair_valid])
        self.error_yaw_std[self.pair_valid] = np.sqrt(np.divide(self.error_yaw_std[self.pair_valid], self.num_pair[self.pair_valid]))
        self.error_delta_mean[self.pair_valid] = np.divide(self.error_delta_mean[self.pair_valid], self.num_pair[self.pair_valid])
        self.error_delta_std[self.pair_valid] = np.sqrt(np.divide(self.error_delta_std[self.pair_valid], self.num_pair[self.pair_valid]))
        self.error_x_mean[~self.pair_valid] = np.nan
        self.error_x_std[~self.pair_valid] = np.nan
        self.error_y_mean[~self.pair_valid] = np.nan
        self.error_y_std[~self.pair_valid] = np.nan
        self.error_yaw_mean[~self.pair_valid] = np.nan
        self.error_yaw_std[~self.pair_valid] = np.nan
        self.error_delta_mean[~self.pair_valid] = np.nan
        self.error_delta_std[~self.pair_valid] = np.nan


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
        # detect_field : PerceptionFieldXY = PerceptionFieldXY(self.config)
        error_field : PerceptionFieldXY = PerceptionFieldXY(self.config)
        uncertainty_field : PerceptionFieldXY = PerceptionFieldXY(self.config)

        # set condition
        grid_idx_x : str = "x"
        grid_idx_y : str = "y"

        # loop for each frame
        for _, item in df.groupby(level=0):
            pair = item.droplevel(level=0)
            is_gt_valid : bool = False
            is_est_valid : bool = False
            is_paired : bool = False
            
            if "ground_truth" in pair.index:
                gt = pair.loc["ground_truth"]
                is_gt_valid = ~np.isnan(gt[grid_idx_x]) and ~np.isnan(gt[grid_idx_y])
                is_paired = is_gt_valid & (gt["status"] == MatchingStatus.TP.value)
            if "estimation" in pair.index:
                est = pair.loc["estimation"]
                is_est_valid = ~np.isnan(est[grid_idx_x]) and ~np.isnan(est[grid_idx_y])
                is_paired = is_paired & is_est_valid

            # if est["status"]!=MatchingStatus.FN:
            #     print("status", gt["status"], est["status"])

            if is_gt_valid:
                idx_gt_x, idx_gt_y = self.getGridIndex(gt[grid_idx_x], gt[grid_idx_y])
                error_field.num[idx_gt_x, idx_gt_y] += 1

                error_field.x[idx_gt_x, idx_gt_y] += gt["x"]
                error_field.y[idx_gt_x, idx_gt_y] += gt["y"]
                error_field.yaw[idx_gt_x, idx_gt_y] += gt["yaw"]
                error_field.vx[idx_gt_x, idx_gt_y] += gt["vx"]
                error_field.vy[idx_gt_x, idx_gt_y] += gt["vy"]
                
                if gt["status"] == MatchingStatus.TP.value:
                    error_field.num_tp[idx_gt_x, idx_gt_y] += 1
                elif gt["status"] == MatchingStatus.FP.value:
                    error_field.num_fp[idx_gt_x, idx_gt_y] += 1
                elif gt["status"] == MatchingStatus.FN.value:
                    error_field.num_fn[idx_gt_x, idx_gt_y] += 1
                elif gt["status"] == MatchingStatus.TN.value:
                    error_field.num_tn[idx_gt_x, idx_gt_y] += 1
                else:
                    pass

            if is_est_valid:
                idx_est_x, idx_est_y = self.getGridIndex(est[grid_idx_x], est[grid_idx_y])
                uncertainty_field.num[idx_gt_x, idx_gt_y] += 1

                uncertainty_field.x[idx_est_x, idx_est_y] += est["x"]
                uncertainty_field.y[idx_est_x, idx_est_y] += est["y"]
                uncertainty_field.yaw[idx_est_x, idx_est_y] += est["yaw"]
                uncertainty_field.vx[idx_est_x, idx_est_y] += est["vx"]
                uncertainty_field.vy[idx_est_x, idx_est_y] += est["vy"]

                if est["status"] == MatchingStatus.TP.value:
                    uncertainty_field.num_tp[idx_est_x, idx_est_y] += 1
                elif est["status"] == MatchingStatus.FP.value:
                    uncertainty_field.num_fp[idx_est_x, idx_est_y] += 1
                elif est["status"] == MatchingStatus.FN.value:
                    uncertainty_field.num_fn[idx_est_x, idx_est_y] += 1
                elif est["status"] == MatchingStatus.TN.value:
                    uncertainty_field.num_tn[idx_est_x, idx_est_y] += 1
                else:
                    pass

            if is_paired:
                # calculate error when paired
                error_x:float = est["x"] - gt["x"]
                error_y:float = est["y"] - gt["y"]
                error_delta:float = np.sqrt(error_x**2 + error_y**2)
                error_yaw:float = est["yaw"] - gt["yaw"]
                if error_yaw > np.pi:
                    error_yaw -= 2*np.pi
                elif error_yaw < -np.pi:
                    error_yaw += 2*np.pi

                idx_gt_x, idx_gt_y = self.getGridIndex(gt[grid_idx_x], gt[grid_idx_y])
                error_field.num_pair[idx_gt_x, idx_gt_y] += 1
                error_field.error_x_mean[idx_gt_x, idx_gt_y] += error_x
                error_field.error_x_std[idx_gt_x, idx_gt_y] += error_x ** 2
                error_field.error_y_mean[idx_gt_x, idx_gt_y] += error_y
                error_field.error_y_std[idx_gt_x, idx_gt_y] += error_y ** 2
                error_field.error_yaw_mean[idx_gt_x, idx_gt_y] += error_yaw
                error_field.error_yaw_std[idx_gt_x, idx_gt_y] += error_yaw ** 2
                error_field.error_delta_mean[idx_gt_x, idx_gt_y] += error_delta
                error_field.error_delta_std[idx_gt_x, idx_gt_y] += error_delta ** 2
                
                idx_est_x, idx_est_y = self.getGridIndex(est[grid_idx_x], est[grid_idx_y])
                uncertainty_field.num_pair[idx_est_x, idx_est_y] += 1
                uncertainty_field.error_x_mean[idx_est_x, idx_est_y] += -error_x
                uncertainty_field.error_x_std[idx_est_x, idx_est_y] += error_x ** 2
                uncertainty_field.error_y_mean[idx_est_x, idx_est_y] += -error_y
                uncertainty_field.error_y_std[idx_est_x, idx_est_y] += error_y ** 2
                uncertainty_field.error_yaw_mean[idx_est_x, idx_est_y] += -error_yaw
                uncertainty_field.error_yaw_std[idx_est_x, idx_est_y] += error_yaw ** 2
                uncertainty_field.error_delta_mean[idx_est_x, idx_est_y] += error_delta
                uncertainty_field.error_delta_std[idx_est_x, idx_est_y] += error_delta ** 2

        # process statistics
        # mean
        error_field.processMeans()
        uncertainty_field.processMeans()
        # ratio
        error_field.processRatios()
        uncertainty_field.processRatios()

        # error score
        error_field.processError()
        uncertainty_field.processError()
            
        return error_field, uncertainty_field


    # get index
    def getGridIndex(self, pos_x: float, pos_y: float) -> Tuple[int, int]:
        idx_x = int((pos_x - self.config.grid.min_x) / self.config.grid.interval_x)
        idx_y = int((pos_y - self.config.grid.min_y) / self.config.grid.interval_y)
        return idx_x, idx_y
   

# Visualizer



