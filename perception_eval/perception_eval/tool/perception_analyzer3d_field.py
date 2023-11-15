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
class PerceptionFieldXY:
    def __init__(self, evaluation_config: PerceptionEvaluationConfig, 
                 grid_axis_x : np.ndarray, 
                 grid_axis_y : np.ndarray
                 ) -> None:
        
        self.__config : PerceptionEvaluationConfig = evaluation_config
        # Set statistics parameters
        self.config_statistic_min_numb : int = 4 # minimum number of pairs to calculate statistics

        # Set grid mesh
        #   Grid axis 1 dimenstion and is defined as follows:
        #      0     1     2          grid points, positions 
        #  ----|-----|-----|---->
        #   0     1     2     3       cell array index, 0 and 3 are outside of grid points
        #      |  |     |  |          cell center positions are defined as average of grid points                           
        #      |           |          cell positions of 0 and 3 are the boundary of the grid 

        self.grid_axis_x : np.ndarray = np.sort(grid_axis_x)
        self.grid_axis_y : np.ndarray = np.sort(grid_axis_y)

        # Generate mesh cells
        #   The mesh is created by two 1D arrays of grid points
        self.cell_pos_x : np.ndarray = np.append(self.grid_axis_x, self.grid_axis_x[-1])
        self.cell_pos_y : np.ndarray = np.append(self.grid_axis_y, self.grid_axis_y[-1])
        self.cell_pos_x[1:-2] = (self.cell_pos_x[0:-3] +  self.cell_pos_x[1:-2]) / 2
        self.cell_pos_y[1:-2] = (self.cell_pos_y[0:-3] +  self.cell_pos_y[1:-2]) / 2

        # cell positions are defined as average of grid points
        self.mesh_center_x, self.mesh_center_y = np.meshgrid(self.cell_pos_x, self.cell_pos_y, indexing='ij')

        # Set layer array size
        #   arrays represent surface, including outside of grid points
        #   therefore, the size of array is (number of grid points + 1)
        self.nx : int = self.mesh_center_x.shape[0]
        self.ny : int = self.mesh_center_x.shape[1]

        # Define layers
        self.x : np.ndarray = np.zeros((self.nx, self.ny))
        self.y : np.ndarray = np.zeros((self.nx, self.ny))
        self.yaw : np.ndarray = np.zeros((self.nx, self.ny))
        self.vx : np.ndarray = np.zeros((self.nx, self.ny))
        self.vy : np.ndarray = np.zeros((self.nx, self.ny))

        self.num : np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_valid : np.ndarray = np.ones((self.nx, self.ny))
        self.num_tp : np.ndarray = np.zeros((self.nx, self.ny))
        self.num_tn : np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fp : np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fn : np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tp : np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tn : np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fp : np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fn : np.ndarray = np.zeros((self.nx, self.ny))

        self.num_pair : np.ndarray = np.zeros((self.nx, self.ny))
        self.pair_valid : np.ndarray = np.ones((self.nx, self.ny))
        self.error_x_mean : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_x_std : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_mean : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_std : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_mean : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_std : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_mean : np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_std : np.ndarray = np.zeros((self.nx, self.ny))



    def processMeans(self)->None:
        self.ratio_valid = self.num > 0
        self.x[self.ratio_valid] = np.divide(self.x[self.ratio_valid], self.num[self.ratio_valid])
        self.y[self.ratio_valid] = np.divide(self.y[self.ratio_valid], self.num[self.ratio_valid])
        self.yaw[self.ratio_valid] = np.divide(self.yaw[self.ratio_valid], self.num[self.ratio_valid])
        self.vx[self.ratio_valid] = np.divide(self.vx[self.ratio_valid], self.num[self.ratio_valid])
        self.vy[self.ratio_valid] = np.divide(self.vy[self.ratio_valid], self.num[self.ratio_valid])
        self.x[~self.ratio_valid] = self.mesh_center_x[~self.ratio_valid]
        self.y[~self.ratio_valid] = self.mesh_center_y[~self.ratio_valid]
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

    # Get index
    def getGridIndex(self, pos_x: float, pos_y: float) -> Tuple[int, int]:
        for idx, x in enumerate(self.grid_axis_x):
            if pos_x > x:
                idx_x = idx + 1
        if pos_x < self.grid_axis_x[0]:
            idx_x = 0
        if idx_x == self.nx:
            idx_x = self.nx - 1

        for idx, y in enumerate(self.grid_axis_y):
            if pos_y > y:
                idx_y = idx +1
        if pos_y < self.grid_axis_y[0]:
            idx_y = 0
        if idx_y == self.ny:
            idx_y = self.ny - 1
        
        return idx_x, idx_y
   
# Analyzer
class PerceptionAnalyzer3DField(PerceptionAnalyzer3D):
    """An analyzer class for 3D perception field evaluation results."""

    def analyzeXY(self, 
                  label_axis_x:str, grid_axis_x:np.ndarray,
                  label_axis_y:str, grid_axis_y:np.ndarray,
                  **kwargs) -> Tuple[Optional[PerceptionFieldXY], Optional[PerceptionFieldXY]]:
        """Analyze 3D perception field evaluation results.

        Args:
            label_axis_x (str): label of x axis.
            grid_axis_x (np.ndarray): grid axis of x axis.
            label_axis_y (str): label of y axis.
            grid_axis_y (np.ndarray): grid axis of y axis.
            **kwargs: Specify scene, frame, area or uuid.  
        
        Returns:
            error_field (Optional[PerceptionFieldXY]): Field of errors.
            uncertainty_field (Optional[PerceptionFieldXY]): Field of uncertainties.

        """

        # Extrack data
        df: pd.DataFrame = self.get(**kwargs)

        # Initialize fields
        error_field : PerceptionFieldXY = PerceptionFieldXY(self.config, grid_axis_x, grid_axis_y)
        uncertainty_field : PerceptionFieldXY = PerceptionFieldXY(self.config, grid_axis_x, grid_axis_y)

        # loop for each frame
        for _, item in df.groupby(level=0):
            pair = item.droplevel(level=0)
            is_gt_valid : bool = False
            is_est_valid : bool = False
            is_paired : bool = False
            
            if "ground_truth" in pair.index:
                gt = pair.loc["ground_truth"]
                is_gt_valid = ~np.isnan(gt[label_axis_x]) and ~np.isnan(gt[label_axis_y])
                is_paired = is_gt_valid & (gt["status"] == MatchingStatus.TP.value)
            if "estimation" in pair.index:
                est = pair.loc["estimation"]
                is_est_valid = ~np.isnan(est[label_axis_x]) and ~np.isnan(est[label_axis_y])
                is_paired = is_paired & is_est_valid

            if is_gt_valid:
                idx_gt_x, idx_gt_y = error_field.getGridIndex(gt[label_axis_x], gt[label_axis_y])
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
                idx_est_x, idx_est_y = uncertainty_field.getGridIndex(est[label_axis_x], est[label_axis_y])
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

                idx_gt_x, idx_gt_y = error_field.getGridIndex(gt[label_axis_x], gt[label_axis_y])
                error_field.num_pair[idx_gt_x, idx_gt_y] += 1
                error_field.error_x_mean[idx_gt_x, idx_gt_y] += error_x
                error_field.error_x_std[idx_gt_x, idx_gt_y] += error_x ** 2
                error_field.error_y_mean[idx_gt_x, idx_gt_y] += error_y
                error_field.error_y_std[idx_gt_x, idx_gt_y] += error_y ** 2
                error_field.error_yaw_mean[idx_gt_x, idx_gt_y] += error_yaw
                error_field.error_yaw_std[idx_gt_x, idx_gt_y] += error_yaw ** 2
                error_field.error_delta_mean[idx_gt_x, idx_gt_y] += error_delta
                error_field.error_delta_std[idx_gt_x, idx_gt_y] += error_delta ** 2
                
                idx_est_x, idx_est_y = uncertainty_field.getGridIndex(est[label_axis_x], est[label_axis_y])
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


    def analyze(self, **kwargs) -> Tuple[Optional[PerceptionFieldXY], Optional[PerceptionFieldXY]]:
        """Analyze TP/FP/FN ratio, metrics score, error. If there is no DataFrame to be able to analyze returns None.

        Args:
            **kwargs: Specify scene, frame, area or uuid.

        Returns:
            error_field (Optional[PerceptionFieldXY]): Field of errors.
            uncertainty_field (Optional[PerceptionFieldXY]): Field of uncertainties.
        """

        # Initialize fields
        grid_axis_x : np.ndarray = np.array([-90, -60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60, 90])
        grid_axis_y : np.ndarray = np.array([-90, -60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60, 90])

        # Set condition
        label_axis_x : str = "x"
        label_axis_y : str = "y"

        return self.analyzeXY(label_axis_x, label_axis_y, grid_axis_x, grid_axis_y, **kwargs)



# Visualizer



