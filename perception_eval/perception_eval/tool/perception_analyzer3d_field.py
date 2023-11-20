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


class PerceptionFieldAxis:
    def __init__(self, type: str, data_label: str, name: str="") -> None:

        self.type : str = type
        self.data_label : str = data_label
        self.name : str = name
        self.plot_scale : float = 1.0
        self.unit : str = "none"
        if name == "":
            self.name = data_label

        # set default parameters by type
        if self.type == "length":
            self.grid_axis : np.ndarray = np.arange(-90.0, 90.0, 10)
            self.plot_range : Tuple[float, float] = (-110.0, 110.0)
            self.plot_aspect_ratio : float = 10.0
            self.unit : str = "m"
        elif self.type == "angle":
            self.grid_axis : np.ndarray = np.arange(-180+22.5, 180.0, 45) * np.pi / 180.0
            self.plot_range : Tuple[float, float] = (-180.0-22.5, 180.0-22.5)
            self.plot_aspect_ratio : float = 45.0
            self.plot_scale = 180.0 / np.pi # convert radian to degree
            self.unit : str = "deg"
        elif self.type == "velocity":
            self.grid_axis : np.ndarray = np.arange(-35.0, 35.0, 5)
            self.plot_range : Tuple[float, float] = (-40.0, 40.0)
            self.plot_aspect_ratio : float = 5.0
            self.unit : str = "m/s"
        elif self.type == "acceleration":
            self.grid_axis : np.ndarray = np.arange(-10.0, 10.0, 2)
            self.plot_range : Tuple[float, float] = (-11.0, 11.0)
            self.plot_aspect_ratio : float = 2.0
            self.unit : str = "m/s^2"
        elif self.type == "time":
            self.grid_axis : np.ndarray = np.arange(0.0, 5.0, 1)
            self.plot_range : Tuple[float, float] = (0.0, 6.0)
            self.plot_aspect_ratio : float = 1.0
            self.unit : str = "s"
        else:
            self.grid_axis : np.ndarray = [0]
            self.plot_range : Tuple[float, float] = (0.0, 1.0)
            self.plot_aspect_ratio : float = 1.0
            self.unit : str = "none"

    def setGridAxis(self, grid_axis : np.ndarray) -> None:
        if (grid_axis.shape[0] < 2) & (self.type != "none"):
            raise ValueError("grid_axis must have more than 2 elements.")
        self.grid_axis = np.sort(grid_axis)
        if self.type == "angle":
            self.grid_axis = self.grid_axis * np.pi / 180.0

    def isLoop(self) -> bool:
        if self.type == "angle":
            return True
        else:
            return False
        
    def isNone(self) -> bool:
        if self.type == "none":
            return True
        else:
            return False
    
    def getTitle(self) -> str:
        return self.name + " [" + self.unit + "]"


# Perception Field
class PerceptionFieldXY:
    def __init__(self, evaluation_config: PerceptionEvaluationConfig, 
                 axis_x : PerceptionFieldAxis, 
                 axis_y : PerceptionFieldAxis
                 ) -> None:
        
        self.__config : PerceptionEvaluationConfig = evaluation_config
        # Set statistics parameters
        self.config_statistic_min_numb : int = 4 # minimum number of pairs to calculate statistics

        self.axis_x : PerceptionFieldAxis = axis_x
        self.axis_y : PerceptionFieldAxis = axis_y

        # Set grid
        self.generateGrid(axis_x, axis_y)

        # Define layers
        self.x : np.ndarray = np.zeros((self.nx, self.ny))
        self.y : np.ndarray = np.zeros((self.nx, self.ny))
        self.yaw : np.ndarray = np.zeros((self.nx, self.ny))
        self.vx : np.ndarray = np.zeros((self.nx, self.ny))
        self.vy : np.ndarray = np.zeros((self.nx, self.ny))
        self.dist : np.ndarray = np.zeros((self.nx, self.ny))

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

    def _getCellPos(self, field_axis:PerceptionFieldAxis) -> np.ndarray:

        if field_axis.isNone():
            return np.array([0.0])

        grid_axis : np.ndarray = field_axis.grid_axis
        cell_pos_array : np.ndarray = np.copy(grid_axis)

        # cell center positions are defined as average of grid points
        cell_pos_array[1:] = (grid_axis[0:-1] + grid_axis[1:]) / 2.0
        x0 : float = 0.0
        xe : float = 0.0

        # Set cell positions of boundary
        if grid_axis[0] == 2:
            x0 = 2 * grid_axis[0] - grid_axis[1]
            xe = 2 * grid_axis[1] - grid_axis[0]
        else:
            x0 : float = grid_axis[0] - 0.5 * (grid_axis[1] - grid_axis[0])
            xe : float = grid_axis[-1] + 0.5 * (grid_axis[-1] - grid_axis[-2])
        cell_pos_array[0] = x0

        # Set additioanl cell for outside of grid points, on when the axis is not loop
        if field_axis.isLoop() != True:
            cell_pos_array = np.append(cell_pos_array, xe)

        return cell_pos_array


    def generateGrid(self, axis_x : PerceptionFieldAxis, axis_y : PerceptionFieldAxis) -> None:
        # Generate mesh cells
        #   Grid axis 1 dimenstion and is defined as follows:
        #      0     1     2          grid points, positions 
        #  ----|-----|-----|---->
        #   0     1     2     3       cell array index, 0 and 3 are outside of grid points
        #   |     |     |     |       cell center positions are defined as average of grid points                           

        # Set grid axis
        self.grid_axis_x = axis_x.grid_axis
        self.grid_axis_y = axis_y.grid_axis

        # Generate cell positions
        #   The mesh is created by two 1D arrays of grid points
        if self.grid_axis_x.shape[0] < 2:
            raise ValueError("grid_axis_x must have more than 2 elements.")
        if self.grid_axis_y.shape[0] < 2:
            raise ValueError("grid_axis_y must have more than 2 elements.")

        self.cell_pos_x : np.ndarray = self._getCellPos(axis_x)
        self.cell_pos_y : np.ndarray = self._getCellPos(axis_y)

        # Generate mesh
        self.mesh_center_x, self.mesh_center_y = np.meshgrid(self.cell_pos_x, self.cell_pos_y, indexing='ij')

        # Set layer array size
        #   arrays represent surface, including outside of grid points
        self.nx : int = self.mesh_center_x.shape[0]
        self.ny : int = self.mesh_center_x.shape[1]

    def _processMeans(self)->None:
        self.ratio_valid = self.num > 0
        self.x[self.ratio_valid] = np.divide(self.x[self.ratio_valid], self.num[self.ratio_valid])
        self.y[self.ratio_valid] = np.divide(self.y[self.ratio_valid], self.num[self.ratio_valid])
        self.yaw[self.ratio_valid] = np.divide(self.yaw[self.ratio_valid], self.num[self.ratio_valid])
        self.vx[self.ratio_valid] = np.divide(self.vx[self.ratio_valid], self.num[self.ratio_valid])
        self.vy[self.ratio_valid] = np.divide(self.vy[self.ratio_valid], self.num[self.ratio_valid])
        self.dist[self.ratio_valid] = np.divide(self.dist[self.ratio_valid], self.num[self.ratio_valid])
        self.x[~self.ratio_valid] = self.mesh_center_x[~self.ratio_valid]
        self.y[~self.ratio_valid] = self.mesh_center_y[~self.ratio_valid]
        self.yaw[~self.ratio_valid] = np.nan
        self.vx[~self.ratio_valid] = np.nan
        self.vy[~self.ratio_valid] = np.nan
        self.dist[~self.ratio_valid] = np.nan

    def _processRatios(self)->None:
        self.ratio_valid = self.num > 0
        self.ratio_tp[self.ratio_valid] = np.divide(self.num_tp[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_tn[self.ratio_valid] = np.divide(self.num_tn[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_fp[self.ratio_valid] = np.divide(self.num_fp[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_fn[self.ratio_valid] = np.divide(self.num_fn[self.ratio_valid], self.num[self.ratio_valid])
        self.ratio_tp[~self.ratio_valid] = np.nan
        self.ratio_tn[~self.ratio_valid] = np.nan
        self.ratio_fp[~self.ratio_valid] = np.nan
        self.ratio_fn[~self.ratio_valid] = np.nan

    def _processError(self)->None:
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

    def doPostprocess(self)->None:
        self._processMeans()
        self._processRatios()
        self._processError()


    def _getAxisIndex(self, axis:PerceptionFieldAxis, value:float) -> int:
        # Get index of grid axis

        # Process differently by its axis type

        if axis.isNone():
            return 0
        
        if axis.type == "angle":
            if value > np.pi:
                value = value - 2*np.pi
            elif value < -np.pi:
                value = value + 2*np.pi

        idx_cell : int = 0
        for idx, x in enumerate(axis.grid_axis):
            if value >= x:
                idx_cell = idx + 1
        
        if value < axis.grid_axis[0]:
            idx_cell = 0

        
        if idx_cell == axis.grid_axis.shape[0]:
            if axis.isLoop():
                idx_cell = 0
            else:
                idx_cell = axis.grid_axis.shape[0] - 1

        return idx_cell

    # Get index
    def getGridIndex(self, pos_x: float, pos_y: float) -> Tuple[int, int]:

        idx_x :int = self._getAxisIndex(self.axis_x, pos_x)
        idx_y :int = self._getAxisIndex(self.axis_y, pos_y)
        
        return idx_x, idx_y
   
# Analyzer
class PerceptionAnalyzer3DField(PerceptionAnalyzer3D):
    """An analyzer class for 3D perception field evaluation results."""

    def addAdditionalColumn(self) -> None:
        """Add additional column to DataFrame."""
        # Add dist column
        self.df["dist"] = np.sqrt(self.df["x"]**2 + self.df["y"]**2)
        # Azimuth angle
        self.df["azimuth"] = np.arctan2(self.df["y"], self.df["x"])
        # Visual heading angle
        self.df["visual_heading"] = self.df["yaw"] - self.df["azimuth"]
        self.df.loc[self.df["visual_heading"] > np.pi, "visual_heading"] -= 2*np.pi
        self.df.loc[self.df["visual_heading"] < -np.pi, "visual_heading"] += 2*np.pi

    def addErrorColumns(self) -> None:
        # Add error columns
        self.df["error_x"] = np.nan
        self.df["error_y"] = np.nan
        self.df["error_yaw"] = np.nan
        self.df["error_delta"] = np.nan

        # Get ground truth and estimation data
        gt_mask = self.df.index.get_level_values(1) == "ground_truth"
        est_mask = self.df.index.get_level_values(1) == "estimation"
        gt = self.df[gt_mask].droplevel(level=1)
        est = self.df[est_mask].droplevel(level=1)

        # Calculate errors
        valid_mask : pd.DataFrame = ~np.isnan(gt["x"]) & ~np.isnan(gt["y"]) & ~np.isnan(gt["yaw"]) & \
                                   ~np.isnan(est["x"]) & ~np.isnan(est["y"]) & ~np.isnan(est["yaw"])
        error_x : np.ndarray = est["x"].values[valid_mask] - gt["x"].values[valid_mask]
        error_y : np.ndarray = est["y"].values[valid_mask] - gt["y"].values[valid_mask]
        error_delta : np.ndarray = np.sqrt(error_x**2 + error_y**2)
        error_yaw : np.ndarray = est["yaw"].values[valid_mask] - gt["yaw"].values[valid_mask]
        error_yaw[error_yaw > np.pi] -= 2*np.pi
        error_yaw[error_yaw < -np.pi] += 2*np.pi

        valid_mask = np.repeat(valid_mask.values,2,axis=0)

        # Update error columns
        self.df.loc[gt_mask&valid_mask, "error_x"] = error_x
        self.df.loc[gt_mask&valid_mask, "error_y"] = error_y
        self.df.loc[gt_mask&valid_mask, "error_delta"] = error_delta
        self.df.loc[gt_mask&valid_mask, "error_yaw"] = error_yaw
        self.df.loc[est_mask&valid_mask, "error_x"] = -error_x
        self.df.loc[est_mask&valid_mask, "error_y"] = -error_y
        self.df.loc[est_mask&valid_mask, "error_delta"] = error_delta
        self.df.loc[est_mask&valid_mask, "error_yaw"] = -error_yaw

    def analyzeXY(self, 
                  axis_x:PerceptionFieldAxis, axis_y:PerceptionFieldAxis,
                  **kwargs) -> Tuple[Optional[PerceptionFieldXY], Optional[PerceptionFieldXY]]:
        """Analyze 3D perception field evaluation results.

        Args:
            axis_x (PerceptionFieldAxis): Axis of x.
            axis_y (PerceptionFieldAxis): Axis of y.
            **kwargs: Specify scene, frame, area or uuid.
        
        Returns:
            error_field (Optional[PerceptionFieldXY]): Field of errors.
            uncertainty_field (Optional[PerceptionFieldXY]): Field of uncertainties.

        """

        # Extrack data
        df: pd.DataFrame = self.get(**kwargs)

        # Set axis data labels
        label_axis_x : str = axis_x.data_label
        label_axis_y : str = axis_y.data_label

        # Initialize fields
        error_field : PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)
        uncertainty_field : PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)

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
                error_field.dist[idx_gt_x, idx_gt_y] += np.sqrt(gt["x"]**2 + gt["y"]**2)
                
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
                # get errors
                error_x:float = gt["error_x"]
                error_y:float = gt["error_y"]
                error_delta:float = gt["error_delta"]
                error_yaw:float = gt["error_yaw"]

                # fill the bins
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
        error_field.doPostprocess()
        uncertainty_field.doPostprocess()
            
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



