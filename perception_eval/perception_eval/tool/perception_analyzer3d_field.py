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

from __future__ import annotations

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
import yaml

from perception_eval.common.status import MatchingStatus
from perception_eval.config import PerceptionEvaluationConfig
from .perception_analyzer3d import PerceptionAnalyzer3D
from .utils import extract_area_results
from .utils import get_metrics_info


# minimum number of GT-Est pairs to calculate statistics
STATISTICS_MIN_NUMB: int = 4


class PerceptionFieldAxis:
    def __init__(self, type: str, data_label: str, name: str = "") -> None:
        """
        Initialize a PerceptionFieldAxis object.

        Args:
            type (str): The type of the axis.
            data_label (str): The label for the data.
            name (str, optional): The name of the axis. Defaults to "".
        """
        # Data parameters
        self.type: str = type
        self.data_label: str = data_label
        # Plot parameters
        self.name: str = name
        self.plot_scale: float = 1.0
        self.unit: str = "none"
        self.grid_axis: np.ndarray = np.ndarray([0])
        self.plot_range: Tuple[float, float] = (-1.0, 1.0)
        self.plot_aspect_ratio: float = 1.0

        if name == "":
            self.name = data_label

        # set default parameters by type
        if self.type == "length":
            self.grid_axis = np.arange(-90.0, 90.0, 10)
            self.plot_range = (-110.0, 110.0)
            self.plot_aspect_ratio = 10.0
            self.unit = "m"
        elif self.type == "angle":
            self.grid_axis = np.arange(-180 + 22.5, 180.0, 45) * np.pi / 180.0
            self.plot_range = (-180.0 - 22.5, 180.0 - 22.5)
            self.plot_aspect_ratio = 45.0
            self.plot_scale = 180.0 / np.pi  # convert radian to degree
            self.unit = "deg"
        elif self.type == "velocity":
            self.grid_axis = np.arange(-35.0, 35.0, 5)
            self.plot_range = (-40.0, 40.0)
            self.plot_aspect_ratio = 5.0
            self.unit = "m/s"
        elif self.type == "acceleration":
            self.grid_axis = np.arange(-10.0, 10.0, 2)
            self.plot_range = (-11.0, 11.0)
            self.plot_aspect_ratio = 2.0
            self.unit = "m/s^2"
        elif self.type == "time":
            self.grid_axis = np.arange(0.0, 5.0, 1)
            self.plot_range = (0.0, 6.0)
            self.plot_aspect_ratio = 1.0
            self.unit = "s"

    def getLabel(self) -> str:
        """
        Get the label for the axis.

        Returns:
            str: The label for the axis.
        """
        if self.type == "none":
            return "x"
        else:
            return self.data_label

    def setGridAxis(self, grid_axis: np.ndarray) -> None:
        """
        Set the grid axis values.

        Args:
            grid_axis (np.ndarray): The grid axis values.
        """
        if (grid_axis.shape[0] < 2) & (self.type != "none"):
            raise ValueError("grid_axis must have more than 2 elements.")
        self.grid_axis = np.sort(grid_axis)
        if self.type == "angle":
            self.grid_axis = self.grid_axis * np.pi / 180.0

    def isLoop(self) -> bool:
        """
        Check if the axis is a loop.

        Returns:
            bool: True if the axis is a loop, False otherwise.
        """
        if self.type == "angle":
            return True
        else:
            return False

    def isNone(self) -> bool:
        """
        Check if the axis is None.

        Returns:
            bool: True if the axis is None, False otherwise.
        """
        if self.type == "none":
            return True
        else:
            return False

    def getTitle(self) -> str:
        """
        Get the title of the axis.

        Returns:
            str: The title of the axis.
        """
        return self.name + " [" + self.unit + "]"


class PerceptionFieldXY:
    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
        axis_x: PerceptionFieldAxis,
        axis_y: PerceptionFieldAxis,
    ) -> None:
        """
        Initializes a PerceptionFieldXY object.

        Args:
            evaluation_config (PerceptionEvaluationConfig): The configuration for perception evaluation.
            axis_x (PerceptionFieldAxis): The x-axis configuration for the perception field.
            axis_y (PerceptionFieldAxis): The y-axis configuration for the perception field.
        """
        self.__config: PerceptionEvaluationConfig = evaluation_config
        # Set statistics parameters
        self.config_statistics_min_numb: int = STATISTICS_MIN_NUMB

        self.axis_x: PerceptionFieldAxis = axis_x
        self.axis_y: PerceptionFieldAxis = axis_y

        self.has_any_error_data: bool = False

        # Set grid
        self.generateGrid(axis_x, axis_y)

        # Define layers
        self.x: np.ndarray = np.zeros((self.nx, self.ny))
        self.y: np.ndarray = np.zeros((self.nx, self.ny))
        self.yaw: np.ndarray = np.zeros((self.nx, self.ny))
        self.vx: np.ndarray = np.zeros((self.nx, self.ny))
        self.vy: np.ndarray = np.zeros((self.nx, self.ny))
        self.dist: np.ndarray = np.zeros((self.nx, self.ny))

        self.num: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_valid: np.ndarray = np.ones((self.nx, self.ny))
        self.num_tp: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_tn: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fp: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fn: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tp: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tn: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fp: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fn: np.ndarray = np.zeros((self.nx, self.ny))

        self.num_pair: np.ndarray = np.zeros((self.nx, self.ny))
        self.pair_valid: np.ndarray = np.ones((self.nx, self.ny))
        self.error_x_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_x_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_std: np.ndarray = np.zeros((self.nx, self.ny))

        # TODO: Array of numpy array
        # Collect all data points on each cell of array
        # then calculate statistics, plot all points, etc.

    def _getCellPos(self, field_axis: PerceptionFieldAxis) -> np.ndarray:
        """
        Calculates the cell positions for a given field axis.

        Args:
            field_axis (PerceptionFieldAxis): The field axis.

        Returns:
            np.ndarray: The array of cell positions.
        """
        if field_axis.isNone():
            return np.array([0.0])

        if field_axis.grid_axis.shape[0] < 2:
            raise ValueError("field_axis.grid_axis must have more than 2 elements.")

        grid_axis: np.ndarray = field_axis.grid_axis
        cell_pos_array: np.ndarray = np.copy(grid_axis)

        # cell center positions are defined as average of grid points
        cell_pos_array[1:] = (grid_axis[0:-1] + grid_axis[1:]) / 2.0
        x0: float = 0.0
        xe: float = 0.0

        # Set cell positions of boundary
        if grid_axis[0] == 2:
            x0 = 2 * grid_axis[0] - grid_axis[1]
            xe = 2 * grid_axis[1] - grid_axis[0]
        else:
            x0 = grid_axis[0] - 0.5 * (grid_axis[1] - grid_axis[0])
            xe = grid_axis[-1] + 0.5 * (grid_axis[-1] - grid_axis[-2])
        cell_pos_array[0] = x0

        # Set additioanl cell for outside of grid points, on when the axis is not loop
        if field_axis.isLoop() != True:
            cell_pos_array = np.append(cell_pos_array, xe)

        return cell_pos_array

    def generateGrid(self, axis_x: PerceptionFieldAxis, axis_y: PerceptionFieldAxis) -> None:
        """
        Generates the grid for the perception field.

        Args:
            axis_x (PerceptionFieldAxis): The x-axis configuration for the perception field.
            axis_y (PerceptionFieldAxis): The y-axis configuration for the perception field.
        """
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
        self.cell_pos_x: np.ndarray = self._getCellPos(axis_x)
        self.cell_pos_y: np.ndarray = self._getCellPos(axis_y)

        # Generate mesh
        self.mesh_center_x, self.mesh_center_y = np.meshgrid(
            self.cell_pos_x, self.cell_pos_y, indexing="ij"
        )

        # Set layer array size
        #   arrays represent surface, including outside of grid points
        self.nx: int = self.mesh_center_x.shape[0]
        self.ny: int = self.mesh_center_x.shape[1]

    def _processMeans(self) -> None:
        """
        Processes the mean values of the perception field.
        """
        self.ratio_valid = self.num > 0
        self.x[self.ratio_valid] = np.divide(self.x[self.ratio_valid], self.num[self.ratio_valid])
        self.y[self.ratio_valid] = np.divide(self.y[self.ratio_valid], self.num[self.ratio_valid])
        self.yaw[self.ratio_valid] = np.divide(
            self.yaw[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.vx[self.ratio_valid] = np.divide(self.vx[self.ratio_valid], self.num[self.ratio_valid])
        self.vy[self.ratio_valid] = np.divide(self.vy[self.ratio_valid], self.num[self.ratio_valid])
        self.dist[self.ratio_valid] = np.divide(
            self.dist[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.x[~self.ratio_valid] = self.mesh_center_x[~self.ratio_valid]
        self.y[~self.ratio_valid] = self.mesh_center_y[~self.ratio_valid]
        self.yaw[~self.ratio_valid] = np.nan
        self.vx[~self.ratio_valid] = np.nan
        self.vy[~self.ratio_valid] = np.nan
        self.dist[~self.ratio_valid] = np.nan

    def _processRatios(self) -> None:
        """
        Processes the ratio values of the perception field.
        """
        self.ratio_valid = self.num > 0
        self.ratio_tp[self.ratio_valid] = np.divide(
            self.num_tp[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.ratio_tn[self.ratio_valid] = np.divide(
            self.num_tn[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.ratio_fp[self.ratio_valid] = np.divide(
            self.num_fp[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.ratio_fn[self.ratio_valid] = np.divide(
            self.num_fn[self.ratio_valid], self.num[self.ratio_valid]
        )
        self.ratio_tp[~self.ratio_valid] = np.nan
        self.ratio_tn[~self.ratio_valid] = np.nan
        self.ratio_fp[~self.ratio_valid] = np.nan
        self.ratio_fn[~self.ratio_valid] = np.nan

    def _processError(self) -> None:
        """
        Processes the error values of the perception field.
        """
        self.has_any_error_data = bool(np.any(self.num_pair > 0))

        self.pair_valid = self.num_pair > self.config_statistics_min_numb
        self.error_x_mean[self.pair_valid] = np.divide(
            self.error_x_mean[self.pair_valid], self.num_pair[self.pair_valid]
        )
        self.error_x_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_x_std[self.pair_valid], self.num_pair[self.pair_valid])
        )
        self.error_y_mean[self.pair_valid] = np.divide(
            self.error_y_mean[self.pair_valid], self.num_pair[self.pair_valid]
        )
        self.error_y_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_y_std[self.pair_valid], self.num_pair[self.pair_valid])
        )
        self.error_yaw_mean[self.pair_valid] = np.divide(
            self.error_yaw_mean[self.pair_valid], self.num_pair[self.pair_valid]
        )
        self.error_yaw_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_yaw_std[self.pair_valid], self.num_pair[self.pair_valid])
        )
        self.error_delta_mean[self.pair_valid] = np.divide(
            self.error_delta_mean[self.pair_valid], self.num_pair[self.pair_valid]
        )
        self.error_delta_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_delta_std[self.pair_valid], self.num_pair[self.pair_valid])
        )
        self.error_x_mean[~self.pair_valid] = np.nan
        self.error_x_std[~self.pair_valid] = np.nan
        self.error_y_mean[~self.pair_valid] = np.nan
        self.error_y_std[~self.pair_valid] = np.nan
        self.error_yaw_mean[~self.pair_valid] = np.nan
        self.error_yaw_std[~self.pair_valid] = np.nan
        self.error_delta_mean[~self.pair_valid] = np.nan
        self.error_delta_std[~self.pair_valid] = np.nan

    def doPostprocess(self) -> None:
        """
        Performs post-processing on the perception field.
        """
        self._processMeans()
        self._processRatios()
        self._processError()

    def _getAxisIndex(self, axis: PerceptionFieldAxis, value: float) -> int:
        """
        Gets the index of a value in the grid axis.

        Args:
            axis (PerceptionFieldAxis): The field axis.
            value (float): The value to find the index for.

        Returns:
            int: The index of the value in the grid axis.
        """
        # Get index of grid axis

        # Process differently by its axis type

        if axis.isNone():
            return 0

        if axis.type == "angle":
            if value > np.pi:
                value = value - 2 * np.pi
            elif value < -np.pi:
                value = value + 2 * np.pi

        idx_cell: int = 0
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

    def getGridIndex(self, pos_x: float, pos_y: float) -> Tuple[int, int]:
        """
        Gets the grid index for a given position.

        Args:
            pos_x (float): The x-coordinate of the position.
            pos_y (float): The y-coordinate of the position.

        Returns:
            Tuple[int, int]: The grid index as a tuple of (x, y).
        """
        idx_x: int = self._getAxisIndex(self.axis_x, pos_x)
        idx_y: int = self._getAxisIndex(self.axis_y, pos_y)

        return idx_x, idx_y


# Analyzer
class PerceptionAnalyzer3DField(PerceptionAnalyzer3D):
    """An analyzer class for 3D perception field evaluation results.

    This class extends the base class PerceptionAnalyzer3D and provides additional functionality
    for analyzing 3D perception field evaluation results. It includes methods for adding additional
    columns to the data, calculating error columns, and analyzing the results based on different axes.

    Args:
        evaluation_config (PerceptionEvaluationConfig): The configuration for perception evaluation.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the perception evaluation results.

    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig) -> None:
        super().__init__(evaluation_config, num_area_division=1)

    @classmethod
    def from_scenario(
        cls,
        result_root_directory: str,
        scenario_path: str,
    ) -> PerceptionAnalyzer3DField:
        """Perception results made by logsim are reproduced from pickle file.

        Args:
            result_root_directory (str): The root path to save result.
            scenario_path (str): The path of scenario file .yaml.

        Returns:
            PerceptionAnalyzer3DField: PerceptionAnalyzer3DField instance.

        Raises:
            ValueError: When unexpected evaluation task is specified in scenario file.
        """

        # Load scenario file
        with open(scenario_path, "r") as scenario_file:
            scenario_obj: Optional[Dict[str, Any]] = yaml.safe_load(scenario_file)

        p_cfg: Dict[str, Any] = scenario_obj["Evaluation"]["PerceptionEvaluationConfig"]
        eval_cfg_dict: Dict[str, Any] = p_cfg["evaluation_config_dict"]

        eval_cfg_dict["label_prefix"] = "autoware"

        evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            dataset_paths=[""],  # dummy path
            frame_id="base_link" if eval_cfg_dict["evaluation_task"] == "detection" else "map",
            result_root_directory=result_root_directory,
            evaluation_config_dict=eval_cfg_dict,
            load_raw_data=False,
        )
        return cls(evaluation_config)

    def addAdditionalColumn(self) -> None:
        """Add additional column to DataFrame."""
        # Add dist column
        self.df["dist"] = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2)
        # Azimuth angle
        self.df["azimuth"] = np.arctan2(self.df["y"], self.df["x"])
        # Visual heading angle
        self.df["visual_heading"] = self.df["yaw"] - self.df["azimuth"]
        self.df.loc[self.df["visual_heading"] > np.pi, "visual_heading"] -= 2 * np.pi
        self.df.loc[self.df["visual_heading"] < -np.pi, "visual_heading"] += 2 * np.pi

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
        valid_mask: pd.DataFrame = (
            ~np.isnan(gt["x"])
            & ~np.isnan(gt["y"])
            & ~np.isnan(gt["yaw"])
            & ~np.isnan(est["x"])
            & ~np.isnan(est["y"])
            & ~np.isnan(est["yaw"])
        )
        error_x: np.ndarray = est["x"].values[valid_mask] - gt["x"].values[valid_mask]
        error_y: np.ndarray = est["y"].values[valid_mask] - gt["y"].values[valid_mask]
        error_delta: np.ndarray = np.sqrt(error_x**2 + error_y**2)
        error_yaw: np.ndarray = est["yaw"].values[valid_mask] - gt["yaw"].values[valid_mask]
        error_yaw[error_yaw > np.pi] -= 2 * np.pi
        error_yaw[error_yaw < -np.pi] += 2 * np.pi

        valid_mask = np.repeat(valid_mask.values, 2, axis=0)

        # Update error columns
        self.df.loc[gt_mask & valid_mask, "error_x"] = error_x
        self.df.loc[gt_mask & valid_mask, "error_y"] = error_y
        self.df.loc[gt_mask & valid_mask, "error_delta"] = error_delta
        self.df.loc[gt_mask & valid_mask, "error_yaw"] = error_yaw
        self.df.loc[est_mask & valid_mask, "error_x"] = -error_x
        self.df.loc[est_mask & valid_mask, "error_y"] = -error_y
        self.df.loc[est_mask & valid_mask, "error_delta"] = error_delta
        self.df.loc[est_mask & valid_mask, "error_yaw"] = -error_yaw

    def analyzeXY(
        self, axis_x: PerceptionFieldAxis, axis_y: PerceptionFieldAxis, **kwargs
    ) -> Tuple[PerceptionFieldXY, PerceptionFieldXY]:
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
        label_axis_x: str = axis_x.getLabel()
        label_axis_y: str = axis_y.getLabel()

        # Initialize fields
        error_field: PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)
        uncertainty_field: PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)

        # loop for each frame
        for _, item in df.groupby(level=0):
            pair = item.droplevel(level=0)
            is_gt_valid: bool = False
            is_est_valid: bool = False
            is_paired: bool = False

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
                error_field.dist[idx_gt_x, idx_gt_y] += np.sqrt(gt["x"] ** 2 + gt["y"] ** 2)

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
                idx_est_x, idx_est_y = uncertainty_field.getGridIndex(
                    est[label_axis_x], est[label_axis_y]
                )
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
                error_x: float = gt["error_x"]
                error_y: float = gt["error_y"]
                error_delta: float = gt["error_delta"]
                error_yaw: float = gt["error_yaw"]

                # fill the bins
                idx_gt_x, idx_gt_y = error_field.getGridIndex(gt[label_axis_x], gt[label_axis_y])
                error_field.num_pair[idx_gt_x, idx_gt_y] += 1
                error_field.error_x_mean[idx_gt_x, idx_gt_y] += error_x
                error_field.error_x_std[idx_gt_x, idx_gt_y] += error_x**2
                error_field.error_y_mean[idx_gt_x, idx_gt_y] += error_y
                error_field.error_y_std[idx_gt_x, idx_gt_y] += error_y**2
                error_field.error_yaw_mean[idx_gt_x, idx_gt_y] += error_yaw
                error_field.error_yaw_std[idx_gt_x, idx_gt_y] += error_yaw**2
                error_field.error_delta_mean[idx_gt_x, idx_gt_y] += error_delta
                error_field.error_delta_std[idx_gt_x, idx_gt_y] += error_delta**2

                idx_est_x, idx_est_y = uncertainty_field.getGridIndex(
                    est[label_axis_x], est[label_axis_y]
                )
                uncertainty_field.num_pair[idx_est_x, idx_est_y] += 1
                uncertainty_field.error_x_mean[idx_est_x, idx_est_y] += -error_x
                uncertainty_field.error_x_std[idx_est_x, idx_est_y] += error_x**2
                uncertainty_field.error_y_mean[idx_est_x, idx_est_y] += -error_y
                uncertainty_field.error_y_std[idx_est_x, idx_est_y] += error_y**2
                uncertainty_field.error_yaw_mean[idx_est_x, idx_est_y] += -error_yaw
                uncertainty_field.error_yaw_std[idx_est_x, idx_est_y] += error_yaw**2
                uncertainty_field.error_delta_mean[idx_est_x, idx_est_y] += error_delta
                uncertainty_field.error_delta_std[idx_est_x, idx_est_y] += error_delta**2

        # process statistics
        error_field.doPostprocess()
        uncertainty_field.doPostprocess()

        return error_field, uncertainty_field
