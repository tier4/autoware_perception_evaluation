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

from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from perception_eval.common.status import MatchingStatus
from perception_eval.config import PerceptionEvaluationConfig

from .perception_analyzer3d import PerceptionAnalyzer3D

# minimum number of GT-Est pairs to calculate statistics
STATISTICS_MIN_NUMB: int = 4


class PerceptionFieldAxis:
    def __init__(self, quantity_type: str, data_label: str, name: str = "") -> None:
        """
        Initialize a PerceptionFieldAxis object.

        Args:
            quantity_type (str): The type of the axis quantity.
            data_label (str): The label for the data.
            name (str, optional): The name of the axis. Defaults to "".
        """
        # Data parameters
        self.quantity_type: str = quantity_type
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

        # set default parameters by quantity_type
        if self.quantity_type == "length":
            self.grid_axis = np.arange(-90.0, 90.0, 10)
            self.plot_range = (-110.0, 110.0)
            self.plot_aspect_ratio = 10.0
            self.unit = "m"
        elif self.quantity_type == "area":
            self.grid_axis = np.arange(-20.0, 20.0, 2)
            self.plot_range = (-25.0, 25.0)
            self.plot_aspect_ratio = 2.0
            self.unit = "m^2"
        elif self.quantity_type == "angle":
            self.grid_axis = np.arange(-180 + 22.5, 180.0, 45) * np.pi / 180.0
            self.plot_range = (-180.0 - 22.5, 180.0 - 22.5)
            self.plot_aspect_ratio = 45.0
            self.plot_scale = 180.0 / np.pi  # convert radian to degree
            self.unit = "deg"
        elif self.quantity_type == "velocity":
            self.grid_axis = np.arange(-35.0, 35.0, 5)
            self.plot_range = (-40.0, 40.0)
            self.plot_aspect_ratio = 5.0
            self.unit = "m/s"
        elif self.quantity_type == "acceleration":
            self.grid_axis = np.arange(-10.0, 10.0, 2)
            self.plot_range = (-11.0, 11.0)
            self.plot_aspect_ratio = 2.0
            self.unit = "m/s^2"
        elif self.quantity_type == "time":
            self.grid_axis = np.arange(0.0, 5.0, 1)
            self.plot_range = (0.0, 6.0)
            self.plot_aspect_ratio = 1.0
            self.unit = "s"
        else:
            self.quantity_type = "none"
            self.grid_axis = np.array([0.0, 1.0])
            self.plot_range = (-1.0, 1.0)
            self.plot_aspect_ratio = 1.0
            self.unit = "none"

    def get_label(self) -> str:
        """
        Get the label for the axis.

        Returns:
            str: The label for the axis.
        """
        if self.quantity_type == "none":
            # return an existing label
            return "x"
        else:
            return self.data_label

    def set_grid_axis(self, grid_axis: np.ndarray) -> None:
        """
        Set the grid axis values.

        Args:
            grid_axis (np.ndarray): The grid axis values.
        """
        if (grid_axis.shape[0] < 2) & (self.quantity_type != "none"):
            raise ValueError("grid_axis must have more than 2 elements.")
        self.grid_axis = np.sort(grid_axis)
        if self.quantity_type == "angle":
            self.grid_axis = self.grid_axis * np.pi / 180.0

    def is_loop(self) -> bool:
        """
        Check if the axis is a loop.

        Returns:
            bool: True if the axis is a loop, False otherwise.
        """
        return self.quantity_type == "angle"

    def is_none(self) -> bool:
        """
        Check if the axis is None.

        Returns:
            bool: True if the axis is None, False otherwise.
        """
        return self.quantity_type == "none"

    def get_title(self) -> str:
        """
        Get the title of the axis.

        Returns:
            str: The title of the axis.
        """
        return self.name + " [" + self.unit + "]"


class DataTableIdx(IntEnum):
    X = 0
    Y = auto()
    YAW = auto()
    VX = auto()
    VY = auto()
    DIST = auto()
    AZIMUTH = auto()
    HEADING = auto()
    WIDTH = auto()
    LENGTH = auto()


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
        self.generate_grid(axis_x, axis_y)

        # Define layers

        # Object statistics
        self.x: np.ndarray = np.zeros((self.nx, self.ny))
        self.y: np.ndarray = np.zeros((self.nx, self.ny))
        self.yaw: np.ndarray = np.zeros((self.nx, self.ny))
        self.vx: np.ndarray = np.zeros((self.nx, self.ny))
        self.vy: np.ndarray = np.zeros((self.nx, self.ny))
        self.dist: np.ndarray = np.zeros((self.nx, self.ny))
        self.num: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_pair: np.ndarray = np.zeros((self.nx, self.ny))

        # Detection statistics
        self.num_tp: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_tn: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fp: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_fn: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_yaw_flip: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_error_yaw_count: np.ndarray = np.zeros((self.nx, self.ny))
        self.num_error_dist_count: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tp: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_tn: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fp: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_fn: np.ndarray = np.zeros((self.nx, self.ny))
        self.ratio_yaw_flip: np.ndarray = np.zeros((self.nx, self.ny))
        self.confidence: np.ndarray = np.zeros((self.nx, self.ny))

        # Error statistics
        self.pair_valid: np.ndarray = np.ones((self.nx, self.ny))
        self.error_x_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_x_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_y_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_delta_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_yaw_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_dist_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_dist_std: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_azimuth_mean: np.ndarray = np.zeros((self.nx, self.ny))
        self.error_azimuth_std: np.ndarray = np.zeros((self.nx, self.ny))

        # Grid of data tables
        # dimension 0: x
        # dimension 1: y
        # dimension 2: ground truth and estimation
        # dimension 3: data
        table_width: int = len(DataTableIdx)
        self.data: List[List[List[np.ndarray]]] = [
            [[np.zeros((0, table_width), float) for k in [0, 1]] for j in range(self.ny)] for i in range(self.nx)
        ]

    def _expand_grid_at_boundary(self, field_axis: PerceptionFieldAxis) -> np.ndarray:
        """
        Expands the grid at the boundary.

        Args:
            field_axis (PerceptionFieldAxis): The field axis.

        Returns:
            np.ndarray: The expanded grid.
        """

        if field_axis.is_none():
            return np.array([0.0, 1.0])

        # Set expanded positions of grid axis
        x0: float = 0.0
        xe: float = 0.0
        grid_axis: np.ndarray = np.copy(field_axis.grid_axis)

        if grid_axis.shape[0] == 1:
            point: float = grid_axis[0]
            x0 = point - 1.0
            xe = point + 1.0
        else:
            x0 = 2 * grid_axis[0] - grid_axis[1]
            xe = 2 * grid_axis[-1] - grid_axis[-2]

        if field_axis.is_loop():
            return np.append(x0, grid_axis)
        else:
            return np.append(np.append(x0, grid_axis), xe)

    def _get_cell_pos(self, field_axis: PerceptionFieldAxis) -> np.ndarray:
        """
        Calculates the cell positions for a given field axis.

        Args:
            field_axis (PerceptionFieldAxis): The field axis.

        Returns:
            np.ndarray: The array of cell positions.
        """
        if field_axis.is_none():
            return np.array([0.0])

        expanded_grid_axis: np.ndarray = self._expand_grid_at_boundary(field_axis)
        cell_pos_array: np.ndarray = (expanded_grid_axis[0:-1] + expanded_grid_axis[1:]) / 2.0

        return cell_pos_array

    def generate_grid(self, axis_x: PerceptionFieldAxis, axis_y: PerceptionFieldAxis) -> None:
        """
        Generates the grid for the perception field.

        Args:
            axis_x (PerceptionFieldAxis): The x-axis configuration for the perception field.
            axis_y (PerceptionFieldAxis): The y-axis configuration for the perception field.
        """
        # Generate mesh cells
        #   Grid axis 1 dimension and is defined as follows:
        #      0     1     2          grid points, positions
        #  ----|-----|-----|---->
        #   0     1     2     3       cell array index, 0 and 3 are outside of grid points
        #   |     |     |     |       cell center positions are defined as average of grid points

        # Set grid axis
        self.grid_axis_x = axis_x.grid_axis
        self.grid_axis_y = axis_y.grid_axis

        # Generate cell positions
        self.cell_pos_x: np.ndarray = self._get_cell_pos(axis_x)
        self.cell_pos_y: np.ndarray = self._get_cell_pos(axis_y)

        # Generate mesh
        self.mesh_center_x, self.mesh_center_y = np.meshgrid(
            self.cell_pos_x, self.cell_pos_y, indexing="ij")

        # Set layer array size
        #   arrays represent surface, including outside of grid points
        self.nx: int = self.mesh_center_x.shape[0]
        self.ny: int = self.mesh_center_x.shape[1]

        # Generate mesh for plotting
        mesh_x = self._expand_grid_at_boundary(axis_x)
        mesh_y = self._expand_grid_at_boundary(axis_y)
        self.mesh_x, self.mesh_y = np.meshgrid(mesh_x, mesh_y, indexing="ij")

    def _process_means(self) -> None:
        """
        Processes the mean values of the perception field.
        """
        mask = self.num > 0
        self.x[mask] = np.divide(self.x[mask], self.num[mask])
        self.y[mask] = np.divide(self.y[mask], self.num[mask])
        self.yaw[mask] = np.divide(self.yaw[mask], self.num[mask])
        self.vx[mask] = np.divide(self.vx[mask], self.num[mask])
        self.vy[mask] = np.divide(self.vy[mask], self.num[mask])
        self.dist[mask] = np.divide(self.dist[mask], self.num[mask])
        self.confidence[mask] = np.divide(self.confidence[mask], self.num[mask])
        self.x[~mask] = self.mesh_center_x[~mask]
        self.y[~mask] = self.mesh_center_y[~mask]
        self.yaw[~mask] = np.nan
        self.vx[~mask] = np.nan
        self.vy[~mask] = np.nan
        self.dist[~mask] = np.nan
        self.confidence[~mask] = np.nan

    def _process_ratios(self) -> None:
        """
        Processes the ratio values of the perception field.
        """
        mask = self.num > 0
        self.ratio_tp[mask] = np.divide(self.num_tp[mask], self.num[mask])
        self.ratio_tn[mask] = np.divide(self.num_tn[mask], self.num[mask])
        self.ratio_fp[mask] = np.divide(self.num_fp[mask], self.num[mask])
        self.ratio_fn[mask] = np.divide(self.num_fn[mask], self.num[mask])
        self.ratio_yaw_flip[mask] = np.divide(self.num_yaw_flip[mask], self.num[mask])
        self.ratio_tp[~mask] = np.nan
        self.ratio_tn[~mask] = np.nan
        self.ratio_fp[~mask] = np.nan
        self.ratio_fn[~mask] = np.nan
        self.ratio_yaw_flip[~mask] = np.nan

    def _process_error(self) -> None:
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
            self.error_yaw_mean[self.pair_valid], self.num_error_yaw_count[self.pair_valid]
        )
        self.error_yaw_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_yaw_std[self.pair_valid],
                      self.num_error_yaw_count[self.pair_valid])
        )

        self.error_dist_mean[self.pair_valid] = np.divide(
            self.error_dist_mean[self.pair_valid], self.num_error_dist_count[self.pair_valid]
        )
        self.error_dist_std[self.pair_valid] = np.sqrt(
            np.divide(self.error_dist_std[self.pair_valid],
                      self.num_error_dist_count[self.pair_valid])
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
        self.error_dist_mean[~self.pair_valid] = np.nan
        self.error_dist_std[~self.pair_valid] = np.nan

    def do_post_process(self) -> None:
        """
        Performs post-processing on the perception field.
        """
        self._process_means()
        self._process_ratios()
        self._process_error()

    def _get_axis_index(self, axis: PerceptionFieldAxis, value: float) -> int:
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

        if axis.is_none():
            return 0

        if axis.quantity_type == "angle":
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
            if axis.is_loop():
                idx_cell = 0
            else:
                idx_cell = axis.grid_axis.shape[0] - 1

        return idx_cell

    def get_grid_index(self, pos_x: float, pos_y: float) -> Tuple[int, int]:
        """
        Gets the grid index for a given position.

        Args:
            pos_x (float): The x-coordinate of the position.
            pos_y (float): The y-coordinate of the position.

        Returns:
            Tuple[int, int]: The grid index as a tuple of (x, y).
        """
        idx_x: int = self._get_axis_index(self.axis_x, pos_x)
        idx_y: int = self._get_axis_index(self.axis_y, pos_y)

        return idx_x, idx_y


class PerceptionFieldTime:
    def __init__(
        self,
        axis_x: PerceptionFieldAxis,
    ) -> None:
        """
        Initializes a PerceptionFieldTime object.

        Args:
            axis_x (PerceptionFieldAxis): The time axis configuration for the perception field.
        """

        self.axis_x: PerceptionFieldAxis = axis_x
        self.mesh_x = axis_x.grid_axis
        self.nt = len(axis_x.grid_axis)
        self.time_intervals = list(round(t, 1) for t in axis_x.grid_axis)

        self.has_any_error_data: bool = False
        self.N: float = 0
        self.data_grouped_by_uuid: Dict[str, list[Dict[str, Any]]] = dict()
        # Basic statistics
        self.dist_thresholds = [10, 20, 40, 60, 80, 120, 1000]

        self.basic_errors = ["yaw_flip", "yaw_err",
                             "dist_err", "tp"] + [f"tp_{t}m" for t in self.dist_thresholds]
        self.basic_mean = dict()
        self.basic_mean_count = dict()
        for err in self.basic_errors:
            self.basic_mean[err] = 0
            self.basic_mean_count[err] = 1e-6

        # AutoRegressive (1) model
        # How to predict the next error noise value from the previous one:
        #  err(t+dt) = mean + phi * (err(t)- mean) + sqrt(sigma2) * N(0, 1)
        self.ar1_errors = ["yaw_err", "dist_err"]
        self.ar1_gama = dict()
        self.ar1_gama_count = dict()
        self.ar1_phi = dict()
        self.ar1_sigma2 = dict()
        for err in self.ar1_errors:
            self.ar1_gama[err] = dict((t, 0) for t in self.time_intervals)
            self.ar1_gama_count[err] = dict((t, 1e-6) for t in self.time_intervals)
            self.ar1_phi[err] = dict((t, 0) for t in self.time_intervals[1:])
            self.ar1_sigma2[err] = dict((t, 0) for t in self.time_intervals[1:])

        # Markov chain
        # How to predict the next error noise value from the previous one:
        self.mc_errors = ["yaw_flip", "tp"] + [f"tp_{t}m" for t in self.dist_thresholds]
        self.mc_p00 = dict()
        self.mc_p11 = dict()
        self.mc_rho = dict()
        self.mc_0_count = dict()
        self.mc_1_count = dict()
        for err in self.mc_errors:
            self.mc_p00[err] = dict((t, 0) for t in self.time_intervals[1:])
            self.mc_p11[err] = dict((t, 0) for t in self.time_intervals[1:])
            self.mc_rho[err] = dict((t, 0) for t in self.time_intervals[1:])
            self.mc_0_count[err] = dict((t, 1e-6) for t in self.time_intervals[1:])
            self.mc_1_count[err] = dict((t, 1e-6) for t in self.time_intervals[1:])

    def process_basic_statics(self) -> None:
        for err in self.basic_errors:
            self.basic_mean[err] /= self.basic_mean_count[err]

    def process_ar1_and_mc_parameters(self) -> None:
        # AR(1)'s parameters
        for err in self.ar1_errors:
            for dt in self.time_intervals:
                self.ar1_gama[err][dt] /= self.ar1_gama_count[err][dt]
                if dt == 0.0:
                    continue
                self.ar1_phi[err][dt] = self.ar1_gama[err][dt] / (self.ar1_gama[err][0]+1e-6)
                self.ar1_sigma2[err][dt] = self.ar1_gama[err][0] - \
                    self.ar1_phi[err][dt] * self.ar1_gama[err][dt]

        # MC's parameters
        for err in self.mc_errors:
            for dt in self.time_intervals[1:]:
                self.mc_p00[err][dt] /= self.mc_0_count[err][dt]
                self.mc_p11[err][dt] /= self.mc_1_count[err][dt]
                self.mc_rho[err][dt] = self.mc_p11[err][dt] + self.mc_p00[err][dt] - 1

    def print_model_parameters(self) -> None:
        for err in self.basic_errors:
            print(f"{err}\'s Statics:\n\
                  \tmean: {round(self.basic_mean[err],3)}\n\
                  \tcount: {int(self.basic_mean_count[err])}"
                  )
        for err in self.ar1_errors:
            print(f"{err}\'s AR(1) model Statics:\n\
                  \tvar: {round(self.ar1_gama[err][0],3)}")
            for dt in self.time_intervals[1:]:
                print(f"{err}\'s parameters for time_interval {dt} \n \
                      \t phi: {round(self.ar1_phi[err][dt],3)}\n\
                      \t sigma2: {round(self.ar1_sigma2[err][dt],3)}\n\
                      \t count: {int(self.ar1_gama_count[err][dt])}"
                      )
            print('\n')
        for err in self.mc_errors:
            print(f"{err}\'s Markov Chain model Statics:")
            for dt in self.time_intervals[1:]:
                print(f"{err}\'s parameters for time_interval {dt} \n \
                      \t keep_deactive_rate(p00): {round(self.mc_p00[err][dt],3)}\n\
                      \t keep_active_rate(p11): {round(self.mc_p11[err][dt],3)}\n\
                      \t rho: {round(self.mc_rho[err][dt],3)}\n\
                      \t deactive_count: {int(self.mc_0_count[err][dt])}\n\
                      \t active_count: {int(self.mc_1_count[err][dt])}"
                      )
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

    def add_additional_column(self) -> None:
        """Add additional column to DataFrame."""
        # Add dist column
        self.df["dist"] = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2)
        # Azimuth angle
        self.df["azimuth"] = np.arctan2(self.df["y"], self.df["x"])
        # Visual heading angle
        self.df["visual_heading"] = self.df["yaw"] - self.df["azimuth"]
        self.df.loc[self.df["visual_heading"] > np.pi, "visual_heading"] -= 2 * np.pi
        self.df.loc[self.df["visual_heading"] < -np.pi, "visual_heading"] += 2 * np.pi

    def add_error_columns(self) -> None:
        # Add error columns
        self.df["error_x"] = np.nan
        self.df["error_y"] = np.nan
        self.df["error_yaw"] = np.nan
        self.df["error_delta"] = np.nan
        self.df["error_dist"] = np.nan
        self.df["error_azimuth"] = np.nan

        # load all data, without filtering
        df = self.df

        # check on each index level 0, that has ground_truth and estimation in index level 1
        # then mask of dataframe index that has both ground_truth and estimation
        gt_est_pair_mask: np.ndarray = np.zeros(df.shape[0], dtype=bool)
        pos: int = 0
        for _, item in df.groupby(level=0):
            item_length = item.shape[0]
            pos += item_length
            if "ground_truth" in item.index.get_level_values(1) and "estimation" in item.index.get_level_values(1):
                gt_est_pair_mask[pos - item_length: pos] = True
        df = df[gt_est_pair_mask]

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
        error_distance: np.ndarray = est["dist"].values[valid_mask] - gt["dist"].values[valid_mask]
        error_azimuth: np.ndarray = est["azimuth"].values[valid_mask] - \
            gt["azimuth"].values[valid_mask]
        error_azimuth[error_azimuth > np.pi] -= 2 * np.pi
        error_azimuth[error_azimuth < -np.pi] += 2 * np.pi

        # repeat valid_mask, since each pair has two rows
        valid_mask = np.repeat(valid_mask.values, 2, axis=0)

        # Update error columns
        self.df.loc[gt_mask & valid_mask, "error_x"] = error_x
        self.df.loc[gt_mask & valid_mask, "error_y"] = error_y
        self.df.loc[gt_mask & valid_mask, "error_delta"] = error_delta
        self.df.loc[gt_mask & valid_mask, "error_yaw"] = error_yaw
        self.df.loc[gt_mask & valid_mask, "error_dist"] = error_distance
        self.df.loc[gt_mask & valid_mask, "error_azimuth"] = error_azimuth
        self.df.loc[est_mask & valid_mask, "error_x"] = -error_x
        self.df.loc[est_mask & valid_mask, "error_y"] = -error_y
        self.df.loc[est_mask & valid_mask, "error_delta"] = error_delta
        self.df.loc[est_mask & valid_mask, "error_yaw"] = -error_yaw
        self.df.loc[est_mask & valid_mask, "error_dist"] = -error_distance
        self.df.loc[est_mask & valid_mask, "error_azimuth"] = -error_azimuth

    # TODO: all point loader
    # three types of tables, GT, Est, and GT-Est pair
    def analyze_points(self, **kwargs) -> None:
        self.analyze_points_tp(**kwargs)
        self.analyze_points_fn(**kwargs)
        self.analyze_points_fp(**kwargs)

    def analyze_points_fn(self, **kwargs) -> None:
        """Analyze ground truth data."""
        # Extract data
        df: pd.DataFrame = self.get(**kwargs)

        gt = df[df.index.get_level_values(1) == "ground_truth"].droplevel(level=1)
        gt = gt[gt["status"] == MatchingStatus.FN.value]

        valid_mask: pd.DataFrame = ~np.isnan(gt["x"]) & ~np.isnan(gt["y"]) & ~np.isnan(gt["yaw"])
        self.data_fn: np.ndarray = np.concatenate(
            (
                [gt["x"].values[valid_mask]],
                [gt["y"].values[valid_mask]],
                [gt["yaw"].values[valid_mask]],
                [gt["vx"].values[valid_mask]],
                [gt["vy"].values[valid_mask]],
                [gt["dist"].values[valid_mask]],
                [gt["azimuth"].values[valid_mask]],
                [gt["visual_heading"].values[valid_mask]],
                [gt["width"].values[valid_mask]],
                [gt["length"].values[valid_mask]],
            )
        ).transpose()

    def analyze_points_fp(self, **kwargs) -> None:
        """Analyze estimation data."""
        # Extract data
        df: pd.DataFrame = self.get(**kwargs)

        est = df[df.index.get_level_values(1) == "estimation"].droplevel(level=1)
        est = est[est["status"] == MatchingStatus.FP.value]

        valid_mask: pd.DataFrame = ~np.isnan(est["x"]) & ~np.isnan(est["y"]) & ~np.isnan(est["yaw"])
        self.data_fp: np.ndarray = np.concatenate(
            (
                [est["x"].values[valid_mask]],
                [est["y"].values[valid_mask]],
                [est["yaw"].values[valid_mask]],
                [est["vx"].values[valid_mask]],
                [est["vy"].values[valid_mask]],
                [est["dist"].values[valid_mask]],
                [est["azimuth"].values[valid_mask]],
                [est["visual_heading"].values[valid_mask]],
                [est["width"].values[valid_mask]],
                [est["length"].values[valid_mask]],
            )
        ).transpose()

    def analyze_points_tp(self, **kwargs) -> None:
        # Extract data
        df: pd.DataFrame = self.get(**kwargs)

        # check on each index level 0, that has ground_truth and estimation in index level 1
        # then mask of dataframe index that has both ground_truth and estimation
        gt_est_pair_mask: np.ndarray = np.zeros(df.shape[0], dtype=bool)
        pos: int = 0
        for _, item in df.groupby(level=0):
            item_length = item.shape[0]
            pos += item_length
            if "ground_truth" in item.index.get_level_values(1) and "estimation" in item.index.get_level_values(1):
                gt_est_pair_mask[pos - item_length: pos] = True
        df = df[gt_est_pair_mask]

        # get ground_truth and estimation data
        gt = df[df.index.get_level_values(1) == "ground_truth"].droplevel(level=1)
        est = df[df.index.get_level_values(1) == "estimation"].droplevel(level=1)

        # create mask that indicates whether the ground_truth and estimation are valid
        valid_mask: pd.DataFrame = (
            ~np.isnan(gt["x"])
            & ~np.isnan(gt["y"])
            & ~np.isnan(gt["yaw"])
            & ~np.isnan(est["x"])
            & ~np.isnan(est["y"])
            & ~np.isnan(est["yaw"])
        )

        # fill the data, following definition of DataTableIdx
        self.data_tp_gt: np.ndarray = np.concatenate(
            (
                [gt["x"].values[valid_mask]],
                [gt["y"].values[valid_mask]],
                [gt["yaw"].values[valid_mask]],
                [gt["vx"].values[valid_mask]],
                [gt["vy"].values[valid_mask]],
                [gt["dist"].values[valid_mask]],
                [gt["azimuth"].values[valid_mask]],
                [gt["visual_heading"].values[valid_mask]],
                [gt["width"].values[valid_mask]],
                [gt["length"].values[valid_mask]],
            )
        ).transpose()
        self.data_tp_est: np.ndarray = np.concatenate(
            (
                [est["x"].values[valid_mask]],
                [est["y"].values[valid_mask]],
                [est["yaw"].values[valid_mask]],
                [est["vx"].values[valid_mask]],
                [est["vy"].values[valid_mask]],
                [est["dist"].values[valid_mask]],
                [est["azimuth"].values[valid_mask]],
                [est["visual_heading"].values[valid_mask]],
                [est["width"].values[valid_mask]],
                [est["length"].values[valid_mask]],
            )
        ).transpose()

    def analyze_xy(
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

        # Extract data
        df: pd.DataFrame = self.get(**kwargs)

        # Set axis data labels
        label_axis_x: str = axis_x.get_label()
        label_axis_y: str = axis_y.get_label()

        # Initialize fields
        error_field: PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)
        uncertainty_field: PerceptionFieldXY = PerceptionFieldXY(self.config, axis_x, axis_y)

        # loop for each index
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
                idx_gt_x, idx_gt_y = error_field.get_grid_index(gt[label_axis_x], gt[label_axis_y])
                error_field.num[idx_gt_x, idx_gt_y] += 1

                error_field.x[idx_gt_x, idx_gt_y] += gt["x"]
                error_field.y[idx_gt_x, idx_gt_y] += gt["y"]
                error_field.yaw[idx_gt_x, idx_gt_y] += gt["yaw"]
                error_field.vx[idx_gt_x, idx_gt_y] += gt["vx"]
                error_field.vy[idx_gt_x, idx_gt_y] += gt["vy"]
                error_field.dist[idx_gt_x, idx_gt_y] += np.sqrt(gt["x"] ** 2 + gt["y"] ** 2)
                error_field.confidence[idx_gt_x, idx_gt_y] += gt["confidence"]

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

                error_field.data[idx_gt_x][idx_gt_y][0] = np.append(
                    error_field.data[idx_gt_x][idx_gt_y][0],
                    [
                        gt[
                            [
                                "x",
                                "y",
                                "yaw",
                                "vx",
                                "vy",
                                "dist",
                                "azimuth",
                                "visual_heading",
                                "length",
                                "width",
                            ]
                        ].values
                    ],
                    axis=0,
                )

            if is_est_valid:
                idx_est_x, idx_est_y = uncertainty_field.get_grid_index(
                    est[label_axis_x], est[label_axis_y])
                uncertainty_field.num[idx_est_x, idx_est_y] += 1

                uncertainty_field.x[idx_est_x, idx_est_y] += est["x"]
                uncertainty_field.y[idx_est_x, idx_est_y] += est["y"]
                uncertainty_field.yaw[idx_est_x, idx_est_y] += est["yaw"]
                uncertainty_field.vx[idx_est_x, idx_est_y] += est["vx"]
                uncertainty_field.vy[idx_est_x, idx_est_y] += est["vy"]
                uncertainty_field.dist[idx_est_x,
                                       idx_est_y] += np.sqrt(est["x"] ** 2 + est["y"] ** 2)
                uncertainty_field.confidence[idx_est_x, idx_est_y] += est["confidence"]

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

                uncertainty_field.data[idx_est_x][idx_est_y][0] = np.append(
                    uncertainty_field.data[idx_est_x][idx_est_y][0],
                    [
                        est[
                            [
                                "x",
                                "y",
                                "yaw",
                                "vx",
                                "vy",
                                "dist",
                                "azimuth",
                                "visual_heading",
                                "length",
                                "width",
                            ]
                        ].values
                    ],
                    axis=0,
                )

            if is_paired:
                # get errors
                error_x: float = gt["error_x"]
                error_y: float = gt["error_y"]
                error_delta: float = gt["error_delta"]
                error_azimuth: float = gt["error_azimuth"]

                # error yaw is flipped if it is more than 90 degree
                if -np.pi / 3 < gt["error_yaw"] < np.pi / 3:
                    error_yaw: float = gt["error_yaw"]
                    error_field.num_error_yaw_count[idx_gt_x, idx_gt_y] += 1
                elif 2 * np.pi / 3 < gt["error_yaw"] or -2 * np.pi / 3 > gt["error_yaw"]:
                    error_yaw: float = 0
                    error_field.num_yaw_flip[idx_gt_x, idx_gt_y] += 1
                else:
                    error_yaw: float = 0

                if gt["error_dist"] <= 10:
                    error_dist: float = gt["error_dist"]
                    error_field.num_error_dist_count[idx_gt_x, idx_gt_y] += 1
                else:
                    error_dist: float = 0

                # fill the bins
                idx_gt_x, idx_gt_y = error_field.get_grid_index(gt[label_axis_x], gt[label_axis_y])
                error_field.num_pair[idx_gt_x, idx_gt_y] += 1
                error_field.error_x_mean[idx_gt_x, idx_gt_y] += error_x
                error_field.error_x_std[idx_gt_x, idx_gt_y] += error_x**2
                error_field.error_y_mean[idx_gt_x, idx_gt_y] += error_y
                error_field.error_y_std[idx_gt_x, idx_gt_y] += error_y**2
                error_field.error_yaw_mean[idx_gt_x, idx_gt_y] += error_yaw
                error_field.error_yaw_std[idx_gt_x, idx_gt_y] += error_yaw**2
                error_field.error_delta_mean[idx_gt_x, idx_gt_y] += error_delta
                error_field.error_delta_std[idx_gt_x, idx_gt_y] += error_delta**2
                error_field.error_dist_mean[idx_gt_x, idx_gt_y] += error_dist
                error_field.error_dist_std[idx_gt_x, idx_gt_y] += error_dist**2
                error_field.error_azimuth_mean[idx_gt_x, idx_gt_y] += error_azimuth
                error_field.error_azimuth_std[idx_gt_x, idx_gt_y] += error_azimuth**2

                error_field.data[idx_gt_x][idx_gt_y][1] = np.append(
                    error_field.data[idx_gt_x][idx_gt_y][1],
                    [
                        est[
                            [
                                "x",
                                "y",
                                "yaw",
                                "vx",
                                "vy",
                                "dist",
                                "azimuth",
                                "visual_heading",
                                "length",
                                "width",
                            ]
                        ].values
                    ],
                    axis=0,
                )

                idx_est_x, idx_est_y = uncertainty_field.get_grid_index(
                    est[label_axis_x], est[label_axis_y])
                uncertainty_field.num_pair[idx_est_x, idx_est_y] += 1
                uncertainty_field.error_x_mean[idx_est_x, idx_est_y] += -error_x
                uncertainty_field.error_x_std[idx_est_x, idx_est_y] += error_x**2
                uncertainty_field.error_y_mean[idx_est_x, idx_est_y] += -error_y
                uncertainty_field.error_y_std[idx_est_x, idx_est_y] += error_y**2
                uncertainty_field.error_yaw_mean[idx_est_x, idx_est_y] += -error_yaw
                uncertainty_field.error_yaw_std[idx_est_x, idx_est_y] += error_yaw**2
                uncertainty_field.error_delta_mean[idx_est_x, idx_est_y] += error_delta
                uncertainty_field.error_delta_std[idx_est_x, idx_est_y] += error_delta**2
                uncertainty_field.error_dist_mean[idx_est_x, idx_est_y] += -error_dist
                uncertainty_field.error_dist_std[idx_est_x, idx_est_y] += error_dist**2
                uncertainty_field.error_azimuth_mean[idx_est_x, idx_est_y] += -error_azimuth
                uncertainty_field.error_azimuth_std[idx_est_x, idx_est_y] += error_azimuth**2

                uncertainty_field.data[idx_est_x][idx_est_y][1] = np.append(
                    uncertainty_field.data[idx_est_x][idx_est_y][1],
                    [
                        gt[
                            [
                                "x",
                                "y",
                                "yaw",
                                "vx",
                                "vy",
                                "dist",
                                "azimuth",
                                "visual_heading",
                                "length",
                                "width",
                            ]
                        ].values
                    ],
                    axis=0,
                )

        # process statistics
        error_field.do_post_process()
        uncertainty_field.do_post_process()

        return error_field, uncertainty_field

    def analyze_time(self, axis_t: PerceptionFieldAxis, **kwargs) -> PerceptionFieldTime:
        # Extract data
        df: pd.DataFrame = self.get(**kwargs)

        # Initialize fields
        time_field: PerceptionFieldTime = PerceptionFieldTime(axis_t)

        # group by uuid
        for _, item in df.groupby(level=0):
            # init
            pair = item.droplevel(level=0)
            if not "ground_truth" in pair.index:
                continue
            if pair.loc["ground_truth"]["uuid"] is None:
                continue
            gt = pair.loc["ground_truth"]
            uuid = gt["uuid"]
            assert isinstance(uuid, str)
            assert gt["timestamp"] is not None
            time_field.data_grouped_by_uuid.setdefault(uuid, [])

            # check if the estimation valid.
            is_tp = gt["status"] == MatchingStatus.TP.value and ~np.isnan(
                gt["error_dist"]) and ~np.isnan(gt["error_yaw"])

            # calculate yaw related err
            is_yaw_fliped = False
            is_yaw_flip_ignored = False
            yaw_err = np.nan

            if -np.pi / 3 < gt["error_yaw"] < np.pi / 3:
                yaw_err = gt["error_yaw"]
            elif 2 * np.pi / 3 < gt["error_yaw"] or -2 * np.pi / 3 > gt["error_yaw"]:
                is_yaw_fliped = True
            else:
                is_yaw_flip_ignored = True
            if gt["speed"] > 0.1:
                is_yaw_flip_ignored = True

            state = dict(
                uuid=gt['uuid'],
                timestamp=gt['timestamp'],
                is_tp=is_tp,
                yaw_err=yaw_err,
                is_yaw_fliped=is_yaw_fliped,
                is_yaw_flip_ignored=is_yaw_flip_ignored,
                dist_err=gt['error_dist'],
                ellipse_dist=np.sqrt((gt['y']/0.7)**2 + gt['x']**2),
            )

            # update time_field
            time_field.data_grouped_by_uuid[uuid].append(state)
            time_field.N += 1

            time_field.basic_mean['tp'] += 1 if is_tp else 0
            time_field.basic_mean_count['tp'] += 1

            if not is_yaw_flip_ignored:
                time_field.basic_mean['yaw_flip'] += 1 if is_yaw_fliped else 0
                time_field.basic_mean_count['yaw_flip'] += 1

            for dist_threshold in time_field.dist_thresholds:
                if state['ellipse_dist'] < dist_threshold:
                    time_field.basic_mean[f'tp_{dist_threshold}m'] += 1 if is_tp else 0
                    time_field.basic_mean_count[f'tp_{dist_threshold}m'] += 1
                    break

            if not np.isnan(yaw_err):
                time_field.basic_mean['yaw_err'] += yaw_err
                time_field.basic_mean_count['yaw_err'] += 1

            if ~np.isnan(gt['error_dist']):
                time_field.basic_mean['dist_err'] += gt['error_dist']
                time_field.basic_mean_count['dist_err'] += 1
        time_field.process_basic_statics()

        # accumulate data for AR(1) / Markov Chain model
        for uuid, data in time_field.data_grouped_by_uuid.items():
            # sort by timestamp
            data = sorted(data, key=lambda x: x['timestamp'])
            n_data = len(data)

            # calculate the gama(s) for AR(1) and the transition matrix for Markov Chain
            for i in range(n_data):
                for j in range(i, n_data):
                    dt = round(data[j]["timestamp"]/1_000_000 -
                               data[i]["timestamp"]/1_000_000, 1)
                    if not dt in time_field.mesh_x:
                        continue

                    # calculate tp's transition matrix
                    if dt != time_field.mesh_x[0]:
                        # tp
                        if not data[j]['is_tp']:
                            time_field.mc_0_count['tp'][dt] += 1
                            if not data[i]['is_tp']:
                                time_field.mc_p00['tp'][dt] += 1
                        else:
                            time_field.mc_1_count['tp'][dt] += 1
                            if data[i]['is_tp']:
                                time_field.mc_p11['tp'][dt] += 1
                        # tp_{t}m
                        for dist_threshold in time_field.dist_thresholds:
                            if data[j]['ellipse_dist'] < dist_threshold:
                                if not data[j]['is_tp']:
                                    time_field.mc_0_count[f'tp_{dist_threshold}m'][dt] += 1
                                    if not data[i]['is_tp']:
                                        time_field.mc_p00[f'tp_{dist_threshold}m'][dt] += 1
                                else:
                                    time_field.mc_1_count[f'tp_{dist_threshold}m'][dt] += 1
                                    if data[i]['is_tp']:
                                        time_field.mc_p11[f'tp_{dist_threshold}m'][dt] += 1
                                break

                        # calculate yaw_flip's transition matrix
                        if not data[j]['is_yaw_flip_ignored'] and not data[i]['is_yaw_flip_ignored']:
                            if not data[j]['is_yaw_fliped']:
                                time_field.mc_0_count['yaw_flip'][dt] += 1
                                if not data[i]['is_yaw_fliped']:
                                    time_field.mc_p00['yaw_flip'][dt] += 1
                            else:
                                time_field.mc_1_count['yaw_flip'][dt] += 1
                                if data[i]['is_yaw_fliped']:
                                    time_field.mc_p11['yaw_flip'][dt] += 1

                    # calculate yaw_err's gama(s)
                    yaw_err_mean = time_field.basic_mean['yaw_err']
                    yaw_err_gamma_dt = (data[i]['yaw_err']-yaw_err_mean) * \
                        (data[j]['yaw_err']-yaw_err_mean)
                    if not np.isnan(yaw_err_gamma_dt):
                        time_field.ar1_gama['yaw_err'][dt] += yaw_err_gamma_dt
                        time_field.ar1_gama_count['yaw_err'][dt] += 1

                    # calculate dist_err's gama(s)
                    dist_err_mean = time_field.basic_mean['dist_err']
                    dist_err_gamma_dt = (data[i]['dist_err']-dist_err_mean) * \
                        (data[j]['dist_err']-dist_err_mean)
                    if not np.isnan(dist_err_gamma_dt):
                        time_field.ar1_gama['dist_err'][dt] += dist_err_gamma_dt
                        time_field.ar1_gama_count['dist_err'][dt] += 1

        # Fit AR(1) / Markov Chain model
        time_field.process_ar1_and_mc_parameters()
        time_field.print_model_parameters()

        return time_field
