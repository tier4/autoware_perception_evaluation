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


# import os.path as osp
# import tempfile
# from typing import List
# from typing import Union

# from perception_eval.tool import PerceptionPerformanceAnalyzer
# from perception_eval.tool import PlotAxes
# import pytest

# @pytest.mark.skip(reason="pickle data is deprecated.")
# class TestPerceptionPerformanceAnalyzer:
#     data_dir: str = osp.join(osp.dirname(__file__), "../sample_data")
#     tmpdir = tempfile.TemporaryDirectory()
#     analyzer = PerceptionPerformanceAnalyzer.from_scenario(
#         tmpdir.name,
#         osp.join(data_dir, "scenario.yaml"),
#     )
#     analyzer.add_from_pkl(osp.join(data_dir, "pkl/sample_result.pkl"))
#     uuid: str = "dcb2b352232fff50c4fad23718f31611"

#     def test_analyze(self):
#         """[summary]
#         Test PerceptionPerformanceAnalyzer.analyze().
#         """
#         self.analyzer.analyze()

#     def test_summarize_ratio(self):
#         """[summary]
#         Test PerceptionPerformanceAnalyzer.summarize_ratio().
#         """
#         self.analyzer.summarize_ratio()

#     def test_summarize_score(self):
#         """[summary]
#         Test PerceptionPerformanceAnalyzer.summarize_score().
#         """
#         self.analyzer.summarize_score()

#     def test_summarize_error(self):
#         """[summary]
#         Test PerceptionPerformanceAnalyzer.summarize_error().
#         """
#         self.analyzer.summarize_error()

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_plot_state(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to plot state for each columns.
#         """
#         for mode in PlotAxes:
#             self.analyzer.plot_state(
#                 uuid=self.uuid,
#                 columns=columns,
#                 mode=mode,
#             )

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_plot_error(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to plot error for each column.
#         """
#         for mode in PlotAxes:
#             for heatmap in (False, True):
#                 self.analyzer.plot_error(
#                     columns=columns,
#                     mode=mode,
#                     heatmap=heatmap,
#                 )

#     def test_plot_num_object(self):
#         """[summary]
#         Test to plot number of objects.
#         """
#         for mode in PlotAxes:
#             self.analyzer.plot_num_object(mode=mode)

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_box_plot(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to box plot for each column.
#         """
#         self.analyzer.box_plot(columns)
