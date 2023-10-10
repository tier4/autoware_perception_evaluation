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


# @pytest.mark.skip(reason="pickle data is deprecated.")
# class TestPerceptionAnalyzer3D:
#         tmpdir.name,

#     def test_analyze(self):
#         """[summary]
#         Test PerceptionAnalyzer3D.analyze().
#         """

#     def test_summarize_ratio(self):
#         """[summary]
#         Test PerceptionAnalyzer3D.summarize_ratio().
#         """

#     def test_summarize_score(self):
#         """[summary]
#         Test PerceptionAnalyzer3D.summarize_score().
#         """

#     def test_summarize_error(self):
#         """[summary]
#         Test PerceptionAnalyzer3D.summarize_error().
#         """

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_plot_state(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to plot state for each columns.
#         """
#         for mode in PlotAxes:
#             self.analyzer.plot_state(

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_plot_error(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to plot error for each column.
#         """
#         for mode in PlotAxes:
#             for heatmap in (False, True):
#                 self.analyzer.plot_error(

#     def test_plot_num_object(self):
#         """[summary]
#         Test to plot number of objects.
#         """
#         for mode in PlotAxes:

#     @pytest.mark.parametrize("columns", (["x", "y"], "yaw", ["w", "l"], ["vx", "vy"]))
#     def test_box_plot(self, columns: Union[str, List[str]]):
#         """[summary]
#         Test to box plot for each column.
#         """
