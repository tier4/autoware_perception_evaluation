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

# from perception_eval.tool import PerceptionAnalyzer2D
# from perception_eval.tool import PlotAxes
# import pytest

# @pytest.mark.skip(reason="pickle data is deprecated.")
# class TestPerceptionAnalyzer2D:
#     data_dir: str = osp.join(osp.dirname(__file__), "../sample_data")
#     tmpdir = tempfile.TemporaryDirectory()
#     analyzer = PerceptionAnalyzer2D.from_scenario(
#         tmpdir.name,
#         osp.join(data_dir, "scenario.yaml"),
#     )
#     analyzer.add_from_pkl(osp.join(data_dir, "pkl/sample_result.pkl"))
#     uuid: str = "dcb2b352232fff50c4fad23718f31611"

#     def test_analyze(self):
#         """[summary]
#         Test PerceptionAnalyzer2D.analyze().
#         """
#         self.analyzer.analyze()

#     def test_summarize_ratio(self):
#         """[summary]
#         Test PerceptionAnalyzer2D.summarize_ratio().
#         """
#         self.analyzer.summarize_ratio()

#     def test_summarize_score(self):
#         """[summary]
#         Test PerceptionAnalyzer2D.summarize_score().
#         """
#         self.analyzer.summarize_score()

#     def test_summarize_error(self):
#         """[summary]
#         Test PerceptionAnalyzer2D.summarize_error().
#         """
#         self.analyzer.summarize_error()
