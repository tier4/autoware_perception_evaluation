# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map (lanelet2) and reachability geometry for the driving-aware metrics.

Modules here are the only place in ``evaluation.metrics`` allowed to import shapely, and they do so
through :mod:`perception_eval.evaluation.metrics.geometry.shapely_compat` so that both shapely 1.8
(Python 3.10) and shapely 2.x are supported.
"""
