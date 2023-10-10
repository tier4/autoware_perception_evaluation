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

import numpy as np

from perception_eval.tool import Gmm


def test_gmm():
    """Test GMM."""
    max_k: int = 10
    x_dim: int = 5
    y_dim: int = 3
    num_sample: int = 100
    data: np.ndarray = np.random.rand(num_sample, x_dim + y_dim)
    gmm = Gmm(max_k=max_k)
    gmm.fit(data)
    gamma = gmm.get_gamma(data)
    assert gamma.shape == (num_sample, gmm.num_k)
    x: np.ndarray = np.random.rand(num_sample, x_dim)

    mean_y: np.ndarray = gmm.predict(x, "mean")
    mode_y: np.ndarray = gmm.predict(x, "mode")
    assert mean_y.shape == (num_sample, y_dim)
    assert mode_y.shape == (num_sample, y_dim)
