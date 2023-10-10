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
import pytest

from perception_eval.util.math import _is_rotation_matrix, rotation_matrix_to_euler


def test_rotation_matrix_to_euler():
    """[summary]
    Check whether returns correct euler angles.
    """
    expected: np.ndarray = np.array([0, 0, 0])
    answer: np.ndarray = rotation_matrix_to_euler(np.eye(3))
    assert np.allclose(answer, expected)

    with pytest.raises(AssertionError):
        rotation_matrix_to_euler(np.zeros(3))


def test_is_rotation_matrix():
    """[summary]
    Check whether returns correct flag.
    """
    assert _is_rotation_matrix(np.eye(3))
