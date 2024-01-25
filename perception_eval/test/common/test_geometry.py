import numpy as np
from perception_eval.common.geometry import interpolate_hopmogeneous_matrix
from perception_eval.common.geometry import interpolate_list
from perception_eval.common.geometry import interpolate_quaternion
from pyquaternion import Quaternion
import pytest


def test_interpolate_hopmogeneous_matrix():
    matrix_1 = np.eye(4)
    matrix_2 = np.eye(4)
    matrix_2[1, 3] = 1  # 小さな変更を加える
    t1, t2, t = 0, 1, 0.5

    result = interpolate_hopmogeneous_matrix(matrix_1, matrix_2, t1, t2, t)
    expected = np.eye(4)
    expected[1, 3] = 0.5  # 期待される結果

    assert np.array_equal(result, expected)


def test_interpolate_list():
    list_1 = [0, 0, 0]
    list_2 = [1, 1, 1]
    t1, t2, t = 0, 1, 0.5

    result = interpolate_list(list_1, list_2, t1, t2, t)
    expected = [0.5, 0.5, 0.5]

    assert result == expected


def test_interpolate_quaternion():
    quat_1 = Quaternion()
    quat_2 = Quaternion(axis=[0, 1, 0], angle=np.pi)
    t1, t2, t = 0, 1, 0.5

    result = interpolate_quaternion(quat_1, quat_2, t1, t2, t)
    expected = Quaternion.slerp(quat_1, quat_2, 0.5)

    assert result == expected
