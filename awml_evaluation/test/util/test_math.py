from awml_evaluation.util.math import _is_rotation_matrix
from awml_evaluation.util.math import rotation_matrix_to_euler
import numpy as np
import pytest


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
