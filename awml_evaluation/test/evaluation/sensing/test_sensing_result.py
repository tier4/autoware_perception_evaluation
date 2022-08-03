from test.util.dummy_object import make_dummy_data

from awml_evaluation.evaluation.sensing.sensing_result import DynamicObjectWithSensingResult
import numpy as np


def test_get_nearest_point():
    """[summary]
    Test calculation of the nearest point.
    """
    objects, _ = make_dummy_data()
    pointcloud: np.ndarray = np.array([[0.5, 1.0, 1.0], [0.8, -1.0, 1.0], [-1.2, 1.0, 1.0]])
    for i, obj in enumerate(objects):
        result = DynamicObjectWithSensingResult(
            obj,
            pointcloud,
            scale_factor=1.0,
            min_points_threshold=1,
        )
        assert np.allclose(result.nearest_point, pointcloud[i])
