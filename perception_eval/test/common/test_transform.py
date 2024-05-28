import numpy as np
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.common.transform import TransformKey


def test_homogeneous_matrix_transform():
    ego2map = HomogeneousMatrix((1, 0, 0), (1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    pos1 = ego2map.transform((1, 0, 0))
    assert np.allclose(pos1, np.array((2, 0, 0)))

    pos2 = ego2map.transform(position=(1, 0, 0))
    assert np.allclose(pos2, np.array((2, 0, 0)))

    pos1, rot1 = ego2map.transform((1, 0, 0), (1, 0, 0, 0))
    assert np.allclose(pos1, np.array((2, 0, 0)))
    assert np.allclose(rot1.rotation_matrix, np.eye(3))

    pos2, rot2 = ego2map.transform(position=(1, 0, 0), rotation=(1, 0, 0, 0))
    assert np.allclose(pos2, np.array((2, 0, 0)))
    assert np.allclose(rot2.rotation_matrix, np.eye(3))

    map2ego = HomogeneousMatrix((-1, 0, 0), (1, 0, 0, 0), src=FrameID.MAP, dst=FrameID.BASE_LINK)
    mat1 = ego2map.transform(map2ego)
    assert np.allclose(mat1.matrix, np.eye(4))
    assert np.allclose(mat1.position, np.zeros(3))
    assert np.allclose(mat1.rotation_matrix, np.eye(3))

    mat2 = ego2map.transform(matrix=map2ego)
    assert np.allclose(mat2.matrix, np.eye(4))
    assert np.allclose(mat2.position, np.zeros(3))
    assert np.allclose(mat2.rotation_matrix, np.eye(3))


def test_homogenous_matrix_dot():
    ego2map = HomogeneousMatrix((1, 1, 1), (1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    cam2ego = HomogeneousMatrix((2, 2, 2), (1, 0, 0, 0), src=FrameID.CAM_FRONT, dst=FrameID.BASE_LINK)
    cam2map = ego2map.dot(cam2ego)
    assert np.allclose(
        cam2map.matrix,
        np.array(
            [
                [1, 0, 0, 3],
                [0, 1, 0, 3],
                [0, 0, 1, 3],
                [0, 0, 0, 1],
            ],
        ),
    )
    assert np.allclose(cam2map.position, np.array([3, 3, 3]))  # cam position in map coords
    assert np.allclose(cam2map.rotation_matrix, np.eye(3))  # cam rotation matrix in map coords
    assert cam2map.src == FrameID.CAM_FRONT
    assert cam2map.dst == FrameID.MAP


def test_homogenous_matrix_inv():
    matrix = np.array(
        [
            [0.70710678, -0.70710678, 0.0, 1.0],
            [0.70710678, 0.70710678, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ego2map = HomogeneousMatrix.from_matrix(matrix, FrameID.BASE_LINK, FrameID.MAP)
    inv = ego2map.inv()
    assert np.allclose(
        inv.matrix,
        np.array(
            [
                [0.70710678, 0.70710678, 0.0, -2.12132034],
                [-0.70710678, 0.70710678, 0.0, -0.70710678],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert np.allclose(inv.position, np.array([-2.12132034, -0.70710678, -3.0]))
    assert np.allclose(
        inv.rotation_matrix,
        np.array(
            [
                [0.70710678, 0.70710678, 0.0],
                [-0.70710678, 0.70710678, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_transform_dict():
    ego2map = HomogeneousMatrix((1, 0, 0), (1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    transforms = TransformDict(ego2map)

    key1 = TransformKey(FrameID.BASE_LINK, FrameID.MAP)
    pos1 = transforms.transform(key1, (1, 0, 0))
    assert np.allclose(pos1, np.array((2, 0, 0)))

    key2 = TransformKey(FrameID.MAP, FrameID.BASE_LINK)
    pos2 = transforms.transform(key2, (1, 0, 0))
    assert np.allclose(pos2, np.zeros(3))
