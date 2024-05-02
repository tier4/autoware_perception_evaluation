import numpy as np
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.common.transform import TransformKey


def test_homogeneous_matrix():
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


def test_transform_dict():
    matrices = [
        HomogeneousMatrix((1, 0, 0), (1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP),
        HomogeneousMatrix((2, 0, 0), (1, 0, 0, 0), src=FrameID.LIDAR, dst=FrameID.BASE_LINK),
    ]
    transforms = TransformDict(matrices)
    key1 = TransformKey(FrameID.BASE_LINK, FrameID.MAP)
    pos1 = transforms.transform(key1, (1, 0, 0))
    assert np.allclose(pos1, np.array((2, 0, 0)))
