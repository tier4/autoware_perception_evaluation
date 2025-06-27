from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from pyquaternion import Quaternion

from .schema import FrameID

RotationType = TypeVar("RotationType", ArrayLike, Quaternion)
FrameIDType = TypeVar("FrameIDType", str, FrameID)


class TransformDict:
    @overload
    def __init__(self, matrix: HomogeneousMatrix) -> None:
        ...

    @overload
    def __init__(self, matrices: Sequence[HomogeneousMatrix]) -> None:
        ...

    @overload
    def __init__(self, matrix=None) -> None:
        ...

    def __init__(self, matrices: TransformDictArgType = None) -> None:
        if matrices is None:
            self.__matrices = []
        elif isinstance(matrices, HomogeneousMatrix):
            self.__matrices = [matrices]
        elif isinstance(matrices, (list, tuple)):
            self.__matrices = list(matrices)
        else:
            raise TypeError(f"Expected HomogeneousMatrix, sequence of them or None, but got: {type(matrices)}")

        self.__data = {TransformKey(mat.src, mat.dst): mat for mat in self.__matrices}

    def __reduce__(self) -> Tuple[TransformDict, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (
            self.__class__,
            (self.__matrices,),
        )

    @staticmethod
    def load_key(src: FrameIDType, dst: FrameIDType) -> TransformKey:
        return TransformKey(src, dst)

    def keys(self):
        return self.__data.keys()

    def items(self):
        return self.__data.items()

    def get(self, key: TransformKeyType) -> Optional[HomogeneousMatrix]:
        if not isinstance(key, TransformKey):
            src, dst = key
            key = self.load_key(src, dst)
        return self.__data.get(key, None)

    def __getitem__(self, key: TransformKeyType) -> HomogeneousMatrix:
        if not isinstance(key, TransformKey):
            src, dst = key
            key = self.load_key(src, dst)
        return self.__data[key]

    def __setitem__(self, key: TransformKeyType, value: HomogeneousMatrix) -> None:
        if not isinstance(key, TransformKey):
            src, dst = key
            key = self.load_key(src, dst)
        self.__data[key] = value

    def __delitem__(self, key: TransformKeyType) -> None:
        if not isinstance(key, TransformKey):
            src, dst = key
            key = self.load_key(src, dst)
        del self.__data[key]

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> Iterator:
        return iter(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__matrices})"

    @overload
    def transform(self, key: TransformKeyType, position: ArrayLike) -> ArrayLike:
        ...

    @overload
    def transform(
        self,
        key: TransformKeyType,
        position: ArrayLike,
        rotation: RotationType,
    ) -> Tuple[ArrayLike, Quaternion]:
        ...

    @overload
    def transform(self, key: TransformKeyType, matrix: HomogeneousMatrix) -> HomogeneousMatrix:
        ...

    def transform(self, key: TransformKeyType, *args: TransformArgType, **kwargs: TransformArgType) -> TransformArgType:
        """
        Args:
        -----
            key (TransformKeyType): _description_

        Raises:
        -------
            ValueError | KeyError: Expecting `transform(position)`, `transform(position, rotation)` or `transform(matrix)`.

        Returns:
        --------
            TransformArgType: Return `NDArray` if the input is `position: ArrayLike`,
                `(NDArray, Quaternion)` if the input is `position: ArrayLike, rotation: RotationType` or
                `HomogeneousMatrix` if the input is `matrix: HomogeneousMatrix`.

        Examples:
        ---------
            >>> matrix = HomogeneousMatrix((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
            >>> transforms = TransformDict(matrix)
            # Use registered transform matrix
            ## specify only position
            >>> key = TransformKey(FrameID.BASE_LINK, FrameID.MAP)
            >>> transforms.transform(key, (1.0, 0.0, 0.0))
            array([2., 0., 0.])
            ## specify position and rotation
            >>> transforms.transform(key, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            (array([2., 0., 0.]), Quaternion(1.0, 0.0, 0.0, 0.0))
            ## specify homogeneous matrix
            >>> other = HomogeneousMatrix(key, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.MAP, dst=FrameID.BASE_LINK)
            >>> matrix.transform(other).matrix
            array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
            # Inverse transform matrix is not registered but can use it
            >>> key = TransformKey(FrameID.MAP, FrameID.MAP)
            >>> transforms.transform(key, (1.0, 0.0, 0.))
            array([0., 0., 0.])
            # If source and destination frame id is same, return input value
            >>> key = TransformKey(FrameID.BASE_LINK, FrameID.BASE_LINK)
            >>> transforms.transform(key, [1.0, 0.0, 0.0])
            array([1., 0., 0.])
        """
        if not isinstance(key, TransformKey):
            src, dst = key
        else:
            src = key.src
            dst = key.dst

        if src == dst:
            s = len(args)
            if s == 0:
                if not kwargs:
                    raise ValueError("At least 1 argument specified")

                if "position" in kwargs:
                    position = kwargs["position"]
                    if "matrix" in kwargs:
                        raise ValueError("Cannot specify `position` and `matrix` at the same time.")
                    elif "rotation" in kwargs:
                        return position, kwargs["rotation"]
                    else:
                        return position
                elif "matrix" in kwargs:
                    return kwargs["matrix"]
                else:
                    raise KeyError(f"Unexpected keys are detected: {list(kwargs.keys())}")
            elif s == 1:
                return args[0]
            elif s == 2:
                return args
            else:
                raise ValueError(f"Unexpected number of arguments {s}")
        else:
            matrix = self.get((src, dst))
            if matrix is None:
                # search matrix dst->src and if exists use inverse matrix
                matrix = self.get((dst, src))
                if matrix is None:
                    raise KeyError(f"No transform matrix is registered both {src}->{dst} and {dst}->{src}")
                matrix = matrix.inv()
            return matrix.transform(*args, **kwargs)

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        if self.__matrices is None:
            matrices = None
            matrices_type = "None"
        elif isinstance(self.__matrices, HomogeneousMatrix):
            matrices = matrices.serialization()
            matrices_type = HomogeneousMatrix.__name__
        elif isinstance(self.__matrices, (list, tuple)):
            matrices = self.__matrices
            matrices_type = list.__name__

        return {"matrices": matrices, "matrices_type": matrices_type}

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> TransformDict:
        """Deserialize the data to TransformDict."""
        if data["matrices_type"] == "None":
            matrices = None
        elif data["matrices_type"] == HomogeneousMatrix.__name__:
            matrices = HomogeneousMatrix.deserialization(data["matrices"])
        elif data["matrices_type"] == list.__name__:
            matrices = data["matrices"]

        return cls(matrices=matrices)


class TransformKey:
    def __init__(self, src: FrameIDType, dst: FrameIDType) -> None:
        self.src = FrameID.from_value(src) if isinstance(src, str) else src
        self.dst = FrameID.from_value(dst) if isinstance(dst, str) else dst

    def __hash__(self) -> int:
        return hash((self.src, self.dst))

    def __str__(self) -> str:
        return str((self.src, self.dst))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.src}, {self.dst})"

    def __eq__(self, other: TransformKeyType) -> bool:
        if isinstance(other, TransformKey):
            return self.src == other.src and self.dst == other.dst
        elif isinstance(other, (tuple, list)):
            other_src, other_dst = other
            return self.src == other_src and self.dst == other_dst
        else:
            return False


TransformKeyType = TypeVar("TransformKeyType", TransformKey, Tuple[Union[str, FrameID], Union[str, FrameID]])


class HomogeneousMatrix:
    def __init__(self, position: ArrayLike, rotation: RotationType, src: FrameIDType, dst: FrameIDType) -> None:
        """
        Args:
            position (ArrayLike): 3D position ordering `(x, y, z)`.
            rotation (ArrayLike | Quaternion): Rotation quaternion, which is ordering `(w, x, y, z)`, or matrix in the shape 3x3 or 4x4.
            src (str | FrameID]): Source frame ID.
            dst (str | FrameID): Destination frame ID.
        """
        self.position = position if isinstance(position, np.ndarray) else np.array(position)

        if isinstance(rotation, np.ndarray) and rotation.ndim == 2:
            rotation = Quaternion(matrix=rotation)
        else:
            rotation = Quaternion(rotation)
        self.rotation = rotation

        self.src = FrameID.from_value(src) if isinstance(src, str) else src
        self.dst = FrameID.from_value(dst) if isinstance(dst, str) else dst

        self.matrix = self.__generate_homogeneous_matrix(position, rotation)

    def __reduce__(self) -> Tuple[HomogeneousMatrix, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (
            self.__class__,
            (
                self.position,
                self.rotation,
                self.src,
                self.dst,
            ),
        )

    @classmethod
    def from_matrix(cls, matrix: NDArray, src: Union[str, FrameID], dst: Union[str, FrameID]) -> HomogeneousMatrix:
        position, rotation = cls.__extract_position_and_rotation_from_matrix(matrix)
        return cls(position, rotation, src, dst)

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def yaw_pitch_roll(self) -> Tuple[float, float, float]:
        """Get the equivalent (yaw, pitch, roll) angles in radius.

        Returns:
        --------
            Tuple[float, float, float]:
                yaw: Rotation angle around the z-axis in radians, in the range `[-pi, pi]`.
                pitch: Rotation angle around the y-axis in radians, in the range `[-pi/2, -pi/2]`.
                roll: Rotation angle around the x-axis in radians, in the range `[-pi, pi]`.
        """
        return self.rotation.yaw_pitch_roll

    @property
    def rotation_matrix(self) -> NDArray:
        """Get the 3x3 rotation matrix equivalent of the quaternion rotation.

        Returns:
        --------
            NDArray: A 3x3 orthogonal rotation matrix.
        """
        return self.rotation.rotation_matrix

    @staticmethod
    def __extract_position_and_rotation_from_matrix(
        matrix: Union[NDArray, HomogeneousMatrix]
    ) -> Tuple[NDArray, Quaternion]:
        """Extract the equivalent position and rotation matrix from a 4x4 homogeneous matrix.

        Args:
        -----
            matrix (NDArray | HomogeneousMatrix): A 4x4 homogeneous matrix.

        Raises:
        -------
            ValueError: Expecting the input NDArray matrix has the shape 4x4.

        Returns:
        --------
            Tuple[NDArray, Quaternion]:
                position: 3D position, ordering `(x, y, z)`.
                rotation: Quaternion ordering `(w, x, y, z)`.
        """
        if isinstance(matrix, np.ndarray):
            if matrix.shape != (4, 4):
                raise ValueError(f"Homogeneous matrix must be 4x4, but got {matrix.shape}")

            position = matrix[:3, 3]
            rotation = matrix[:3, :3]
            return position, Quaternion(matrix=rotation)
        else:
            return matrix.position, matrix.rotation

    @staticmethod
    def __generate_homogeneous_matrix(position: ArrayLike, rotation: RotationType) -> NDArray:
        """Generate 4x4 homogeneous matrix.

        Args:
        -----
            position (ArrayLike): 3D position.
            rotation (ArrayLike | Quaternion): Rotation quaternion or matrix.

        Returns:
        --------
            NDArray: A 4x4 homogeneous matrix.
        """
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        assert position.shape == (3,), f"Expected position shape is (3,), but got {position.shape}"

        if not isinstance(rotation, Quaternion):
            if isinstance(rotation, np.ndarray) and rotation.ndim == 2:
                rotation = Quaternion(matrix=rotation)
            else:
                rotation = Quaternion(rotation)

        matrix = np.eye(4)
        matrix[:3, 3] = position
        matrix[:3, :3] = rotation.rotation_matrix
        return matrix

    def dot(self, other: HomogeneousMatrix) -> HomogeneousMatrix:
        """Return a dot product between self and other.

        Args:
        -----
            other (HomogeneousMatrix): Other matrix.

        Raises:
        -------
            ValueError: Expecting `self.src` and `other.dst` frame id is same.

        Returns:
        --------
            HomogeneousMatrix: Result of dot product.

        Examples:
        ---------
            >>> ego2map = HomogeneousMatrix((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
            >>> cam2ego = HomogeneousMatrix((2.0, 2.0, 2.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.CAM_FRONT, dst=FrameID.BASE_LINK)
            >>> cam2map = ego2map.dot(cam2ego)
            >>> cam2map.matrix
                array([[1., 0., 0., 3.],
                       [0., 1., 0., 3.],
                       [0., 0., 1., 3.],
                       [0., 0., 0., 1.]])
        """
        if self.src != other.dst:
            raise ValueError(f"self.src != other.dst: self.src={self.src}, other.dst={other.dst}")

        ret_mat = self.matrix.dot(other.matrix)
        position, rotation = self.__extract_position_and_rotation_from_matrix(ret_mat)
        return HomogeneousMatrix(position, rotation, src=other.src, dst=self.dst)

    def inv(self) -> HomogeneousMatrix:
        """Return an inverse matrix.

        Returns:
        --------
            HomogeneousMatrix: Inverse matrix.
        """
        ret_mat = np.linalg.inv(self.matrix)
        position, rotation = self.__extract_position_and_rotation_from_matrix(ret_mat)
        return HomogeneousMatrix(position, rotation, src=self.dst, dst=self.src)

    def __transform_matrix(self, matrix: HomogeneousMatrix) -> HomogeneousMatrix:
        """Transform with the specified homogeneous matrix.

        Args:
        -----
            matrix (HomogeneousMatrix): A `HomogeneousMatrix` instance, which `matrix.dst` frame id must be same with `self.src`.

        Returns:
        --------
            HomogeneousMatrix: Result of a dot product.
        """
        return matrix.dot(self)

    def __transform_position(self, position: ArrayLike) -> NDArray:
        """Transform with the specified 3D position ordering `(x, y, z)`.

        Args:
            position (ArrayLike): 3D position.

        Returns:
            NDArray: Transformed 3D position.
        """
        rotation = Quaternion()
        matrix = self.__generate_homogeneous_matrix(position, rotation)
        ret_mat = self.matrix.dot(matrix)
        ret_pos, _ = self.__extract_position_and_rotation_from_matrix(ret_mat)
        return ret_pos

    def __transform_position_and_rotation(
        self, position: ArrayLike, rotation: Union[ArrayLike, Quaternion]
    ) -> Tuple[NDArray, Quaternion]:
        """Transform with specified position and rotation.

        Args:
        -----
            position (ArrayLike): 3D position.
            rotation (Union[ArrayLike, Quaternion]): Rotation quaternion or matrix.

        Returns:
        --------
            Tuple[NDArray, Quaternion]:
                position: Transformed 3D position.
                rotation: Transformed quaternion rotation.
        """
        matrix = self.__generate_homogeneous_matrix(position, rotation)
        ret_mat = self.matrix.dot(matrix)
        return self.__extract_position_and_rotation_from_matrix(ret_mat)

    @overload
    def transform(self, position: ArrayLike) -> NDArray:
        ...

    @overload
    def transform(self, position: ArrayLike, rotation: RotationType) -> Tuple[NDArray, Quaternion]:
        ...

    @overload
    def transform(self, matrix: HomogeneousMatrix) -> HomogeneousMatrix:
        ...

    def transform(self, *args, **kwargs) -> TransformArgType:
        """Transform with specified position, rotation or homogeneous matrix.

        Raises:
        -------
            KeyError: Unexpected kwargs are specified.
            ValueError: Unexpected number of arguments are specified, expecting 1 or 2.
            TypeError: Unexpected type is input.

        Returns:
        --------
            TransformReturnType: Transform result(s).

        Examples:
        ---------
            >>> matrix = HomogeneousMatrix((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
            # specify only position
            >>> matrix.transform((1.0, 0.0, 0.0))
            array([2., 0., 0.])
            >>> matrix.transform(position=(1, 0, 0))
            array([2., 0., 0.])
            # specify position and rotation
            >>> matrix.transform((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            (array([2., 0., 0.]), Quaternion(1.0, 0.0, 0.0, 0.0))
            >>> matrix.transform(position=(1.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))
            (array([2., 0., 0.]), Quaternion(1.0, 0.0, 0.0, 0.0))
            # specify homogeneous matrix
            >>> ego2map = HomogeneousMatrix((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
            >>> cam2ego = HomogeneousMatrix((2.0, 2.0, 2.0), (1.0, 0.0, 0.0, 0.0), src=FrameID.CAM_FRONT, dst=FrameID.BASE_LINK)
            >>> cam2map = cam2ego.transform(ego2map)
            >>> cam2map.matrix
                array([[1., 0., 0., 3.],
                       [0., 1., 0., 3.],
                       [0., 0., 1., 3.],
                       [0., 0., 0., 1.]])
            >>> cam2map.src
                <FrameID.CAM_FRONT: 'cam_front'>
            >>> cam2map.dst
                <FrameID.MAP: 'map'>
        """
        s = len(args)
        if s == 0:
            if not kwargs:
                raise ValueError("At least 1 argument specified")

            if "position" in kwargs:
                position = kwargs["position"]
                if "matrix" in kwargs:
                    raise ValueError("Cannot specify `position` and `matrix` at the same time.")
                elif "rotation" in kwargs:
                    rotation = kwargs["rotation"]
                    return self.__transform_position_and_rotation(position, rotation)
                else:
                    return self.__transform_position(position)
            elif "matrix" in kwargs:
                matrix = kwargs["matrix"]
                return self.__transform_matrix(matrix)
            else:
                raise KeyError(f"Unexpected keys are detected: {list(kwargs.keys())}")
        elif s == 1:
            arg = args[0]
            if isinstance(arg, HomogeneousMatrix):
                return self.__transform_matrix(matrix=arg)
            else:
                return self.__transform_position(position=arg)
        elif s == 2:
            position, rotation = args
            return self.__transform_position_and_rotation(position, rotation)
        else:
            raise ValueError(f"Unexpected number of arguments {s}")

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.elements,
            "src": self.src.value,
            "dst": self.dst.value,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> HomogeneousMatrix:
        """Deserialize the data to HomogeneousMatrix."""
        return cls(
            position=np.array(data["position"]), rotation=Quaternion(data["rotation"]), src=data["src"], dst=data["dst"]
        )


TransformArgType = TypeVar("TransformArgType", HomogeneousMatrix, NDArray, Tuple[NDArray, Quaternion])
TransformDictArgType = TypeVar("TransformDictArgType", HomogeneousMatrix, Sequence[HomogeneousMatrix], None)
