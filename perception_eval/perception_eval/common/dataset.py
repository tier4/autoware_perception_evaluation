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

from copy import deepcopy
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from nuimages import NuImages
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from perception_eval.common import ObjectType
from perception_eval.common.dataset_utils import _sample_to_frame
from perception_eval.common.dataset_utils import _sample_to_frame_2d
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.geometry import interpolate_homogeneous_matrix
from perception_eval.common.geometry import interpolate_object_list
from perception_eval.common.label import LabelConverter
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.util.math import get_pose_transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm


class FrameGroundTruth:
    """
    Ground truth data per frame

    Attributes:
        unix_time (float): The unix time for the frame [us].
        frame_name (str): The file name for the frame.
        objects (List[DynamicObject]): Objects data.
        pointcloud (Optional[numpy.ndarray], optional):
                Pointcloud data. Defaults to None, but if you want to visualize dataset,
                you should load pointcloud data.
        transform_matrix (Optional[np.ndarray]): The numpy array to transform position.
        objects (List[ObjectType]): Objects data.
        ego2map (Optional[np.ndarray]): The numpy array to transform position.
        raw_data (Optional[Dict[str, numpy.ndarray]]): Raw data for each sensor modality.

    Args:
        unix_time (int): The unix time for the frame [us]
        frame_name (str): The file name for the frame
        objects (List[DynamicObject]): Objects data.
        ego2map (Optional[np.ndarray]): The array of 4x4 matrix.
            Transform position with respect to vehicle coord system to map one.
        raw_data (Optional[Dict[str, numpy.ndarray]]): Raw data for each sensor modality.
    """

    def __init__(
        self,
        unix_time: int,
        frame_name: str,
        objects: List[DynamicObject],
        ego2map: Optional[np.ndarray] = None,
        raw_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.objects: List[ObjectType] = objects
        self.ego2map: Optional[np.ndarray] = ego2map
        self.raw_data: Optional[Dict[str, np.ndarray]] = raw_data


def load_all_datasets(
    dataset_paths: List[str],
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_id: Union[FrameID, Sequence[FrameID]],
    load_raw_data: bool = False,
) -> List[FrameGroundTruth]:
    """
    Load tier4 datasets.

    Args:
        dataset_paths (List[str]): The list of root paths to dataset
        evaluation_tasks (EvaluationTask): The evaluation task
        label_converter (LabelConverter): Label convertor
        frame_id (FrameID): FrameID instance, where objects are with respect.
        load_raw_data (bool): The flag of setting pointcloud or image.
            For 3D task, pointcloud will be loaded. For 2D, image will be loaded. Defaults to False.

    Returns:
        List[FrameGroundTruth]: FrameGroundTruth instance list.

    Examples:
        >>> evaluation_task = EvaluationTask.DETECTION
        >>> converter = LabelConverter(evaluation_task, False, "autoware")
        >>> load_all_datasets(["./data"], evaluation_task, converter, "base_link")
        [<perception_eval.common.dataset.FrameGroundTruth object at 0x7f66040c36a0>, ...]
    """
    logging.info(f"Start to load dataset {dataset_paths}")

    if isinstance(frame_id, FrameID):
        frame_ids: List[FrameID] = [frame_id]
    elif isinstance(frame_id, (list, tuple)):
        frame_ids = list(frame_id)
    else:
        raise TypeError(f"Unexpected frame id type: {type(frame_id)}")

    logging.info(
        f"config: load_raw_data: {load_raw_data}, evaluation_task: {evaluation_task}, "
        f"frame_id: {[fr.value for fr in frame_ids]}"
    )

    all_datasets: List[FrameGroundTruth] = []

    for dataset_path in dataset_paths:
        all_datasets += _load_dataset(
            dataset_path=dataset_path,
            evaluation_task=evaluation_task,
            label_converter=label_converter,
            frame_ids=frame_ids,
            load_raw_data=load_raw_data,
        )
    logging.info("Finish loading dataset\n" + _get_str_objects_number_info(label_converter))
    return all_datasets


def _load_dataset(
    dataset_path: str,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_ids: List[FrameID],
    load_raw_data: bool,
) -> List[FrameGroundTruth]:
    """
    Load one tier4 dataset.
    Args:
        dataset_path (str): The root path to dataset.
        evaluation_tasks (EvaluationTask): The evaluation task.
        label_converter (LabelConverter): LabelConvertor instance.
        frame_ids (List[FrameID]): FrameID instance, where objects are with respect.
        load_raw_data (bool): Whether load pointcloud/image data.

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """

    nusc: NuScenes = NuScenes(version="annotation", dataroot=dataset_path, verbose=False)
    nuim: Optional[NuImages] = (
        NuImages(version="annotation", dataroot=dataset_path, verbose=False) if evaluation_task.is_2d() else None
    )
    helper: PredictHelper = PredictHelper(nusc)

    if len(nusc.visibility) == 0:
        logging.warn("visibility is not annotated")

    # Load category list
    category_list = []
    for category_dict in nusc.category:
        category_list.append(category_dict["name"])

    # Read out all sample_tokens in DB.
    sample_tokens = _get_sample_tokens(nusc.sample)

    dataset: List[FrameGroundTruth] = []
    for n, sample_token in enumerate(tqdm(sample_tokens)):
        if evaluation_task.is_2d():
            frame = _sample_to_frame_2d(
                nusc=nusc,
                nuim=nuim,
                sample_token=sample_token,
                evaluation_task=evaluation_task,
                label_converter=label_converter,
                frame_ids=frame_ids,
                frame_name=str(n),
                load_raw_data=load_raw_data,
            )
        else:
            assert len(frame_ids) == 1, f"For 3D evaluation, only one Frame ID must be specified, but got {frame_ids}"
            frame = _sample_to_frame(
                nusc=nusc,
                helper=helper,
                sample_token=sample_token,
                evaluation_task=evaluation_task,
                label_converter=label_converter,
                frame_id=frame_ids[0],
                frame_name=str(n),
                load_raw_data=load_raw_data,
            )
        dataset.append(frame)
    return dataset


def _get_str_objects_number_info(
    label_converter: LabelConverter,
) -> str:
    """Get str for the information of object label number.
    Args:
        label_converter (LabelConverter): label convertor.
    Returns:
        str: string.
    """
    str_: str = ""
    for label_info in label_converter.label_infos:
        str_ += f"{label_info.name} (-> {label_info.label}): {label_info.num} \n"
    return str_


class DatasetLoadingError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def _get_sample_tokens(nuscenes_sample: dict) -> List[Any]:
    """Get sample tokens

    Args:
        nuscenes_sample (dict): nusc.sample

    Raises:
        DatasetLoadingError: Dataset loading error

    Returns:
        List[Any]: [description]
    """

    sample_tokens_all: List[Any] = [s["token"] for s in nuscenes_sample]
    if len(sample_tokens_all) < 1:
        raise DatasetLoadingError("Error: Database has no samples!")

    sample_tokens = []
    for sample_token in sample_tokens_all:
        sample_tokens.append(sample_token)

    return sample_tokens_all


def get_now_frame(
    ground_truth_frames: List[FrameGroundTruth],
    unix_time: int,
    threshold_min_time: int,
) -> Optional[FrameGroundTruth]:
    """Select the ground truth frame whose unix time is most close to args unix time from dataset.

    Args:
        ground_truth_frames (List[FrameGroundTruth]): FrameGroundTruth instance list.
        unix_time (int): Unix time [us].
        threshold_min_time (int): Min time for unix time difference [us].

    Returns:
        Optional[FrameGroundTruth]:
                The ground truth frame whose unix time is most close to args unix time
                from dataset.
                If the difference time between unix time parameter and the most close time
                ground truth frame is larger than threshold_min_time, return None.

    Examples:
        >>> ground_truth_frames = load_all_datasets(...)
        >>> get_now_frame(ground_truth_frames, 1624157578750212, 7500)
        <perception_eval.common.dataset.FrameGroundTruth object at 0x7f66040c36a0>
    """

    # error handling
    threshold_max_time = 10**17
    if unix_time > threshold_max_time:
        raise DatasetLoadingError(
            f"Error: The unit time of unix time is micro second,\
             but you may input nano second {unix_time}"
        )

    ground_truth_now_frame: FrameGroundTruth = ground_truth_frames[0]
    min_time: int = abs(unix_time - ground_truth_now_frame.unix_time)

    for ground_truth_frame in ground_truth_frames:
        diff_time = abs(unix_time - ground_truth_frame.unix_time)
        if diff_time < min_time:
            ground_truth_now_frame = ground_truth_frame
            min_time = diff_time
        # if 0 < len(ground_truth_frame.objects) and (any([obj.uuid in ("10321", "179948") for obj in ground_truth_frame.objects]) and diff_time < threshold_min_time or ground_truth_frame.objects[0].uuid == "10321"):
        #     logging.info(f"uuids: {[obj.uuid for obj in ground_truth_frame.objects]}, diff time: {diff_time / 1000}")
    if min_time > threshold_min_time:
        logging.info(
            f"Now frame is {ground_truth_now_frame.unix_time} and time difference \
                 is {min_time / 1000} ms > {threshold_min_time / 1000} ms"
        )
        return None
    else:
        return ground_truth_now_frame


def get_interpolated_now_frame(
    ground_truth_frames: List[FrameGroundTruth],
    unix_time: int,
    threshold_min_time: int,
) -> Optional[FrameGroundTruth]:
    """Get interpolated ground truth frame in specified unix time.
    It searches before and after frames which satisfy the time difference condition and if found both, interpolate them.

    Args:
        ground_truth_frames (List[FrameGroundTruth]): FrameGroundTruth instance list.
        unix_time (int): Unix time [us].
        threshold_min_time (int): Min time for unix time difference [us].

    Returns:
        Optional[FrameGroundTruth]:
                The ground truth frame whose unix time is most close to args unix time
                from dataset.
                If the difference time between unix time parameter and the most close time
                ground truth frame is larger than threshold_min_time, return None.

    Examples:
        >>> ground_truth_frames = load_all_datasets(...)
        >>> get_interpolated_now_frame(ground_truth_frames, 1624157578750212, 7500)
        <perception_eval.common.dataset.FrameGroundTruth object at 0x7f66040c36a0>
    """
    # extract closest two frames
    before_frame = None
    after_frame = None
    dt_before = 0.0
    dt_after = 0.0
    for ground_truth_frame in ground_truth_frames:
        diff_time = unix_time - ground_truth_frame.unix_time
        if diff_time >= 0:
            before_frame = ground_truth_frame
            dt_before = diff_time
        else:
            after_frame = ground_truth_frame
            dt_after = -diff_time
        if before_frame is not None and after_frame is not None:
            break

    # disable frame if time difference is too large
    if dt_before > threshold_min_time:
        before_frame = None
    if dt_after > threshold_min_time:
        after_frame = None

    # check frame availability
    if before_frame is None and after_frame is None:
        logging.info(f"No frame is available for interpolation")
        return None
    elif before_frame is None:
        logging.info(f"Only after frame is available for interpolation")
        return after_frame
    elif after_frame is None:
        logging.info(f"Only before frame is available for interpolation")
        return before_frame
    else:
        # do interpolation
        return interpolate_ground_truth_frames(before_frame, after_frame, unix_time)


def interpolate_ground_truth_frames(
    before_frame: FrameGroundTruth,
    after_frame: FrameGroundTruth,
    unix_time: int,
):
    """Interpolate ground truth frame with linear interpolation.

    Args:
        before_frame (FrameGroundTruth): input frame1
        after_frame (FrameGroundTruth): input frame2
        unix_time (int): target time
    """
    # interpolate ego2map
    ego2map = interpolate_homogeneous_matrix(
        before_frame.ego2map, after_frame.ego2map, before_frame.unix_time, after_frame.unix_time, unix_time
    )

    # TODO: Need refactor for simplicity
    # if frame is base_link, need to interpolate with global coordinate
    # 1. convert object list to global
    before_frame_objects = convert_objects_to_global(before_frame.objects, before_frame.ego2map)
    after_frame_objects = convert_objects_to_global(after_frame.objects, after_frame.ego2map)

    # 2. interpolate objects
    object_list = interpolate_object_list(
        before_frame_objects, after_frame_objects, before_frame.unix_time, after_frame.unix_time, unix_time
    )
    # 3. convert object list to base_link
    # object_list = convert_objects_to_base_link(object_list, ego2map)

    # interpolate raw data
    output_frame = deepcopy(before_frame)
    output_frame.ego2map = ego2map
    output_frame.objects = object_list
    output_frame.unix_time = unix_time
    return output_frame


def convert_objects_to_global(
    object_list: List[ObjectType],
    ego2map: np.ndarray,
) -> List[ObjectType]:
    """Convert object list to global coordinate.

    Args:
        object_list (List[ObjectType]): object list
        ego2map (np.ndarray): ego2map matrix

    Returns:
        List[ObjectType]: object list in global coordinate
    """
    output_object_list = []
    for object in object_list:
        if object.frame_id == "map":
            output_object_list.append(deepcopy(object))
            continue
        elif object.frame_id == "base_link":
            src: np.ndarray = get_pose_transform_matrix(
                position=object.state.position,
                rotation=object.state.orientation.rotation_matrix,
            )
            dst: np.ndarray = ego2map.dot(src)
            updated_position: np.ndarray = tuple(dst[:3, 3].flatten())
            updated_rotation: np.ndarray = Quaternion(dst[:3, :3])
            output_object = deepcopy(object)
            output_object.state.position = updated_position
            output_object.state.orientation = updated_rotation
            output_object.frame_id = "map"
            output_object_list.append(output_object)
        else:
            raise NotImplementedError(f"Unexpected frame_id: {object.frame_id}")
    return output_object_list


def convert_objects_to_base_link(
    object_list: List[ObjectType],
    ego2map: np.ndarray,
) -> List[ObjectType]:
    """Convert object list to base_link coordinate.

    Args:
        object_list (List[ObjectType]): object list
        ego2map (np.ndarray): ego2map matrix

    Returns:
        List[ObjectType]: object list in base_link coordinate
    """
    output_object_list = []
    for object in object_list:
        if object.frame_id == "base_link":
            output_object_list.append(deepcopy(object))
            continue
        elif object.frame_id == "map":
            src: np.ndarray = get_pose_transform_matrix(
                position=object.state.position,
                rotation=object.state.orientation.rotation_matrix,
            )
            dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
            updated_position: np.ndarray = tuple(dst[:3, 3].flatten())
            updated_rotation: Quaternion = Quaternion(matrix=dst[:3, :3])
            output_object = deepcopy(object)
            output_object.state.position = updated_position
            output_object.state.orientation = updated_rotation
            output_object.frame_id = "base_link"
            output_object_list.append(output_object)
        else:
            raise NotImplementedError(f"Unexpected frame_id: {object.frame_id}")
    return output_object_list
