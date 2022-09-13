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

import logging
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import LabelConverter
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.status import Visibility
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.utils.data_classes import Box
from pyquaternion.quaternion import Quaternion
from tqdm import tqdm

logger = getLogger(__name__)


class FrameGroundTruth:
    """
    Ground truth data per frame

    Attributes:
        self.unix_time (float): The unix time for the frame [us].
        self.frame_name (str): The file name for the frame.
        self.objects (List[DynamicObject]): Objects data.
        self.pointcloud (Optional[numpy.ndarray], optional):
                Pointcloud data. Defaults to None, but if you want to visualize dataset,
                you should load pointcloud data.
        self.transform_matrix (Optional[np.ndarray]): The numpy array to transform position.
    """

    def __init__(
        self,
        unix_time: int,
        frame_name: str,
        frame_id: str,
        objects: List[DynamicObject],
        ego2map: Optional[np.ndarray] = None,
        pointcloud: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            unix_time (int): The unix time for the frame [us]
            frame_name (str): The file name for the frame
            frame_id (str): The coord system which objects with respected to, base_link or map.
            objects (List[DynamicObject]): Objects data
            pointcloud (Optional[numpy.ndarray]):
                    Pointcloud data in (x, y, z, i).
                    Defaults to None, but if you want to visualize dataset,
                    you should load pointcloud data.
            ego2map (Optional[np.ndarray]): The array of 4x4 matrix.
                Transform position with respect to vehicle coord system to map one.
        """
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.frame_id: str = frame_id
        self.objects: List[DynamicObject] = objects
        self.pointcloud: Optional[np.ndarray] = pointcloud
        self.ego2map: Optional[np.ndarray] = ego2map


def load_all_datasets(
    dataset_paths: List[str],
    does_use_pointcloud: bool,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_id: str,
) -> List[FrameGroundTruth]:
    """
    Load tier4 datasets.
    Args:
        dataset_paths (List[str]): The list of root paths to dataset
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (EvaluationTask): The evaluation task
        label_converter (LabelConverter): Label convertor

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """
    logger.info(f"Start to load dataset {dataset_paths}")
    logger.info(
        f"config: does_use_pointcloud: {does_use_pointcloud}, evaluation_task: {evaluation_task}, frame_id: {frame_id}"
    )

    all_datasets: List[FrameGroundTruth] = []

    for dataset_path in dataset_paths:
        all_datasets += _load_dataset(
            dataset_path=dataset_path,
            does_use_pointcloud=does_use_pointcloud,
            evaluation_task=evaluation_task,
            label_converter=label_converter,
            frame_id=frame_id,
        )
    logger.info("Finish loading dataset\n" + _get_str_objects_number_info(label_converter))
    return all_datasets


def _load_dataset(
    dataset_path: str,
    does_use_pointcloud: bool,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_id: str,
) -> List[FrameGroundTruth]:
    """
    Load one tier4 dataset.
    Args:
        dataset_path (str): The root path to dataset
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (EvaluationTask): The evaluation task
        label_converter (LabelConverter): Label convertor

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """

    nusc: NuScenes = NuScenes(version="annotation", dataroot=dataset_path, verbose=False)
    helper: PredictHelper = PredictHelper(nusc)

    if len(nusc.visibility) == 0:
        logging.warn("visibility is not annotated")

    # Load category list
    category_list = []
    for category_dict in nusc.category:
        category_list.append(category_dict["name"])

    # Only keep samples from this split.
    # splits = create_splits_scenes()

    # Read out all sample_tokens in DB.
    sample_tokens = _get_sample_tokens(nusc.sample)

    dataset: List[FrameGroundTruth] = []
    for n, sample_token in enumerate(tqdm(sample_tokens)):
        frame = _sample_to_frame(
            nusc=nusc,
            helper=helper,
            sample_token=sample_token,
            does_use_pointcloud=does_use_pointcloud,
            evaluation_task=evaluation_task,
            label_converter=label_converter,
            frame_id=frame_id,
            frame_name=str(n),
        )
        dataset.append(frame)
    return dataset


def _get_str_objects_number_info(
    label_converter: LabelConverter,
) -> str:
    """[summary]
    Get str for the information of object label number
    Args:
        label_converter (LabelConverter): label convertor
    Returns:
        str: print str
    """
    str_: str = ""
    for label in label_converter.labels:
        str_ += f"{label.label} (-> {label.autoware_label}): {label.num} \n"
    return str_


class DatasetLoadingError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def _get_sample_tokens(nuscenes_sample: dict) -> List[Any]:
    """[summary]
    Get sample tokens

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
        # TODO impl for evaluation scene filter
        # scene_token = nusc.get("sample", sample_token)["scene_token"]
        # scene_record = nusc.get("scene", scene_token)
        # if scene_record["name"] in splits[eval_split]:
        #    sample_tokens.append(sample_token)
        sample_tokens.append(sample_token)

    return sample_tokens_all


def _sample_to_frame(
    nusc: NuScenes,
    helper: PredictHelper,
    sample_token: Any,
    does_use_pointcloud: bool,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_id: str,
    frame_name: str,
) -> FrameGroundTruth:
    """[summary]
    Convert Nuscenes sample to FrameGroundTruth

    Args:
        nusc (NuScenes): Nuscenes instance
        helper (PredictHelper): PredictHelper instance
        sample_token (Any): Nuscenes sample token
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (EvaluationTask): The evaluation task
        label_converter (LabelConverter): Label convertor
        frame_name (str): Name of frame, number of frame is used.

    Raises:
        NotImplementedError:

    Returns:
        FrameGroundTruth: Ground truth per frame
    """
    sample = nusc.get("sample", sample_token)

    # frame information
    unix_time_ = sample["timestamp"]
    if "LIDAR_TOP" in sample["data"]:
        lidar_path_token = sample["data"]["LIDAR_TOP"]
    elif "LIDAR_CONCAT" in sample["data"]:
        lidar_path_token = sample["data"]["LIDAR_CONCAT"]
    else:
        raise ValueError("lidar data isn't found")
    frame_data = nusc.get("sample_data", lidar_path_token)

    lidar_path, object_boxes, ego2map = _get_sample_boxes(nusc, frame_data, frame_id)

    # pointcloud
    if does_use_pointcloud:
        assert lidar_path.endswith(".bin"), f"Error: Unsupported filetype {lidar_path}"
        pointcloud_: np.ndarray = np.fromfile(lidar_path, dtype=np.float32)
        pointcloud_ = pointcloud_.reshape(-1, 5)[:, :4]
    else:
        pointcloud_ = None

    objects_: List[DynamicObject] = []

    for object_box in object_boxes:
        sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
        instance_token_: str = sample_annotation_["instance_token"]

        if len(nusc.visibility) == 0:
            visibility = None
        else:
            visibility_token: str = sample_annotation_["visibility_token"]
            visibility_info: Dict[str, Any] = nusc.get("visibility", visibility_token)
            visibility: Visibility = Visibility.from_value(visibility_info["level"])

        object_: DynamicObject = _convert_nuscenes_box_to_dynamic_object(
            nusc=nusc,
            helper=helper,
            frame_id=frame_id,
            object_box=object_box,
            unix_time=unix_time_,
            evaluation_task=evaluation_task,
            label_converter=label_converter,
            instance_token=instance_token_,
            sample_token=sample_token,
            visibility=visibility,
        )
        objects_.append(object_)

    frame = FrameGroundTruth(
        unix_time=unix_time_,
        frame_name=frame_name,
        frame_id=frame_id,
        objects=objects_,
        pointcloud=pointcloud_,
        ego2map=ego2map,
    )
    return frame


def _convert_nuscenes_box_to_dynamic_object(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: str,
    object_box: Box,
    unix_time: int,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    instance_token: str,
    sample_token: str,
    visibility: Optional[Visibility] = None,
    seconds: float = 3.0,
) -> DynamicObject:
    """[summary]
    Convert nuscenes object bounding box to dynamic object

    Args:
        nusc (NuScenes): NuScenes instance
        helper (PredictHelper): PredictHelper instance
        object_box (Box): Annotation data from nuscenes dataset defined by Box
        unix_time (int): The unix time [us]
        evaluation_task (EvaluationTask): Evaluation task
        label_converter (LabelConverter): LabelConverter
        instance_token (str): Instance token
        sample_token (str): Sample token, used to get past/future record
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
        seconds (float): Seconds to be referenced past/future record

    Returns:
        DynamicObject: Converted dynamic object class
    """
    position_: Tuple[float, float, float] = tuple(object_box.center.tolist())  # type: ignore
    orientation_: Quaternion = object_box.orientation
    size_: Tuple[float, float, float] = tuple(object_box.wlh.tolist())  # type: ignore
    semantic_score_: float = 1.0
    autoware_label_: AutowareLabel = label_converter.convert_label(
        label=object_box.name,
        count_label_number=True,
    )

    sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
    pointcloud_num_: int = sample_annotation_["num_lidar_pts"]
    velocity_: Tuple[float, float, float] = tuple(
        nusc.box_velocity(sample_annotation_["token"]).tolist()
    )

    if evaluation_task == EvaluationTask.TRACKING:
        (
            tracked_positions,
            tracked_orientations,
            tracked_sizes,
            tracked_velocities,
        ) = _get_tracking_data(
            nusc=nusc,
            helper=helper,
            frame_id=frame_id,
            instance_token=instance_token,
            sample_token=sample_token,
            seconds=seconds,
        )
    else:
        tracked_positions = None
        tracked_orientations = None
        tracked_sizes = None
        tracked_velocities = None

    if evaluation_task == EvaluationTask.PREDICTION:
        pass

    dynamic_object = DynamicObject(
        unix_time=unix_time,
        position=position_,
        orientation=orientation_,
        size=size_,
        velocity=velocity_,
        semantic_score=semantic_score_,
        semantic_label=autoware_label_,
        pointcloud_num=pointcloud_num_,
        uuid=instance_token,
        tracked_positions=tracked_positions,
        tracked_orientations=tracked_orientations,
        tracked_sizes=tracked_sizes,
        tracked_twists=tracked_velocities,
        visibility=visibility,
    )
    return dynamic_object


def _get_sample_boxes(
    nusc: NuScenes,
    frame_data: Dict[str, Any],
    frame_id: str,
    use_sensor_frame: bool = True,
) -> Tuple[str, List[Box], np.ndarray]:
    """[summary]
    Get bbox from frame data.

    Args:
        nusc (NuScenes): NuScenes object.
        frame_data (Dict[str, Any]):
        frame_id (str): base_link or map.

    Returns:
        lidar_path (str)
        object_boxes (List[Box])
        ego2map (np.ndarray)
        use_sensor_frame (bool): The fla

    Raises:
        ValueError: If got unexpected frame_id except of base_link or map.
    """
    lidar_path: str
    object_boxes: List[Box]
    if frame_id == "base_link":
        # Get boxes moved to ego vehicle coord system.
        lidar_path, object_boxes, _ = nusc.get_sample_data(frame_data["token"])
    elif frame_id == "map":
        # Get boxes map based coord system.
        lidar_path = nusc.get_sample_data_path(frame_data["token"])
        object_boxes = nusc.get_boxes(frame_data["token"])
    else:
        raise ValueError(f"Expected frame_id base_link or map, but got {frame_id}")

    # Get a sensor2map transform matrix
    vehicle2map = np.eye(4)
    vehicle_pose = nusc.get("ego_pose", frame_data["ego_pose_token"])
    vehicle2map[:3, :3] = Quaternion(vehicle_pose["rotation"]).rotation_matrix
    vehicle2map[:3, 3] = vehicle_pose["translation"]

    if use_sensor_frame:
        sensor2vehicle = np.eye(4)
        sensor_pose = nusc.get("calibrated_sensor", frame_data["calibrated_sensor_token"])
        sensor2vehicle[:3, :3] = Quaternion(sensor_pose["rotation"]).rotation_matrix
        sensor2vehicle[:3, 3] = sensor_pose["translation"]
        ego2map: np.ndarray = vehicle2map.dot(sensor2vehicle)
    else:
        ego2map: np.ndarray = vehicle2map

    return lidar_path, object_boxes, ego2map


def _get_tracking_data(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: str,
    instance_token: str,
    sample_token: str,
    seconds: float,
) -> Tuple[List[Tuple[float, float, float]], List[Quaternion], List[Tuple[float, float, float]]]:
    """Get tracking data with PredictHelper.get_past_for_agent()

    Args:
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        frame_id (str): The frame_id base_link or map.
        instance_token (str): The unique token to access to instance.
        sample_token (str): The unique Token to access to sample.
        seconds (float): Seconds to be referenced.[s]

    Returns:
        past_positions (List[Tuple[float, float, float]])
        past_orientations (List[Quaternion])
        past_sizes (List[Tuple[float, float, float]]])
        past_velocities (List[Tuple[float, float]])
    """
    if frame_id == "base_link":
        in_agent_frame: bool = True
    elif frame_id == "map":
        in_agent_frame: bool = False
    else:
        raise ValueError(f"Unexpected frame_id: {frame_id}")

    past_records_: List[Dict[str, Any]] = helper.get_past_for_agent(
        instance_token=instance_token,
        sample_token=sample_token,
        seconds=seconds,
        in_agent_frame=in_agent_frame,
        just_xy=False,
    )
    past_positions: List[Tuple[float, float, float]] = []
    past_orientations: List[Quaternion] = []
    past_sizes: List[Tuple[float, float, float]] = []
    past_velocities: List[Tuple[float, float, float]] = []
    for record_ in past_records_:
        past_positions.append(tuple(record_["translation"]))
        past_orientations.append(Quaternion(record_["rotation"]))
        past_sizes.append(record_["size"])
        past_velocities.append(nusc.box_velocity(record_["token"]))

    return past_positions, past_orientations, past_sizes, past_velocities


def _get_prediction_data(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: str,
    instance_token: str,
    sample_token: str,
    seconds: str,
):
    """Get prediction data with PredictHelper.get_future_for_agent()

    Args:
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        instance_token (str): The unique token to access to instance.
        sample_token (str): The unique token to access to sample.
        seconds (float): Seconds to be referenced.[s]

    Returns:
        future_positions (List[Tuple[float, float, float]])
        future_orientations (List[Tuple[float, float, float]])
        future_sizes (List[Tuple[float, float, float]])
        future_velocities (List[Tuple[float, float, float]])
    """
    pass


def get_now_frame(
    ground_truth_frames: List[FrameGroundTruth],
    unix_time: int,
    threshold_min_time: int,
) -> Optional[FrameGroundTruth]:
    """
    Select the ground truth frame whose unix time is most close to args unix time from dataset.

    Args:
        ground_truth_frames (List[FrameGroundTruth]): datasets
        unix_time (int): Unix time [us]
        threshold_min_time (int): Min time for unix time difference [us].

    Returns:
        Optional[FrameGroundTruth]:
                The ground truth frame whose unix time is most close to args unix time
                from dataset.
                If the difference time between unix time parameter and the most close time
                ground truth frame is larger than threshold_min_time, return None.
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
    if min_time > threshold_min_time:
        logger.info(
            f"Now frame is {ground_truth_now_frame.unix_time} and time difference \
                 is {min_time / 1000} ms > {threshold_min_time / 1000} ms"
        )
        return None
    else:
        return ground_truth_now_frame
