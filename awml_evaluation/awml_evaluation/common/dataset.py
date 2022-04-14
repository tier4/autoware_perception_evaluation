from logging import getLogger
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import LabelConverter
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.util.file import divide_file_path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion.quaternion import Quaternion
import tqdm

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
    """

    def __init__(
        self,
        unix_time: int,
        frame_name: str,
        objects: List[DynamicObject],
        pointcloud: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            unix_time (int): The unix time for the frame [us]
            frame_name (str): The file name for the frame
            objects (List[DynamicObject]): Objects data
            pointcloud (Optional[numpy.ndarray]):
                    Pointcloud data in (x, y, z, i).
                    Defaults to None, but if you want to visualize dataset,
                    you should load pointcloud data.
        """
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.objects: List[DynamicObject] = objects
        self.pointcloud: Optional[np.ndarray] = pointcloud


def load_all_datasets(
    dataset_paths: List[str],
    does_use_pointcloud: bool,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
    target_uuids: Optional[List[str]] = None,
) -> List[FrameGroundTruth]:
    """
    Load tier4 datasets.
    Args:
        dataset_paths (List[str]): The list of root paths to dataset
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (List[EvaluationTask]): The evaluation tasks
        label_converter (LabelConverter): Label convertor
        target_uuids: List[str]:
                The list of object instance tokens for all data in all frames.
                It should be specified in case of selecting specific objects.
                The order of the list is same as dataset paths. Defaults to None.
                NOTE: assuming dataset_paths is only one.

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """
    logger.info(f"Start to load dataset {dataset_paths}")
    logger.info(
        f"config: does_set_pointcloud {does_use_pointcloud}, evaluation_tasks {evaluation_tasks}"
    )

    all_datasets: List[FrameGroundTruth] = []

    for dataset_path in dataset_paths:
        all_datasets += _load_dataset(
            dataset_path=dataset_path,
            does_use_pointcloud=does_use_pointcloud,
            evaluation_tasks=evaluation_tasks,
            label_converter=label_converter,
            target_uuids=target_uuids,
        )
    logger.info("Finish loading dataset\n" + _get_str_objects_number_info(label_converter))
    return all_datasets


def _load_dataset(
    dataset_path: str,
    does_use_pointcloud: bool,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
    target_uuids: Optional[List[str]] = None,
) -> List[FrameGroundTruth]:
    """
    Load one tier4 dataset.
    Args:
        dataset_path (str): The root path to dataset
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (List[EvaluationTask]): The evaluation tasks
        label_converter (LabelConverter): Label convertor
        target_uuids: (Optional[List[str]]):
                The list of object instance tokens for the data will be loaded in all frames.
                It should be specified in case of selecting specific objects. Defaults to None

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """

    nusc: NuScenes = NuScenes(version="annotation", dataroot=dataset_path, verbose=False)

    # Load category list
    category_list = []
    for category_dict in nusc.category:
        category_list.append(category_dict["name"])

    # Only keep samples from this split.
    # splits = create_splits_scenes()

    # Read out all sample_tokens in DB.
    sample_tokens = _get_sample_tokens(nusc.sample)

    dataset: List[FrameGroundTruth] = []

    # If target_uuids is not specified, set it as empty list
    if target_uuids is None:
        target_uuids = []

    for sample_token in tqdm.tqdm(sample_tokens):
        frame = _sample_to_frame(
            nusc=nusc,
            sample_token=sample_token,
            does_use_pointcloud=does_use_pointcloud,
            evaluation_tasks=evaluation_tasks,
            label_converter=label_converter,
            target_uuids=target_uuids,
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
        DatasetLoadingError: Dataset loding error

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
    sample_token: Any,
    does_use_pointcloud: bool,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
    target_uuids: List[str],
) -> FrameGroundTruth:
    """[summary]
    Convert Nuscenes sample to FrameGroundTruth

    Args:
        nusc (NuScenes): Nuscenes instance
        sample_token (Any): Nuscenese sample token
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (List[EvaluationTask]): The evaluation tasks
        label_converter (LabelConverter): Label convertor
        target_uuids (Optional[List[str]]):
                The list of specific objects' instance tokens.
                It should be specified in case of selecting specific objects. Defaults to None

    Raises:
        NotImplementedError:

    Returns:
        FrameGroundTruth: Ground truth per frame
    """
    sample = nusc.get("sample", sample_token)

    # frame information
    unix_time_ = sample["timestamp"]
    lidar_path_token = sample["data"]["LIDAR_CONCAT"]
    # lidar_path = nusc.get_sample_data_path(lidar_path_token)
    frame_data = nusc.get("sample_data", lidar_path_token)

    lidar_path: str
    object_boxes: List[Box]
    lidar_path, object_boxes, _ = nusc.get_sample_data(frame_data["token"])

    # pointcloud
    if does_use_pointcloud:
        assert lidar_path.endswith(".bin"), f"Error: Unsupported filetype {lidar_path}"
        pointcloud_: np.ndarray = np.fromfile(lidar_path, dtype=np.float32)
        pointcloud_ = pointcloud_.reshape(-1, 5)[:, :4]
    else:
        pointcloud_ = None

    # frame name
    _, _, _, basename_without_ext, _ = divide_file_path(lidar_path)

    objects_: List[DynamicObject] = []

    for object_box in object_boxes:
        sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
        instance_token_: str = sample_annotation_["instance_token"]
        # Skip if target_uuids is not specified(=empty list)
        # or it is specified but object token is not in target_uuids
        if len(target_uuids) != 0 and instance_token_ not in target_uuids:
            continue

        pointcloud_num_: int = sample_annotation_["num_lidar_pts"]
        velocity_: Tuple[float, float, float] = tuple(
            nusc.box_velocity(sample_annotation_["token"]).tolist()
        )

        object_: DynamicObject = _convert_nuscenes_box_to_dynamic_object(
            object_box,
            unix_time_,
            evaluation_tasks,
            label_converter,
            instance_token_,
            pointcloud_num_,
            velocity_,
        )
        objects_.append(object_)

    frame = FrameGroundTruth(
        unix_time=unix_time_,
        frame_name=basename_without_ext,
        objects=objects_,
        pointcloud=pointcloud_,
    )
    return frame


def _convert_nuscenes_box_to_dynamic_object(
    object_box: Box,
    unix_time: int,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
    instance_token: str,
    pointcloud_num: int,
    velocity: Tuple[float, float, float],
) -> DynamicObject:
    """[summary]
    Convert nuscenes object bounding box to dynamic object

    Args:
        object_box (Box): Annotation data from nuscenes dataset defined by Box
        unix_time (int): The unix time [us]
        evaluation_tasks (List[EvaluationTask]): Evaluation task
        label_converter (LabelConverter): LabelConverter
        instance_token (str): Instance token
        pointcloud_num (int): The number of pointcloud in object box
        velocity (Tuple[float, float, float]): The Veclocity of object

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

    # tracking data
    if EvaluationTask.TRACKING in evaluation_tasks:
        raise NotImplementedError()

    # prediction data
    if EvaluationTask.PREDICTION in evaluation_tasks:
        raise NotImplementedError()

    dynamic_object = DynamicObject(
        unix_time=unix_time,
        position=position_,
        orientation=orientation_,
        size=size_,
        velocity=velocity,
        semantic_score=semantic_score_,
        semantic_label=autoware_label_,
        pointcloud_num=pointcloud_num,
        uuid=instance_token,
    )
    return dynamic_object


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
