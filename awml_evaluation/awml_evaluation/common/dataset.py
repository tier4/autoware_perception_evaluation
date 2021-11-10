from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion.quaternion import Quaternion
import tqdm

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import LabelConverter
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.util.file import divide_file_path

logger = getLogger(__name__)


class FrameGroundTruth:
    """
    Ground truth data for each frame.

    Attributes:
        self.unix_time (float): The unix time for the frame [us]
        self.frame_name (str): The file name for the frame
        self.objects (List[DynamicObject]): Objects data
        self.pointcloud (Optional[List[Tuple[float, float, float, float]]], optional):
                Pointcloud data. Defaults to None, but if you want to visualize dataset,
                you should load pointcloud data
    """

    def __init__(
        self,
        unix_time: float,
        frame_name: str,
        objects: List[DynamicObject],
        pointcloud: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        """[summary]

        Args:
            unix_time (float): The unix time for the frame [us]
            frame_name (str): The file name for the frame
            objects (List[DynamicObject]): Objects data
            pointcloud (Optional[List[Tuple[float, float, float, float]]], optional):
                    Pointcloud data. Defaults to None, but if you want to visualize dataset,
                    you should load pointcloud data
        """
        self.unix_time: float = unix_time
        self.frame_name: str = frame_name
        self.objects: List[DynamicObject] = objects
        # [[x, y, z, i]]
        self.pointcloud: Optional[List[Tuple[float, float, float, float]]] = pointcloud


class DatasetNoSampleError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


def _check_sample_error(sample_num: int) -> None:
    if sample_num < 1:
        raise DatasetNoSampleError("Error: Database has no samples!")


def load_datasets(
    dataset_path: str,
    does_use_pointcloud: bool,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
) -> List[FrameGroundTruth]:
    """
    Args:
        dataset_path (str): The root path to dataset
        does_use_pointcloud (bool): The flag of setting pointcloud
        evaluation_tasks (List[EvaluationTask]): The evaluation tasks
        label_converter (LabelConverter): Label convertor

    Reference
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """

    logger.info(f"Start to load dataset {dataset_path}")
    logger.info(
        f"config: does_set_pointcloud {does_use_pointcloud}, evaluation_tasks {evaluation_tasks}"
    )

    nusc: NuScenes = NuScenes(version="annotation", dataroot=dataset_path, verbose=False)

    # Load category list
    category_list = []
    for category_dict in nusc.category:
        category_list.append(category_dict["name"])

    # Only keep samples from this split.
    # splits = create_splits_scenes()

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s["token"] for s in nusc.sample]
    _check_sample_error(len(sample_tokens_all))

    sample_tokens = []

    for sample_token in sample_tokens_all:
        # TODO impl for evaluation scene filter
        # scene_token = nusc.get("sample", sample_token)["scene_token"]
        # scene_record = nusc.get("scene", scene_token)
        # if scene_record["name"] in splits[eval_split]:
        #    sample_tokens.append(sample_token)
        sample_tokens.append(sample_token)

    datasets: List[FrameGroundTruth] = []

    # for tracking data
    # tracking_id_set = set()

    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nusc.get("sample", sample_token)

        # frame information
        unix_time_ = sample["timestamp"]
        lidar_path_token = sample["data"]["LIDAR_TOP"]
        # lidar_path = nusc.get_sample_data_path(lidar_path_token)
        frame_data = nusc.get("sample_data", lidar_path_token)

        lidar_path: str = ""
        object_annotations: List[Box] = []
        lidar_path, object_annotations, _ = nusc.get_sample_data(frame_data["token"])

        # pointcloud
        pointcloud_ = None
        if does_use_pointcloud:
            # load from lidar path
            raise NotImplementedError()

        # frame name
        _, _, _, basename_without_ext, _ = divide_file_path(lidar_path)

        objects_: List[DynamicObject] = []

        for object_annotation in object_annotations:
            object_: DynamicObject = _convert_nuscenes_annotation_to_dynamic_object(
                object_annotation,
                unix_time_,
                evaluation_tasks,
                label_converter,
            )
            objects_.append(object_)

        frame = FrameGroundTruth(
            unix_time=unix_time_,
            frame_name=basename_without_ext,
            objects=objects_,
            pointcloud=pointcloud_,
        )
        datasets.append(frame)

    logger.info("Finish loading dataset")
    return datasets


def _convert_nuscenes_annotation_to_dynamic_object(
    object_annotation: Box,
    unix_time: int,
    evaluation_tasks: List[EvaluationTask],
    label_converter: LabelConverter,
) -> DynamicObject:
    """[summary]
    Convert from nuscenes object annotation to dynamic object

    Args:
        object_annotation (Box): Annotation data from nuscenes dataset
        unix_time (int): The unix time [us]
        evaluation_tasks (List[EvaluationTask]): Evaluation task
        label_converter (LabelConverter): LabelConverter

    Returns:
        DynamicObject: Converted dynamic object class
    """

    position_: Tuple[float, float, float] = tuple(object_annotation.center.tolist())
    orientation_: Quaternion = object_annotation.orientation
    size_: Tuple[float, float, float] = tuple(object_annotation.wlh.tolist())
    semantic_score_: float = 1.0
    autoware_label_: AutowareLabel = label_converter.convert_label(
        label=object_annotation.name,
        count_label_number=True,
    )
    # TODO impl for velocity
    velocity_ = tuple(object_annotation.velocity.tolist())
    # TODO impl for pointcloud_num

    # prediction data
    if EvaluationTask.PREDICTION in evaluation_tasks:
        # TODO implement
        raise NotImplementedError()

    dynamic_object = DynamicObject(
        unix_time=unix_time,
        position=position_,
        orientation=orientation_,
        size=size_,
        velocity=velocity_,
        semantic_score=semantic_score_,
        semantic_label=autoware_label_,
    )
    return dynamic_object


def get_now_frame(ground_truth_frames: List[FrameGroundTruth], unix_time: int) -> FrameGroundTruth:
    """
    Select the ground truth frame whose unix time is most close to args unix time from dataset.

    Args:
        ground_truth_frames (List[FrameGroundTruth]): dataset
        unix_time (int): unix time

    Returns:
        FrameGroundTruth: The ground truth frame
    """
    ground_truth_now_frame: FrameGroundTruth = ground_truth_frames[0]
    min_time: int = abs(unix_time - ground_truth_now_frame.unix_time)

    for ground_truth_frame in ground_truth_frames:
        diff_time = abs(unix_time - ground_truth_frame.unix_time)
        if diff_time < min_time:
            ground_truth_now_frame = ground_truth_frame
            min_time = diff_time
    if min_time > 75 * 1000:
        # min_time [us] > 75ms * 1000
        logger.warning(
            f"now frame is {ground_truth_now_frame.unix_time} and time diffrence \
                 is {min_time} > 75ms"
        )
    return ground_truth_now_frame
