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

from __future__ import annotations

import os.path as osp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from nuimages import NuImages
import numpy as np
from numpy.typing import NDArray
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.utils.data_classes import Box
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import Label
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.schema import Visibility
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import HomogeneousMatrix
from PIL import Image
from pyquaternion.quaternion import Quaternion

from . import dataset

#################################
#           Dataset 3D          #
#################################


def _sample_to_frame(
    nusc: NuScenes,
    helper: PredictHelper,
    sample_token: Any,
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_id: FrameID,
    frame_name: str,
    load_raw_data: bool,
    path_seconds: float,
) -> dataset.FrameGroundTruth:
    """Load FrameGroundTruth instance from sample record.

    Args:
        nusc (NuScenes): Nuscenes instance.
        helper (PredictHelper): PredictHelper instance.
        sample_token (Any): Nuscenes sample token.
        evaluation_tasks (EvaluationTask): The evaluation task.
        label_converter (LabelConverter): LabelConvertor instance.
        frame_id (FrameID): FrameID instance.
        frame_name (str): Name of frame, number of frame is used.
        load_raw_data (bool): Whether load pointcloud/image data.
        path_seconds (float): Seconds to be referenced past/future record.

    Raises:
        ValueError: When both `LIDAR_TOP` or `LIDAR_CONCAT`  are not included in data.

    Returns:
        FrameGroundTruth: Ground truth per frame
    """
    sample = nusc.get("sample", sample_token)

    # frame information
    unix_time_ = sample["timestamp"]
    if "LIDAR_TOP" in sample["data"]:
        sample_data_token = sample["data"]["LIDAR_TOP"]
    elif "LIDAR_CONCAT" in sample["data"]:
        sample_data_token = sample["data"]["LIDAR_CONCAT"]
    else:
        raise ValueError("lidar data isn't found")

    object_boxes = _get_sample_boxes(nusc, sample_data_token, frame_id)
    transforms = _get_transforms(nusc, sample_data_token)
    raw_data = _load_raw_data(nusc, sample_token) if load_raw_data else None

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

        attribute_tokens: List[str] = sample_annotation_["attribute_tokens"]
        attributes: List[str] = [nusc.get("attribute", token)["name"] for token in attribute_tokens]
        semantic_label = label_converter.convert_label(object_box.name, attributes)

        if evaluation_task.is_fp_validation() and semantic_label.is_fp() is False:
            raise ValueError(f"Unexpected GT label for {evaluation_task.value}, got {semantic_label.label}")

        object_: DynamicObject = _convert_nuscenes_box_to_dynamic_object(
            nusc=nusc,
            helper=helper,
            frame_id=frame_id,
            object_box=object_box,
            unix_time=unix_time_,
            evaluation_task=evaluation_task,
            semantic_label=semantic_label,
            instance_token=instance_token_,
            sample_token=sample_token,
            path_seconds=path_seconds,
            visibility=visibility,
        )
        objects_.append(object_)

    frame = dataset.FrameGroundTruth(
        unix_time=unix_time_,
        frame_name=frame_name,
        objects=objects_,
        transforms=transforms,
        raw_data=raw_data,
    )
    return frame


def _convert_nuscenes_box_to_dynamic_object(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: FrameID,
    object_box: Box,
    unix_time: int,
    evaluation_task: EvaluationTask,
    semantic_label: Label,
    instance_token: str,
    sample_token: str,
    path_seconds: float,
    visibility: Optional[Visibility] = None,
) -> DynamicObject:
    """Convert nuscenes object bounding box to dynamic object.

    Args:
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        object_box (Box): Annotation data from nuscenes dataset defined by Box.
        unix_time (int): The unix time [us].
        evaluation_task (EvaluationTask): Evaluation task.
        semantic_label (Label): Label instance.
        instance_token (str): Instance token.
        sample_token (str): Sample token, used to get past/future record.
        path_seconds (float): Seconds to be referenced past/future record.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.

    Returns:
        DynamicObject: Converted dynamic object class
    """
    position_: Tuple[float, float, float] = tuple(object_box.center.astype(np.float64).tolist())
    orientation_: Quaternion = object_box.orientation
    shape_: Shape = Shape(
        shape_type=ShapeType.BOUNDING_BOX,
        size=tuple(object_box.wlh.astype(np.float64).tolist()),
    )
    semantic_score_: float = 1.0

    sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
    pointcloud_num_: int = sample_annotation_["num_lidar_pts"]
    velocity_: Optional[Tuple[float, float, float]] = _get_box_velocity(nusc, object_box.token)

    if evaluation_task == EvaluationTask.TRACKING:
        (
            tracked_positions,
            tracked_orientations,
            tracked_shapes,
            tracked_velocities,
        ) = _get_tracking_data(
            nusc=nusc,
            helper=helper,
            frame_id=frame_id,
            instance_token=instance_token,
            sample_token=sample_token,
            seconds=path_seconds,
        )
    else:
        tracked_positions = None
        tracked_orientations = None
        tracked_shapes = None
        tracked_velocities = None

    if evaluation_task == EvaluationTask.PREDICTION:
        (
            predicted_positions,
            predicted_orientations,
            predicted_shapes,
            predicted_twists,
        ) = _get_prediction_data(
            nusc=nusc,
            helper=helper,
            frame_id=frame_id,
            instance_token=instance_token,
            sample_token=sample_token,
            seconds=path_seconds,
        )
        predicted_positions = [predicted_positions]
        predicted_orientations = [predicted_orientations]
        predicted_shapes = [predicted_shapes]
        predicted_twists = [predicted_twists]
        predicted_scores = [1.0]
    else:
        predicted_positions = None
        predicted_orientations = None
        predicted_shapes = None
        predicted_twists = None
        predicted_scores = None

    dynamic_object = DynamicObject(
        unix_time=unix_time,
        frame_id=frame_id,
        position=position_,
        orientation=orientation_,
        shape=shape_,
        velocity=velocity_,
        semantic_score=semantic_score_,
        semantic_label=semantic_label,
        pointcloud_num=pointcloud_num_,
        uuid=instance_token,
        tracked_positions=tracked_positions,
        tracked_orientations=tracked_orientations,
        tracked_shapes=tracked_shapes,
        tracked_twists=tracked_velocities,
        predicted_positions=predicted_positions,
        predicted_orientations=predicted_orientations,
        predicted_shapes=predicted_shapes,
        predicted_twists=predicted_twists,
        predicted_scores=predicted_scores,
        visibility=visibility,
    )
    return dynamic_object


def _get_sample_boxes(nusc: NuScenes, sample_data_token: str, frame_id: FrameID) -> List[Box]:
    """Get the lidar raw data path, boxes.

    Args:
        nusc (NuScenes): `NuScenes` instance.
        sample_data_token (str): Sample data token.
        frame_id (FrameID): Frame ID where loaded boxes are with respect to.

    Raises:
        ValueError: Expecting `BASE_LINK` or `MAP` for the target `frame_id`.

    Returns:
        list[Box]: Boxes.
    """
    if frame_id == FrameID.BASE_LINK:
        # Get boxes moved to ego vehicle coord system.
        _, boxes, _ = nusc.get_sample_data(sample_data_token)
    elif frame_id == FrameID.MAP:
        # Get boxes map based coord system.
        boxes = nusc.get_boxes(sample_data_token)
    else:
        raise ValueError(f"Expected frame id is `BASE_LINK` or `MAP`, but got {frame_id}")

    return boxes


def _get_transforms(nusc: NuScenes, sample_data_token: str) -> List[HomogeneousMatrix]:
    """Load transform matrices.
    Additionally, for traffic light cameras, add transforms from BASE_LINK to TRAFFIC_LIGHT.

    Args:
        nusc (NuScenes): NuScenes instance.
        sample_data_token (str): Sample data token.

    Returns:
        List[HomogeneousMatrix]: List of matrices transforming position from sensor coordinate to map coordinate.
    """
    # Get a ego2map transform matrix
    sample_data = nusc.get("sample_data", sample_data_token)
    ego_record = nusc.get("ego_pose", sample_data["ego_pose_token"])
    ego_position = np.array(ego_record["translation"])
    ego_rotation = Quaternion(ego_record["rotation"])
    ego2map = HomogeneousMatrix(ego_position, ego_rotation, src=FrameID.BASE_LINK, dst=FrameID.MAP)

    matrices = [ego2map]
    tlr_avg_pos: List[NDArray] = []
    tlr_avg_quat: List[Quaternion] = []
    for cs_record in nusc.calibrated_sensor:
        sensor_position = cs_record["translation"]
        sensor_rotation = Quaternion(cs_record["rotation"])
        sensor_record = nusc.get("sensor", cs_record["sensor_token"])
        sensor_frame_id = FrameID.from_value(sensor_record["channel"])
        sensor2ego = HomogeneousMatrix(sensor_position, sensor_rotation, src=sensor_frame_id, dst=FrameID.BASE_LINK)
        sensor2map = ego2map.dot(sensor2ego)
        matrices.extend((sensor2ego, sensor2map))
        if "CAM_TRAFFIC_LIGHT" in sensor_frame_id.value.upper():
            tlr_avg_pos.append(sensor_position)
            tlr_avg_quat.append(sensor_rotation)

    # NOTE: Average positions and rotations are used for matrices of cameras related to TLR.
    if len(tlr_avg_pos) > 0 and len(tlr_avg_quat) > 0:
        tlr_cam_pos: NDArray = np.mean(tlr_avg_pos, axis=0)
        tlr_cam_rot: Quaternion = sum(tlr_avg_quat) / sum(tlr_avg_quat).norm
        tlr2ego = HomogeneousMatrix(tlr_cam_pos, tlr_cam_rot, src=FrameID.CAM_TRAFFIC_LIGHT, dst=FrameID.BASE_LINK)
        tlr2map = ego2map.dot(tlr2ego)
        matrices.extend((tlr2ego, tlr2map))

    return matrices


def _load_raw_data(nusc: NuScenes, sample_token: str) -> Dict[FrameID, NDArray]:
    """Load raw data for each sensor frame.

    Args:
        nusc (NuScenes): NuScenes instance.
        sample_token (str): Sample token.

    Returns:
        Dict[FrameID, NDArray]: Raw data at each sensor frame.
    """
    sample = nusc.get("sample", sample_token)
    output: Dict[FrameID, NDArray] = {}
    for sensor_name, sample_data_token in sample["data"].items():
        frame_id = FrameID.from_value(sensor_name)
        filepath: str = nusc.get_sample_data_path(sample_data_token)
        if osp.basename(filepath).endswith("bin"):
            raw_data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)[:, :4]
        else:
            raw_data = np.array(Image.open(filepath), dtype=np.uint8)
        output[frame_id] = raw_data
    return output


def _get_box_velocity(
    nusc: NuScenes,
    sample_annotation_token: str,
    max_time_diff: float = 1.5,
) -> Optional[Tuple[float, float, float]]:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to None.

    Args:
        sample_annotation_token (str): Unique sample_annotation identifier.
        max_time_diff (float): Max allowed time diff between consecutive samples that are used to estimate velocities.

    Returns:
        Optional[Tuple[float, float, float]]: Velocity in x/y/z direction in m/s,
            which is with respect to object coordinates system.
    """

    current = nusc.get("sample_annotation", sample_annotation_token)
    has_prev = current["prev"] != ""
    has_next = current["next"] != ""

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return None

    if has_prev:
        first = nusc.get("sample_annotation", current["prev"])
    else:
        first = current

    if has_next:
        last = nusc.get("sample_annotation", current["next"])
    else:
        last = current

    pos_last = np.array(last["translation"], dtype=np.float64)
    pos_first = np.array(first["translation"], dtype=np.float64)
    pos_diff = pos_last - pos_first

    object2map = np.eye(4, dtype=np.float64)
    object2map[:3, :3] = Quaternion(first["rotation"]).rotation_matrix
    object2map[3, :3] = first["translation"]

    pos_diff: np.ndarray = np.linalg.inv(object2map).dot((pos_diff[0], pos_diff[1], pos_diff[2], 1.0))[:3]

    time_last: float = 1e-6 * nusc.get("sample", last["sample_token"])["timestamp"]
    time_first: float = 1e-6 * nusc.get("sample", first["sample_token"])["timestamp"]
    time_diff: float = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    # If time_diff is too big, don't return an estimate.
    return tuple((pos_diff / time_diff).tolist()) if time_diff <= max_time_diff else None


def _get_tracking_data(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: FrameID,
    instance_token: str,
    sample_token: str,
    seconds: float,
) -> Tuple[List[Tuple[float, float, float]], List[Quaternion], List[Tuple[float, float, float]]]:
    """Get tracking data with PredictHelper.get_past_for_agent()

    Args:
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        instance_token (str): The unique token to access to instance.
        sample_token (str): The unique Token to access to sample.
        seconds (float): Seconds to reference past/future records.

    Returns:
        past_positions (List[Tuple[float, float, float]])
        past_orientations (List[Quaternion])
        past_shapes (List[Shape])
        past_velocities (List[Tuple[float, float, float]])
    """
    if frame_id == FrameID.BASE_LINK:
        in_agent_frame: bool = True
    elif frame_id == FrameID.MAP:
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
    past_shapes: List[Shape] = []
    past_velocities: List[Tuple[float, float, float]] = []
    for record_ in past_records_:
        translation: Tuple[float, float, float] = tuple(float(t) for t in record_["translation"])
        past_positions.append(translation)
        past_orientations.append(Quaternion(record_["rotation"]))
        size: Tuple[float, float, float] = tuple(float(s) for s in record_["size"])
        past_shapes.append(Shape(shape_type=ShapeType.BOUNDING_BOX, size=size))
        past_velocities.append(nusc.box_velocity(record_["token"]))

    return past_positions, past_orientations, past_shapes, past_velocities


def _get_prediction_data(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: str,
    instance_token: str,
    sample_token: str,
    seconds: float,
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
]:
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
        future_shapes (List[Shape])
        future_twists (List[Tuple[float, float, float]])
    """
    if frame_id == "base_link":
        in_agent_frame: bool = True
    elif frame_id == "map":
        in_agent_frame: bool = False
    else:
        raise ValueError(f"Unexpected frame_id: {frame_id}")

    future_records_: List[Dict[str, Any]] = helper.get_future_for_agent(
        instance_token=instance_token,
        sample_token=sample_token,
        seconds=seconds,
        in_agent_frame=in_agent_frame,
        just_xy=False,
    )
    future_positions: List[Tuple[float, float, float]] = []
    future_orientations: List[Quaternion] = []
    future_shapes: List[Shape] = []
    future_velocities: List[Tuple[float, float, float]] = []
    for record_ in future_records_:
        future_positions.append(tuple(record_["translation"]))
        future_orientations.append(Quaternion(record_["rotation"]))
        future_shapes.append(Shape(shape_type=ShapeType.BOUNDING_BOX, size=record_["size"]))
        future_velocities.append(nusc.box_velocity(record_["token"]))

    return future_positions, future_orientations, future_shapes, future_velocities


#################################
#           Dataset 2D          #
#################################


def _sample_to_frame_2d(
    nusc: NuScenes,
    nuim: NuImages,
    sample_token: Union[FrameID, Sequence[FrameID]],
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_ids: List[FrameID],
    frame_name: str,
    load_raw_data: bool,
) -> dataset.FrameGroundTruth:
    """Returns FrameGroundTruth constructed with DynamicObject2D.

    Args:
        nusc (NuScenes): NuScenes instance.
        nuim (NuImages): NuImages instance.
        sample_token (str): Sample token.
        evaluation_task (EvaluationTask): 2D evaluation Task.
        label_converter (LabelConverter): LabelConverter instance.
        frame_ids (List[FrameID]): List of FrameID instances, where 2D objects are with respect, related to CAM_**.
        frame_name (str): Name of frame.
        load_raw_data (bool): The flag to load image data.

    Returns:
        frame (FrameGroundTruth): GT objects in one frame.
    """
    nusc_sample: Dict[str, Any] = nusc.get("sample", sample_token)
    sample: Dict[str, Any] = nuim.get("sample", sample_token)

    unix_time: int = sample["timestamp"]

    sample_data_tokens: List[str] = []
    frame_id_mapping: Dict[str, FrameID] = {}
    transforms = None
    for frame_id_ in frame_ids:
        camera_type: str = frame_id_.value.upper()
        if nusc_sample["data"].get(camera_type) is None:
            continue
        sample_data_token = nusc_sample["data"][camera_type]
        sample_data_tokens.append(sample_data_token)
        frame_id_mapping[sample_data_token] = frame_id_

        sd_record = nusc.get("sample_data", sample_data_token)
        if sd_record["is_key_frame"]:
            transforms = _get_transforms(nusc, sample_data_token)

    if load_raw_data:
        raw_data = _load_raw_data(nusc, sample_token)
    else:
        raw_data = None

    object_annotations: List[Dict[str, Any]] = [
        ann for ann in nuim.object_ann if ann["sample_data_token"] in sample_data_tokens
    ]

    objects_: List[DynamicObject2D] = []
    # used to merge multiple traffic lights with same regulatory element ID
    uuids: List[str] = []
    for ann in object_annotations:
        if evaluation_task in (EvaluationTask.DETECTION2D, EvaluationTask.TRACKING2D):
            bbox: List[float] = ann["bbox"]
            x_offset: int = int(bbox[0])
            y_offset: int = int(bbox[1])
            width: int = int(bbox[2]) - x_offset
            height: int = int(bbox[3]) - y_offset
            roi = (x_offset, y_offset, width, height)
        else:
            roi = None

        category_info: Dict[str, Any] = nuim.get("category", ann["category_token"])
        attribute_tokens: List[str] = ann["attribute_tokens"]
        attributes: List[str] = [nuim.get("attribute", token)["name"] for token in attribute_tokens]
        semantic_label: LabelType = label_converter.convert_label(category_info["name"], attributes)

        if label_converter.label_type == TrafficLightLabel:
            # NOTE: Check whether Regulatory Element is used
            # in scene.json => description: "TLR, regulatory_element"
            for instance_record in nusc.instance:
                if instance_record["token"] == ann["instance_token"]:
                    instance_name: str = instance_record["instance_name"]
                    uuid: str = instance_name.split(":")[-1]
                    break
            uuids.append(uuid)
        else:
            uuid: str = ann["instance_token"]

        object_: DynamicObject2D = DynamicObject2D(
            unix_time=unix_time,
            frame_id=frame_id_mapping[ann["sample_data_token"]],
            semantic_score=1.0,
            semantic_label=semantic_label,
            roi=roi,
            uuid=uuid,
            visibility=None,
        )
        objects_.append(object_)

    if label_converter.label_type == TrafficLightLabel and evaluation_task == EvaluationTask.CLASSIFICATION2D:
        objects_ = _merge_duplicated_traffic_lights(unix_time, objects_, uuids)

    frame = dataset.FrameGroundTruth(
        unix_time=unix_time,
        frame_name=frame_name,
        objects=objects_,
        transforms=transforms,
        raw_data=raw_data,
    )

    return frame


def _merge_duplicated_traffic_lights(
    unix_time: int,
    objects: List[DynamicObject2D],
    uuids: List[str],
) -> List[DynamicObject2D]:
    """Merge traffic light objects which have same uuids and set its frame id as `FrameID.CAM_TRAFFIC_LIGHT`.

    Args:
        unix_time (int): Current unix timestamp.
        objects (List[DynamicObject2D]): List of traffic light objects.
            It can contain the multiple traffic lights with same uuid.
        uuids (List[str]): List of uuids.

    Returns:
        List[DynamicObject2D]: List of merged results.
    """
    uuids = set(uuids)
    ret_objects: List[DynamicObject2D] = []
    for uuid in uuids:
        candidates = [obj for obj in objects if obj.uuid == uuid]
        candidate_labels = [obj.semantic_label for obj in candidates]
        if all([label == candidate_labels[0] for label in candidate_labels]):
            # all unknown or not unknown
            semantic_label = candidate_labels[0]
        else:
            unique_labels = set([obj.semantic_label.label for obj in candidates])
            assert len(unique_labels) == 2, (
                "If the same regulatory element ID is assigned to multiple traffic lights, "
                f"it must annotated with only two labels: (unknown, another one). But got, {unique_labels}"
            )
            semantic_label = [label for label in candidate_labels if label.label != TrafficLightLabel.UNKNOWN][0]
            assert semantic_label.label != TrafficLightLabel.UNKNOWN
        merged_object = DynamicObject2D(
            unix_time=unix_time,
            frame_id=FrameID.CAM_TRAFFIC_LIGHT,
            semantic_score=1.0,
            semantic_label=semantic_label,
            roi=None,
            uuid=uuid,
            visibility=None,
        )
        ret_objects.append(merged_object)
    return ret_objects
