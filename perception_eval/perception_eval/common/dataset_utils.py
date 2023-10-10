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

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from PIL import Image
from pyquaternion.quaternion import Quaternion

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import Label, LabelConverter, LabelType, TrafficLightLabel
from perception_eval.common.object import DynamicObject
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID, Visibility
from perception_eval.common.shape import Shape, ShapeType

from . import dataset

if TYPE_CHECKING:
    from nuimages import NuImages
    from nuscenes.nuscenes import NuScenes
    from nuscenes.prediction.helper import PredictHelper
    from nuscenes.utils.data_classes import Box

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
) -> dataset.FrameGroundTruth:
    """Load FrameGroundTruth instance from sample record.

    Args:
    ----
        nusc (NuScenes): Nuscenes instance.
        helper (PredictHelper): PredictHelper instance.
        sample_token (Any): Nuscenes sample token.
        evaluation_tasks (EvaluationTask): The evaluation task.
        label_converter (LabelConverter): LabelConvertor instance.
        frame_id (FrameID): FrameID instance.
        frame_name (str): Name of frame, number of frame is used.
        load_raw_data (bool): Whether load pointcloud/image data.

    Raises:
    ------
        ValueError: When both `LIDAR_TOP` or `LIDAR_CONCAT`  are not included in data.

    Returns:
    -------
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
        msg = "lidar data isn't found"
        raise ValueError(msg)
    frame_data = nusc.get("sample_data", lidar_path_token)

    lidar_path, object_boxes, ego2map = _get_sample_boxes(nusc, frame_data, frame_id)

    # pointcloud
    raw_data: dict[str, np.ndarray] | None = {} if load_raw_data else None
    if load_raw_data:
        assert lidar_path.endswith(".bin"), f"Error: Unsupported filetype {lidar_path}"
        pointcloud: np.ndarray = np.fromfile(lidar_path, dtype=np.float32)
        # The Other modalities would be used, (e.g. radar)
        raw_data["lidar"] = pointcloud.reshape(-1, 5)[:, :4]

    objects_: list[DynamicObject] = []

    for object_box in object_boxes:
        sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
        instance_token_: str = sample_annotation_["instance_token"]

        if len(nusc.visibility) == 0:
            visibility = None
        else:
            visibility_token: str = sample_annotation_["visibility_token"]
            visibility_info: dict[str, Any] = nusc.get("visibility", visibility_token)
            visibility: Visibility = Visibility.from_value(visibility_info["level"])

        attribute_tokens: list[str] = sample_annotation_["attribute_tokens"]
        attributes: list[str] = [nusc.get("attribute", token)["name"] for token in attribute_tokens]
        semantic_label = label_converter.convert_label(object_box.name, attributes)

        if evaluation_task.is_fp_validation() and semantic_label.is_fp() is False:
            msg = f"Unexpected GT label for {evaluation_task.value}, got {semantic_label.label}"
            raise ValueError(msg)

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
            visibility=visibility,
        )
        objects_.append(object_)

    return dataset.FrameGroundTruth(
        unix_time=unix_time_,
        frame_name=frame_name,
        objects=objects_,
        ego2map=ego2map,
        raw_data=raw_data,
    )


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
    visibility: Visibility | None = None,
    seconds: float = 3.0,
) -> DynamicObject:
    """Convert nuscenes object bounding box to dynamic object.

    Args:
    ----
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        object_box (Box): Annotation data from nuscenes dataset defined by Box.
        unix_time (int): The unix time [us].
        evaluation_task (EvaluationTask): Evaluation task.
        semantic_label (Label): Label instance.
        instance_token (str): Instance token.
        sample_token (str): Sample token, used to get past/future record.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
        seconds (float): Seconds to be referenced past/future record. Defaults to 3.0.

    Returns:
    -------
        DynamicObject: Converted dynamic object class
    """
    position_: tuple[float, float, float] = tuple(object_box.center.astype(np.float64).tolist())
    orientation_: Quaternion = object_box.orientation
    shape_: Shape = Shape(
        shape_type=ShapeType.BOUNDING_BOX,
        size=tuple(object_box.wlh.astype(np.float64).tolist()),
    )
    semantic_score_: float = 1.0

    sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
    pointcloud_num_: int = sample_annotation_["num_lidar_pts"]
    velocity_: tuple[float, float, float] | None = _get_box_velocity(nusc, object_box.token)

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
            seconds=seconds,
        )
    else:
        tracked_positions = None
        tracked_orientations = None
        tracked_shapes = None
        tracked_velocities = None

    if evaluation_task == EvaluationTask.PREDICTION:
        pass

    return DynamicObject(
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
        visibility=visibility,
    )


def _get_sample_boxes(
    nusc: NuScenes,
    frame_data: dict[str, Any],
    frame_id: str,
    use_sensor_frame: bool = True,
) -> tuple[str, list[Box], np.ndarray]:
    """Get bbox from frame data.

    Args:
    ----
        nusc (NuScenes): NuScenes instance.
        frame_data (Dict[str, Any]): Set of frame record.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        use_sensor_frame (bool): Whether use sensor frame. Defaults to True.

    Returns:
    -------
        lidar_path (str): File path of lidar pointcloud.
        object_boxes (List[Box]): A list of boxes.
        ego2map (np.ndarray): 4x4 transformation matrix.

    Raises:
    ------
        ValueError: If got unexpected frame_id except of base_link or map.
    """
    lidar_path: str
    object_boxes: list[Box]
    if frame_id == FrameID.BASE_LINK:
        # Get boxes moved to ego vehicle coord system.
        lidar_path, object_boxes, _ = nusc.get_sample_data(frame_data["token"])
    elif frame_id == FrameID.MAP:
        # Get boxes map based coord system.
        lidar_path = nusc.get_sample_data_path(frame_data["token"])
        object_boxes = nusc.get_boxes(frame_data["token"])
    else:
        msg = f"Expected frame_id base_link or map, but got {frame_id}"
        raise ValueError(msg)

    # Get a sensor2map transform matrix
    vehicle2map = np.eye(4, dtype=np.float64)
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


def _get_box_velocity(
    nusc: NuScenes,
    sample_annotation_token: str,
    max_time_diff: float = 1.5,
) -> tuple[float, float, float] | None:
    """Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to None.

    Args:
    ----
        sample_annotation_token (str): Unique sample_annotation identifier.
        max_time_diff (float): Max allowed time diff between consecutive samples that are used to estimate velocities.

    Returns:
    -------
        Optional[Tuple[float, float, float]]: Velocity in x/y/z direction in m/s,
            which is with respect to object coordinates system.
    """
    current = nusc.get("sample_annotation", sample_annotation_token)
    has_prev = current["prev"] != ""
    has_next = current["next"] != ""

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return None

    first = nusc.get("sample_annotation", current["prev"]) if has_prev else current

    last = nusc.get("sample_annotation", current["next"]) if has_next else current

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
) -> tuple[list[tuple[float, float, float]], list[Quaternion], list[tuple[float, float, float]]]:
    """Get tracking data with PredictHelper.get_past_for_agent().

    Args:
    ----
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        instance_token (str): The unique token to access to instance.
        sample_token (str): The unique Token to access to sample.
        seconds (float): Seconds to reference past/future records.

    Returns:
    -------
        past_positions (List[Tuple[float, float, float]])
        past_orientations (List[Quaternion])
        past_shapes (List[Shape])
        past_velocities (List[Tuple[float, float]])
    """
    if frame_id == FrameID.BASE_LINK:
        in_agent_frame: bool = True
    elif frame_id == FrameID.MAP:
        in_agent_frame: bool = False
    else:
        msg = f"Unexpected frame_id: {frame_id}"
        raise ValueError(msg)

    past_records_: list[dict[str, Any]] = helper.get_past_for_agent(
        instance_token=instance_token,
        sample_token=sample_token,
        seconds=seconds,
        in_agent_frame=in_agent_frame,
        just_xy=False,
    )
    past_positions: list[tuple[float, float, float]] = []
    past_orientations: list[Quaternion] = []
    past_shapes: list[Shape] = []
    past_velocities: list[tuple[float, float, float]] = []
    for record_ in past_records_:
        translation: tuple[float, float, float] = (float(t) for t in record_["translation"])
        past_positions.append(translation)
        past_orientations.append(Quaternion(record_["rotation"]))
        size: tuple[float, float, float] = (float(s) for s in record_["size"])
        past_shapes.append(Shape(shape_type=ShapeType.BOUNDING_BOX, size=size))
        past_velocities.append(nusc.box_velocity(record_["token"]))

    return past_positions, past_orientations, past_shapes, past_velocities


def _get_prediction_data(
    nusc: NuScenes,
    helper: PredictHelper,
    frame_id: str,
    instance_token: str,
    sample_token: str,
    seconds: str,
):
    """Get prediction data with PredictHelper.get_future_for_agent().

    Args:
    ----
        nusc (NuScenes): NuScenes instance.
        helper (PredictHelper): PredictHelper instance.
        instance_token (str): The unique token to access to instance.
        sample_token (str): The unique token to access to sample.
        seconds (float): Seconds to be referenced.[s]

    Returns:
    -------
        future_positions (List[Tuple[float, float, float]])
        future_orientations (List[Tuple[float, float, float]])
        future_sizes (List[Tuple[float, float, float]])
        future_velocities (List[Tuple[float, float, float]])
    """


#################################
#           Dataset 2D          #
#################################


def _sample_to_frame_2d(
    nusc: NuScenes,
    nuim: NuImages,
    sample_token: FrameID | Sequence[FrameID],
    evaluation_task: EvaluationTask,
    label_converter: LabelConverter,
    frame_ids: list[FrameID],
    frame_name: str,
    load_raw_data: bool,
) -> dataset.FrameGroundTruth:
    """Returns FrameGroundTruth constructed with DynamicObject2D.

    Args:
    ----
        nusc (NuScenes): NuScenes instance.
        nuim (NuImages): NuImages instance.
        sample_token (str): Sample token.
        evaluation_task (EvaluationTask): 2D evaluation Task.
        label_converter (LabelConverter): LabelConverter instance.
        frame_ids (List[FrameID]): List of FrameID instances, where 2D objects are with respect, related to CAM_**.
        frame_name (str): Name of frame.
        load_raw_data (bool): The flag to load image data.

    Returns:
    -------
        frame (FrameGroundTruth): GT objects in one frame.
    """
    nusc_sample: dict[str, Any] = nusc.get("sample", sample_token)
    sample: dict[str, Any] = nuim.get("sample", sample_token)

    unix_time: int = sample["timestamp"]

    sample_data_tokens: list[str] = []
    frame_id_mapping: dict[str, FrameID] = {}
    for frame_id_ in frame_ids:
        camera_type: str = frame_id_.value.upper()
        sample_data_token = nusc_sample["data"][camera_type]
        sample_data_tokens.append(sample_data_token)
        frame_id_mapping[sample_data_token] = frame_id_

    raw_data: dict[str, np.ndarray] | None = {} if load_raw_data else None
    if load_raw_data:
        for sensor_name, sample_data_token in nusc_sample["data"].items():
            if (label_converter.label_type == TrafficLightLabel and "TRAFFIC_LIGHT" in sensor_name) or (
                "CAM" in sensor_name and "TRAFFIC_LIGHT" not in sensor_name
            ):
                img_path: str = nusc.get_sample_data_path(sample_data_token)
                raw_data[sensor_name.lower()] = np.array(Image.open(img_path), dtype=np.uint8)

    object_annotations: list[dict[str, Any]] = [
        ann for ann in nuim.object_ann if ann["sample_data_token"] in sample_data_tokens
    ]

    objects_: list[DynamicObject2D] = []
    for ann in object_annotations:
        if evaluation_task in (EvaluationTask.DETECTION2D, EvaluationTask.TRACKING2D):
            bbox: list[float] = ann["bbox"]
            x_offset: int = int(bbox[0])
            y_offset: int = int(bbox[1])
            width: int = int(bbox[2]) - x_offset
            height: int = int(bbox[3]) - y_offset
            roi = (x_offset, y_offset, width, height)
        else:
            roi = None

        category_info: dict[str, Any] = nuim.get("category", ann["category_token"])
        attribute_tokens: list[str] = ann["attribute_tokens"]
        attributes: list[str] = [nuim.get("attribute", token)["name"] for token in attribute_tokens]
        semantic_label: LabelType = label_converter.convert_label(category_info["name"], attributes)

        if label_converter.label_type == TrafficLightLabel:
            for instance_record in nusc.instance:
                if instance_record["token"] == ann["instance_token"]:
                    instance_name: str = instance_record["instance_name"]
                    uuid: str = instance_name.split(":")[-1]
        else:
            uuid: str = ann["instance_token"]

        visibility = None

        object_: DynamicObject2D = DynamicObject2D(
            unix_time=unix_time,
            frame_id=frame_id_mapping[ann["sample_data_token"]],
            semantic_score=1.0,
            semantic_label=semantic_label,
            roi=roi,
            uuid=uuid,
            visibility=visibility,
        )
        objects_.append(object_)

    return dataset.FrameGroundTruth(
        unix_time=unix_time,
        frame_name=frame_name,
        objects=objects_,
        raw_data=raw_data,
    )
