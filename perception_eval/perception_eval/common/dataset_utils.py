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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from nuimages import NuImages
import numpy as np
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

    Raises:
        ValueError: When both `LIDAR_TOP` or `LIDAR_CONCAT`  are not included in data.

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
    raw_data: Optional[Dict[str, np.ndarray]] = {} if load_raw_data else None
    if load_raw_data:
        assert lidar_path.endswith(".bin"), f"Error: Unsupported filetype {lidar_path}"
        pointcloud: np.ndarray = np.fromfile(lidar_path, dtype=np.float32)
        # The Other modalities would be used, (e.g. radar)
        raw_data["lidar"] = pointcloud.reshape(-1, 5)[:, :4]

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
            raise ValueError(
                f"Unexpected GT label for {evaluation_task.value}, got {semantic_label.label}"
            )

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

    frame = dataset.FrameGroundTruth(
        unix_time=unix_time_,
        frame_name=frame_name,
        objects=objects_,
        ego2map=ego2map,
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
    visibility: Optional[Visibility] = None,
    seconds: float = 3.0,
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
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
        seconds (float): Seconds to be referenced past/future record. Defaults to 3.0.

    Returns:
        DynamicObject: Converted dynamic object class
    """
    position_: Tuple[float, float, float] = tuple(object_box.center.tolist())  # type: ignore
    orientation_: Quaternion = object_box.orientation
    shape_: Shape = Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(object_box.wlh.tolist()))
    semantic_score_: float = 1.0

    sample_annotation_: dict = nusc.get("sample_annotation", object_box.token)
    pointcloud_num_: int = sample_annotation_["num_lidar_pts"]
    velocity_: Tuple[float, float, float] = tuple(
        nusc.box_velocity(sample_annotation_["token"]).tolist()
    )

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
        visibility=visibility,
    )
    return dynamic_object


def _get_sample_boxes(
    nusc: NuScenes,
    frame_data: Dict[str, Any],
    frame_id: str,
    use_sensor_frame: bool = True,
) -> Tuple[str, List[Box], np.ndarray]:
    """Get bbox from frame data.

    Args:
        nusc (NuScenes): NuScenes instance.
        frame_data (Dict[str, Any]): Set of frame record.
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        use_sensor_frame (bool): Whether use sensor frame. Defaults to True.

    Returns:
        lidar_path (str): File path of lidar pointcloud.
        object_boxes (List[Box]): A list of boxes.
        ego2map (np.ndarray): 4x4 transformation matrix.

    Raises:
        ValueError: If got unexpected frame_id except of base_link or map.
    """
    lidar_path: str
    object_boxes: List[Box]
    if frame_id == FrameID.BASE_LINK:
        # Get boxes moved to ego vehicle coord system.
        lidar_path, object_boxes, _ = nusc.get_sample_data(frame_data["token"])
    elif frame_id == FrameID.MAP:
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
        past_velocities (List[Tuple[float, float]])
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
        past_positions.append(tuple(record_["translation"]))
        past_orientations.append(Quaternion(record_["rotation"]))
        past_shapes.append(Shape(shape_type=ShapeType.BOUNDING_BOX, size=record_["size"]))
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
    for frame_id_ in frame_ids:
        camera_type: str = frame_id_.value.upper()
        sample_data_token = nusc_sample["data"][camera_type]
        sample_data_tokens.append(sample_data_token)
        frame_id_mapping[sample_data_token] = frame_id_

    raw_data: Optional[Dict[str, np.ndarray]] = {} if load_raw_data else None
    if load_raw_data:
        for sensor_name, sample_data_token in nusc_sample["data"].items():
            if (
                label_converter.label_type == TrafficLightLabel and "TRAFFIC_LIGHT" in sensor_name
            ) or ("CAM" in sensor_name and "TRAFFIC_LIGHT" not in sensor_name):
                img_path: str = nusc.get_sample_data_path(sample_data_token)
                raw_data[sensor_name.lower()] = np.array(Image.open(img_path), dtype=np.uint8)

    object_annotations: List[Dict[str, Any]] = [
        ann for ann in nuim.object_ann if ann["sample_data_token"] in sample_data_tokens
    ]

    objects_: List[DynamicObject2D] = []
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

    frame = dataset.FrameGroundTruth(
        unix_time=unix_time,
        frame_name=frame_name,
        objects=objects_,
        raw_data=raw_data,
    )

    return frame
