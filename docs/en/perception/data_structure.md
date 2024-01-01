# Data structure for Perception Evaluation

## Data structure per one Evaluation

- Catalog: A set of scenario
  - Example: turn right UseCase = (UC-0001, UC-0003, UC-0006)

```yaml
- Catalog
  - List[Scenario]
```

- Scenario: An unit of evaluation

  - There are two types of scenario, DataBase(DB) and UseCase(UC).

```yaml
- Scenario
  - List[Scene]
```

- Scene: An unit of one sequence of rosbag.
  - A set of data constructed by one rosbag.
  - A set of one rosbag and one .pcd file and some .jpeg files.

```yaml
- Scene
  - List[PerceptionFrameResult]
```

## Data structure per one Frame

### `<class> PerceptionFrameResult(...)`

For the details, see [perception_eval/result/perception/frame_result.py](../../../perception_eval/perception_eval/result/perception/frame_result.py)

- Initialization

  | Attributes           |                   type                    | Description                                     |
  | :------------------- | :---------------------------------------: | :---------------------------------------------- |
  | `frame_name`         |                   `str`                   | Name of frame                                   |
  | `unix_time`          |                   `int`                   | Unix time of frame                              |
  | `target_labels`      |             `List[LabelType]`             | List of name of target labels                   |
  | `object_results`     | `List[DynamicObjectWithPerceptionResult]` | List of pair of Estimation and ground truth(GT) |
  | `frame_ground_truth` |            `FrameGroundTruth`             | GT objects for one frame                        |
  | `metrics_score`      |              `MetricsScore`               | Score of metrics result                         |
  | `pass_fail_result`   |             `PassFailResult`              | Result of pass / fail                           |

- Methods

  | Methods            | Returns | Description                                                           |
  | :----------------- | :-----: | :-------------------------------------------------------------------- |
  | `evaluate_frame()` | `None`  | Execute evaluation of metrics and decision of pass/fail for one frame |

- metrics_score (MetricsScore): Score of metrics result
- pass_fail_result (PassFailResult): Result of usecase evaluation
  - tp_object_results (List[DynamicObjectWithPerceptionResult]): TP results in usecase evaluation
  - fp_object_results (List[DynamicObjectWithPerceptionResult]): FP results in usecase evaluation
  - fn_objects (List[ObjectType]): FN objects in usecase evaluation

```yaml
[2022-08-10 10:38:11,341] [INFO] [perception_lsim.py:258 <module>] Frame result example (frame_results[0]):
{'frame_ground_truth': {'ego2map': ' --- length of element 4 ---,',
                        'frame_name': '0',
                        'objects': ' --- length of element 89 ---,',
                        'pointcloud': None,
                        'unix_time': 1624164470849887},
 'frame_name': '0',
 'metrics_score': {'detection_config': {'center_distance_thresholds': ' --- length of element 2 ---,',
                                        'iou_3d_thresholds': [' --- length of element 4 ---,'],
                                        'iou_2d_thresholds': [' --- length of element 4 ---,'],
                                        'plane_distance_thresholds': ' --- length of element 2 ---,',
                                        'target_labels': ' --- length of element 4 ---,'},
                   'maps': ' --- length of element 6 ---,',
                   'prediction_config': None,
                   'prediction_scores': [],
                   'tracking_config': {'center_distance_thresholds': ' --- length of element 2 ---,',
                                       'iou_3d_thresholds': [' --- length of element 4 ---,'],
                                       'iou_2d_thresholds': [' --- length of element 4 ---,'],
                                       'plane_distance_thresholds': ' --- length of element 2 ---,',
                                       'target_labels': ' --- length of element 4 ---,'},
                   'tracking_scores': ' --- length of element 6 ---,'},
 'object_results': ' --- length of element 88 ---,',
 'pass_fail_result': {'critical_ground_truth_objects': ' --- length of element 23 ---,',
                      'critical_object_filter_config': {'confidence_threshold_list': None,
                                                        'filtering_params': {'confidence_threshold_list': None,
                                                                             'max_distance_list': None,
                                                                             'max_x_position_list': ' --- length of '
                                                                                                    'element 4 ---,',
                                                                             'max_y_position_list': ' --- length of '
                                                                                                    'element 4 ---,',
                                                                             'min_distance_list': None,
                                                                             'min_point_numbers': None,
                                                                             'target_labels': ' --- length of element '
                                                                                              '4 ---,',
                                                                             'target_uuids': None},
                                                        'max_distance_list': None,
                                                        'max_x_position_list': ' --- length of element 4 ---,',
                                                        'max_y_position_list': ' --- length of element 4 ---,',
                                                        'min_distance_list': None,
                                                        'min_point_numbers': None,
                                                        'target_labels': ' --- length of element 4 ---,',
                                                        'target_uuids': None},
                      'ego2map': ' --- length of element 4 ---,',
                      'fn_objects': ' --- length of element 17 ---,',
                      'fp_object_results': ' --- length of element 17 ---,',
                      'frame_pass_fail_config': {'confidence_threshold_list': None,
                                                 'matching_threshold_list': ' --- length of element 4 ---,',
                                                 'target_labels': ' --- length of element 4 ---,'},
                      'tp_object_results': ' --- length of element 22 ---,'},
 'target_labels': ' --- length of element 4 ---,',
 'unix_time': 1624164470849887}
```

## Data structure per one Object

### `<class> DynamicObjectWithPerceptionResult(...)`

Call `<func> get_object_results(...)` function to generate a set of matching pairs `List[DynamicObjectWithPerceptionResult]` from a set of Estimated objects `List[ObjectType]`and a set of GT objects `List[ObjectType]`.

For the details，see [perception_eval/result/perception/object_result.py](../../../perception_eval/perception_eval/result/perception/object_result.py)

```python
from perception_eval.evaluation.result.object_results import get_object_results

# REQUIRED:
#   estimated_objects: List[ObjectType]
#   ground_truth_objects: List[ObjectType]

object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(estimated_objects, ground_truth_objects)
```

- Initialization

  | Arguments             |          type          | Description |
  | :-------------------- | :--------------------: | :---------- |
  | `estimated_object`    |      `ObjectType`      | Estimation  |
  | `ground_truth_object` | `Optional[ObjectType]` | GT object   |

- Attributes

  | Attributes            |           type           | Description                                                    |
  | :-------------------- | :----------------------: | :------------------------------------------------------------- |
  | `estimated_object`    |       `ObjectType`       | Estimation                                                     |
  | `ground_truth_object` |  `Optional[ObjectType]`  | GT object                                                      |
  | `is_label_correct`    |          `bool`          | Whether the labels which estimation and GT object has are same |
  | `center_distance`     | `CenterDistanceMatching` | Distance of center between two objects                         |
  | `plane_distance`      | `PlaneDistanceMatching`  | Distance of the nearest plane between two objects              |
  | `iou_2d`              |     `IOU2DMatching`      | IOU score in 2-dimensions                                      |
  | `iou_3d`              |     `IOU3dMatching`      | IOU score in 3-dimensions                                      |

- Methods

  | Methods               |     Returns      | Description                          |
  | :-------------------- | :--------------: | :----------------------------------- |
  | `get_matching()`      | `MatchingMethod` | Returns the corresponding matching   |
  | `is_result_correct()` |      `bool`      | Returns the result if it is TP or FP |

```yaml
[2022-08-09 18:56:45,237] [INFO] [perception_lsim.py:208 <module>] Object result example (frame_results[0].object_results[0]):
{'center_distance': {'mode': 'MatchingMode.CENTERDISTANCE', 'value': 0.5051040904718623},
 'estimated_object': {'pointcloud_num': 113,
                      'predicted_confidence': None,
                      'predicted_path': None,
                      'semantic_label': 'AutowareLabel.PEDESTRIAN',
                      'semantic_score': 0.6965261662052492,
                      'state': {'orientation': {'q': [0.9394992040041219, 0.016502364866547787, -0.0022243908564357263,
                                                      0.34214612333731087]},
                                'position': [12.830938811123243, -28.461605817964994, 0.4005466639700647],
                                'size': [0.758, 1.138, 1.861],
                                'velocity': [-0.8304898624100426, 1.0906433133424918, -0.0700413136982921]},
                      'tracked_path': None,
                      'unix_time': 1624164470849887,
                      'uuid': 'c28556c19064ad491ff1dc438a38a3a7'},
 'ground_truth_object': {'pointcloud_num': 128,
                         'predicted_confidence': None,
                         'predicted_path': None,
                         'semantic_label': 'AutowareLabel.PEDESTRIAN',
                         'semantic_score': 1.0,
                         'state': {'orientation': {'q': [0.9947797942640241, 0.012007913693226124,
                                                         0.0001662701515344235, 0.10133579469762255]},
                                   'position': [12.629186624221628, -28.881230126682482, 0.2047364578387355],
                                   'size': [0.822, 1.199, 1.822],
                                   'velocity': [-0.570336411550396, 1.520897097467723, -0.48028329393123725]},
                         'tracked_path': None,
                         'unix_time': 1624164470849887,
                         'uuid': '912ae043cbc5a6ad4950f5ac0e94778e'},
 'iou_3d': {'mode': 'MatchingMode.IOU3D', 'value': 0.24986054835978477},
 'iou_2d': {'mode': 'MatchingMode.IOU2D', 'value': 0.2878950915821158},
 'is_label_correct': True,
 'plane_distance': {'estimated_nn_plane': [[13.02303048243653, -27.805782945059786, 0.4205253823079967],
                                           [12.151479338961119, -28.537310518275785, 0.40291816982528683]],
                    'ground_truth_nn_plane': [[13.133512578820893, -28.35791997396456, 0.21582995052066864],
                                              [11.959137571117656, -28.59965947050987, 0.213308623084141]],
                    'mode': 'MatchingMode.PLANEDISTANCE',
                    'value': 0.4230510251796533}}
```
