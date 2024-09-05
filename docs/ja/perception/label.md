# オブジェクト　ラベル

- 詳細は[perception_eval.common.label.py](../../../perception_eval/perception_eval/common/label.py)を参照

## Autoware Label

| type                       |    value     | support labels                         |
| :------------------------- | :----------: | :------------------------------------- |
| `AutowareLabel.CAR`        |    "car"     | car                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
|                            |              | vehicle.police                         |
|                            |              | vehicle.fire                           |
|                            |              | vehicle.ambulance                      |
| `AutowareLabel.TRUCK`      |   "truck"    | truck                                  |
|                            |              | vehicle.truck                          |
|                            |              | trailer                                |
|                            |              | vehicle.trailer                        |
| `AutowareLabel.BUS`        |    "bus"     | bus                                    |
|                            |              | vehicle.bus                            |
|                            |              | vehicle.bus (bendy & rigid)            |
| `AutowareLabel.BICYCLE`    |  "bicycle"   | bicycle                                |
|                            |              | vehicle.bicycle                        |
| `AutowareLabel.MOTORBIKE`  | "motorbike"  | motorbike                              |
|                            |              | motorcycle                             |
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
|                            |              | stroller                               |
|                            |              | pedestrian.adult                       |
|                            |              | pedestrian.child                       |
|                            |              | pedestrian.construction_worker         |
|                            |              | pedestrian.personal_mobility           |
|                            |              | pedestrian.police_officer              |
|                            |              | pedestrian.stroller                    |
|                            |              | pedestrian.wheelchair                  |
| `AutowareLabel.ANIMAL`     |   "animal"   |                                        |
| `AutowareLabel.UNKNOWN`    |  "unknown"   | unknown                                |
|                            |              | animal                                 |
|                            |              | movable_object.barrier                 |
|                            |              | movable_object.debris                  |
|                            |              | movable_object.pushable_pullable       |
|                            |              | movable_object.trafficcone             |
|                            |              | movable_object.traffic_cone            |
|                            |              | static_object.bicycle rack             |
|                            |              | static_object.bollard                  |
|                            |              | static_object.forklift                 |

### 類似ラベルのマージ

- `PerceptionEvaluationConfig`の引数`merge_similar_labels=True`とすると，以下のように類似ラベルがマージされた状態で評価される．

  - `AutowareLabel.TRUCK` / `AutowareLabel.BUS` -> `AutowareLabel.CAR`
  - `AutowareLabel.MOTORBIKE` -> `AutowareLabel.BICYCLE`

| type                       |    value     | support labels                         |
| :------------------------- | :----------: | :------------------------------------- |
| `AutowareLabel.CAR`        |    "car"     | car                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
|                            |              | vehicle.police                         |
|                            |              | vehicle.fire                           |
|                            |              | vehicle.ambulance                      |
|                            |              | truck                                  |
|                            |              | vehicle.truck                          |
|                            |              | trailer                                |
|                            |              | vehicle.trailer                        |
|                            |              | bus                                    |
|                            |              | vehicle.bus                            |
|                            |              | vehicle.bus (bendy & rigid)            |
| `AutowareLabel.BICYCLE`    |  "bicycle"   | bicycle                                |
|                            |              | motorcycle                             |
|                            |              | vehicle.bicycle                        |
|                            |              | motorbike                              |
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
|                            |              | stroller                               |
|                            |              | pedestrian.adult                       |
|                            |              | pedestrian.child                       |
|                            |              | pedestrian.construction_worker         |
|                            |              | pedestrian.personal_mobility           |
|                            |              | pedestrian.police_officer              |
|                            |              | pedestrian.stroller                    |
|                            |              | pedestrian.wheelchair                  |
| `AutowareLabel.ANIMAL`     |   "animal"   |                                        |
| `AutowareLabel.UNKNOWN`    |  "unknown"   | unknown                                |
|                            |              | animal                                 |
|                            |              | movable_object.barrier                 |
|                            |              | movable_object.debris                  |
|                            |              | movable_object.pushable_pullable       |
|                            |              | movable_object.trafficcone             |
|                            |              | movable_object.traffic_cone            |
|                            |              | static_object.bicycle rack             |
|                            |              | static_object.bollard                  |
|                            |              | static_object.forklift                 |

## `TrafficLightLabel`

- `EvaluationTask.DETECTION2D` もしくは `EvaluationTask.Tracking2D`の場合

  | type                              |      value      | support labels     |
  | :-------------------------------- | :-------------: | :----------------- |
  | `TrafficLightLabel.TRAFFIC_LIGHT` | "traffic_light" | traffic_light      |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | green              |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red                |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | yellow             |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_straight       |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_left           |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_left_straight  |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_right          |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_right_straight |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | red_right_diagonal |
  | `TrafficLightLabel.TRAFFIC_LIGHT` |                 | yellow_right       |
  | `TrafficLightLabel.UNKNOWN`       |    "unknown"    | unknown            |

- `EvaluationTask.CLASSIFICATION2D`の場合

  | type                                   |        value         | support labels     |
  | :------------------------------------- | :------------------: | :----------------- |
  | `TrafficLightLabel.GREEN`              |       "green"        | green              |
  | `TrafficLightLabel.RED`                |        "red"         | red                |
  | `TrafficLightLabel.YELLOW`             |       "yellow"       | yellow             |
  | `TrafficLightLabel.RED_STRAIGHT`       |    "red_straight"    | red_straight       |
  | `TrafficLightLabel.RED_LEFT`           |      "red_left"      | red_left           |
  | `TrafficLightLabel.RED_LEFT_STRAIGHT`  | "red_left_straight"  | red_left_straight  |
  | `TrafficLightLabel.RED_RIGHT`          |     "red_right"      | red_right          |
  | `TrafficLightLabel.RED_RIGHT_STRAIGHT` | "red_right_straight" | red_right_straight |
  | `TrafficLightLabel.RED_RIGHT_DIAGONAL` | "red_right_diagonal" | red_right_diagonal |
  | `TrafficLightLabel.YELLOW_RIGHT`       |    "yellow_right"    | yellow_right       |
  | `TrafficLightLabel.UNKNOWN`            |      "unknown"       | unknown            |

## [TBD]`BlinkerLabel`

TBD

## [TBD]`BrakeLampLabel`

TBD
