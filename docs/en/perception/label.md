# Object Labels

- For the details, see [perception_eval.common.label.py](../../../perception_eval/perception_eval/common/label.py)

## `AutowareLabel`

| type                       |    value     | support labels                         |
| :------------------------- | :----------: | :------------------------------------- |
| `AutowareLabel.CAR`        |    "car"     | car                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
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
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
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

### Merge similar labels option

- Specify `merge_similar_labels=True` in `PerceptionEvaluationConfig`，then evaluation will be executed with merged labels like below.

  - `AutowareLabel.TRUCK` / `AutowareLabel.BUS` -> `AutowareLabel.CAR`
  - `AutowareLabel.MOTORBIKE` -> `AutowareLabel.BICYCLE`

| type                       |    value     | support labels                         |
| :------------------------- | :----------: | :------------------------------------- |
| `AutowareLabel.CAR`        |    "car"     | car                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
|                            |              | truck                                  |
|                            |              | vehicle.truck                          |
|                            |              | trailer                                |
|                            |              | vehicle.trailer                        |
|                            |              | bus                                    |
|                            |              | vehicle.bus                            |
|                            |              | vehicle.bus (bendy & rigid)            |
| `AutowareLabel.BICYCLE`    |  "bicycle"   | bicycle                                |
|                            |              | vehicle.bicycle                        |
|                            |              | motorbike                              |
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
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

## `TrafficLightLabel`

| type                                   |        value         | support labels     |
| :------------------------------------- | :------------------: | :----------------- |
| `TrafficLightLabel.GREEN`              |       "green"        | green              |
| `TrafficLightLabel.RED`                |        "red"         | red                |
| `TrafficLightLabel.YELLOW`             |       "yellow"       | yellow             |
| `TrafficLightLabel.RED_STRAIGHT`       |    "red_straight"    | red_straight       |
| `TrafficLightLabel.RED_LEFT`           |      "red_left"      | red_left           |
| `TrafficLightLabel.RED_left_straight`  | "red_left_straight"  | red_left_straight  |
| `TrafficLightLabel.RED_RIGHT`          |     "red_right"      | red_right          |
| `TrafficLightLabel.RED_RIGHT_STRAIGHT` | "red_right_straight" | red_right_straight |
| `TrafficLightLabel.RED_RIGHT_DIAGONAL` | "red_right_diagonal" | red_right_diagonal |
| `TrafficLightLabel.YELLOW_RIGHT`       |    "yellow_right"    | yellow_right       |
| `TrafficLightLabel.UNKNOWN`            |      "unknown"       | unknown            |

## [TBD]`BlinkerLabel`

TBD

## [TBD]`BrakeLampLabel`

TBD
