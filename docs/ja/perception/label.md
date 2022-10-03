# Autoware Label

- 詳細は[perception_eval.common.label.py](../../perception_eval/perception_eval/common/label.py)を参照

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

## 類似ラベルのマージ

- `PerceptionEvaluationConfig`の引数`merge_similar_labels=True`とすると，以下のように類似ラベルがマージされた状態で評価される．

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
