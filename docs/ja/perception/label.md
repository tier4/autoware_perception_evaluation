# Autoware Label

- About detail, see [awml_evaluation.common.label.py](../../awml_evaluation/awml_evaluation/common/label.py)

| type                       |    value     | support labels                         |
| :------------------------- | :----------: | :------------------------------------- |
| `AutowareLabel.CAR`        |    "car"     | car                                    |
|                            |              | CAR                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
| `AutowareLabel.TRUCK`      |   "truck"    | truck                                  |
|                            |              | TRUCK                                  |
|                            |              | vehicle.truck                          |
|                            |              | vehicle.trailer                        |
| `AutowareLabel.BUS`        |    "bus"     | bus                                    |
|                            |              | BUS                                    |
|                            |              | vehicle.bus                            |
|                            |              | vehicle.bus (bendy & rigid)            |
| `AutowareLabel.BICYCLE`    |  "bicycle"   | bicycle                                |
|                            |              | BICYCLE                                |
|                            |              | vehicle.bicycle                        |
| `AutowareLabel.MOTORBIKE`  | "motorbike"  | motorbike                              |
|                            |              | MOTORBIKE                              |
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
|                            |              | PEDESTRIAN                             |
|                            |              | pedestrian.adult                       |
|                            |              | pedestrian.child                       |
|                            |              | pedestrian.construction_worker         |
|                            |              | pedestrian.personal_mobility           |
|                            |              | pedestrian.police_officer              |
|                            |              | pedestrian.stroller                    |
|                            |              | pedestrian.wheelchair                  |
| `AutowareLabel.ANIMAL`     |   "animal"   |                                        |
| `AutowareLabel.UNKNOWN`    |  "unknown"   | unknown                                |
|                            |              | UNKNOWN                                |
|                            |              | animal                                 |
|                            |              | ANIMAL                                 |
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
|                            |              | CAR                                    |
|                            |              | vehicle.car                            |
|                            |              | vehicle.construction                   |
|                            |              | vehicle.emergency (ambulance & police) |
|                            |              | truck                                  |
|                            |              | TRUCK                                  |
|                            |              | vehicle.truck                          |
|                            |              | vehicle.trailer                        |
|                            |              | bus                                    |
|                            |              | BUS                                    |
|                            |              | vehicle.bus                            |
|                            |              | vehicle.bus (bendy & rigid)            |
| `AutowareLabel.BICYCLE`    |  "bicycle"   | bicycle                                |
|                            |              | BICYCLE                                |
|                            |              | vehicle.bicycle                        |
|                            |              | motorbike                              |
|                            |              | MOTORBIKE                              |
|                            |              | vehicle.motorcycle                     |
| `AutowareLabel.PEDESTRIAN` | "pedestrian" | pedestrian                             |
|                            |              | PEDESTRIAN                             |
|                            |              | pedestrian.adult                       |
|                            |              | pedestrian.child                       |
|                            |              | pedestrian.construction_worker         |
|                            |              | pedestrian.personal_mobility           |
|                            |              | pedestrian.police_officer              |
|                            |              | pedestrian.stroller                    |
|                            |              | pedestrian.wheelchair                  |
| `AutowareLabel.ANIMAL`     |   "animal"   |                                        |
| `AutowareLabel.UNKNOWN`    |  "unknown"   | unknown                                |
|                            |              | UNKNOWN                                |
|                            |              | animal                                 |
|                            |              | ANIMAL                                 |
|                            |              | movable_object.barrier                 |
|                            |              | movable_object.debris                  |
|                            |              | movable_object.pushable_pullable       |
|                            |              | movable_object.trafficcone             |
|                            |              | movable_object.traffic_cone            |
|                            |              | static_object.bicycle rack             |
