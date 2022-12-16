# autoware_perception_evaluation

**perception_eval** is a tool to evaluate perception tasks.

## Documents

[English](docs/en/README.md) | [日本語](docs/ja/README.md)

## Overview

### Evaluate Perception(=Detection/Tracking/Prediction/Classification) & Sensing task

- 3D task

| Task       |     Metrics      | Sub-metrics                         |
| :--------- | :--------------: | :---------------------------------- |
| Detection  |       mAP        | AP, APH                             |
| Tracking   |      CLEAR       | MOTA, MOTP, IDswitch                |
| Prediction |       WIP        | WIP                                 |
| Sensing    | Check Pointcloud | Detection Area & Non-detection Area |

- 2D task

| Task             | Metrics  | Sub-metrics          |
| :--------------- | :------: | :------------------- |
| Detection2D      |   mAP    | AP                   |
| Tracking2D       |  CLEAR   | MOTA, MOTP, IDswitch |
| Classification2D | Accuracy | Accuracy             |
