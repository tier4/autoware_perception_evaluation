# autoware_perception_evaluation

**perception_eval** is a tool to evaluate perception tasks.

## Documents

[English](docs/en/README.md) | [日本語](docs/ja/README.md)

## Overview

### Evaluate Detection/Tracking/Prediction results

- We support 3D Detection/Tracking/Prediction and 2D Detection evaluation.

  - 3D task

  | Task       | Metrics | Sub-metrics          |
  | :--------- | :-----: | :------------------- |
  | Detection  |   mAP   | AP, APH              |
  | Tracking   |  CLEAR  | MOTA, MOTP, IDswitch |
  | Prediction |   WIP   | WIP                  |

  - 2D task

  | Task      | Metrics | Sub-metrics |
  | :-------- | :-----: | :---------- |
  | Detection |   mAP   | AP          |

- Also, we can evaluate sensing results.
