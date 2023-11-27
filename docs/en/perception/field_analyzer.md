# [`<class> PerceptionAnalyzer3DField(...)`](../../../perception_eval/perception_eval/tool/perception_analyzer3dfield.py)

Analyze perception performance in various axes of variables. 
This analysis is an extension of `PerceptionAnalyzer3D`. 

The perception field analyzer enables to
  - set custom axis and grid boundaries
  - get detection probability (TP, TN, FN, FP rates) on each grid cells
  - get error and uncertainties of TP cases



## How to use
As same as the `PerceptionAnalyzer3D`, the evaluator uses saved evaluation results from driving_log_replayer.

An example script is [perception_field_analysis.py](../../../perception_eval/test/perception_field_analysis.py).

Inputs: result_root_directory, scenario_path

```sh
source install/setup.bash

python3 'src/simulator/perception_eval/perception_eval/test/perception_field_analysis.py' -r <your-result_root_directory> -s <your-scenario_path>
```

Outputs: The plots will be saved in `<your-result_root_directory>/plot/<object class>/*`


## Procedure

1. Load files

2. Set axes

3. Analyze

4. Plot
