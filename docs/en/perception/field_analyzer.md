# [`PerceptionAnalyzer3DField`](../../../perception_eval/perception_eval/tool/perception_analyzer3dfield.py)

`PerceptionAnalyzer3DField` is a class designed to analyze the performance of perception across various axes of variables.
This class extends the functionality of `PerceptionAnalyzer3D` by providing statistical analysis of perception results divided into given axes and intervals (bins).

The analysis includes the number of samples (histogram) for each grid cell, as well as the detection probability (True Positive, True Negative, False Negative, False Positive rates). For True Positive cases, it also calculates the mean and standard deviation of positions and object sizes, estimation errors, and uncertainties.

Key features of the `PerceptionAnalyzer3DField` class include:

- Ability to set custom axes and grid boundaries
- Calculation of detection probability on each grid cell
- Analysis of error and uncertainties of True Positive cases

The term "uncertainty" refers to the probability distribution of the ground truth when an estimation is given.

Similar to `PerceptionAnalyzer3D`, this class uses saved evaluation results from `driving_log_replayer`.

## Grid cell Analysis

### Usage

An example script for performing perception field analysis can be found at [perception_field_analysis.py](../../../perception_eval/test/perception_field_analysis.py).

To run the script, you need to provide the following inputs:

- `result_root_directory`: The root directory where the result files are located.
- `scenario_path`: The path to the scenario file.

The script will generate plots and save them in the `<your-result_root_directory>/plot/<object class>/` directory.

Here is an example folder structure to work with this test script:

```md
BASE_FOLDER  
├── archives # root directory of the analyzing simulation results
│ ├── xxx_archive
│ ├── core.xxx # core file will not used
│ └── scene_result.pkl # analyzing .pkl files
└── scenario_x.yml # scenario file
```

To run the script, use the following command:

```sh
python3 test/perception_field_analysis.py -r ${BASE_FOLDER} -s ${BASE_FOLDER}/${SCENARIO_FILE}
```

### Procedure

1. Load files
    
    Initialize the `PerceptionAnalyzer3DField` analyzer and load the `scene_result.pkl` files stored in all of subfolder of the root directory.
    This will parse the .pkl files and load the data into a dataframe.
    
    ```python
    # Initialize
    analyzer: PerceptionAnalyzer3DField = PerceptionAnalyzer3DField.from_scenario(
       result_root_directory,
       scenario_path,
    )
    
    # Load files
    pickle_file_paths = Path(result_root_directory).glob("**/scene_result.pkl")
    for filepath in pickle_file_paths:
       analyzer.add_from_pkl(filepath.as_posix())
    ```

2. (Optional) Post-process

    Add columns to the dataframe for extended analysis.
    
    ```python
    # Add columns
    analyzer.addAdditionalColumn()
    analyzer.addErrorColumns()
    ```

3. Set axes

    Set one or two axes to be used for the field analysis. Any label (column) in the dataframe can be used as an axis.
    A typical axis is position.

    ```python
    axis_x: PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="x")
    axis_y: PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="y")
    ```

    Each axis can have an array of boundary of bins.

    ```python
    grid_axis_xy: np.ndarray = np.array([-35, -25, -15, -5, 5, 15, 25, 35])
    axis_x.setGridAxis(grid_axis_xy)
    axis_y.setGridAxis(grid_axis_xy)
    ```

4. Analyze

    Set a pair of axes to the analyzer and process the data.

    ```python
    error_field, uncertainty_field = analyzer.analyzeXY(axis_x, axis_y, **kwargs)
    ```

    - `error_field`: The target bin of the given data will be sorted by the value of the ground truth.
    - `uncertainty_field`: The target bin of the given data will be sorted by the value of the estimation.

5. Plot

    Set plots and hand over the fields to the plot methods.
    
    ```python
    plots: PerceptionFieldPlots = PerceptionFieldPlots(plot_dir)
    plots.plot_field_basics(error_field, prefix="XY")
    
    plots.save()
    ```

## (Grid) point Analysis

This analysis method is utilize all of the object as points.
At this moment, all of the points are in the same table, but will be separated into grid of tables in further development.

### Usage

An example script can be found at [perception_field_points_analysis.py](../../../perception_eval/test/perception_field_points_analysis.py).
