# EDA Visualizer

- EDA 用の可視化ツール

## How to Use

- See [perception_eval/test/eda.py](../../../perception_eval/test/eda.py)

  ```python
      visualizer = EDAVisualizer(all_ground_truths, save_path + "/" + objects_name)
  ```

  - `eda.py`では ground truth を可視化しているが，推論結果（`List[DynamicObjectWithPerceptionResult]`）の可視化も可能．

    ```python

        # Show histogram of number of objects that are less than the certain distance in x-y plane
        visualizer.hist_object_count_for_each_distance(class_names, ranges_xy = ranges_xy)
    ```

  - どの距離までの object の数をヒストグラムとして可視化したいかを`ranges_xy`で指定

    ```python
        # Show histogram of distance in x-y plane of objects
        visualizer.hist_object_dist2d_for_each_class(class_names)

        # Show 2d-histogram of width and length in each class
        visualizer.hist2d_object_wl_for_each_class(class_names, width_lim_dict=width_lim_dict, length_lim_dict=length_lim_dict)
    ```

  - どの範囲の object について width, length を可視化したいかを`width_lim_dict`と`length_lim_dict`で指定

    ```python
        # Show 2d-histogram of x and y in each class
        visualizer.hist2d_object_center_xy_for_each_class(class_names, xlim_dict=xylim_dict, ylim_dict=xylim_dict)
    ```

  - どの範囲の object について中心位置を可視化したいかを`xlim_dict`と`ylim_dict`で指定

    ```python
        # Show 2d-histogram of number of point clouds in each class
        visualizer.hist2d_object_num_points_for_each_class(class_names)

        # Get pandas profiling report
        visualizer.get_pandas_profiling(class_names, "profiling_" + objects_name)
    ```
