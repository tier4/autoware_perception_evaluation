# EDA visualizer

- Visualization tool for EDA

## How to Use

- See [perception_eval/test/eda.py](../../../perception_eval/test/eda.py)

  - You can visualize both of GT and estimation in `eda.py`

  ```python
      visualizer = EDAVisualizer(all_ground_truths, save_path + "/" + objects_name)
  ```

  - Use `range_xy` to specify how far the objects you want to visualize the number of them with histogram

  ```python
      # Show histogram of number of objects that are less than the certain distance in x-y plane
      visualizer.hist_object_count_for_each_distance(class_names, ranges_xy = ranges_xy)
  ```

  - Use `width_lim_dict` and `length_lim_dict` to specify how far the objects you want to visualize the width and length of them

  ```python
        # Show histogram of distance in x-y plane of objects
        visualizer.hist_object_dist2d_for_each_class(class_names)

        # Show 2d-histogram of width and length in each class
        visualizer.hist2d_object_wl_for_each_class(class_names, width_lim_dict=width_lim_dict, length_lim_dict=length_lim_dict)
  ```

  - Use `xlim_dict` and `ylim_dict` to specify how far the objects you want to visualize the center position of them

  ```python
        # Show 2d-histogram of x and y in each class
        visualizer.hist2d_object_center_xy_for_each_class(class_names, xlim_dict=xylim_dict, ylim_dict=xylim_dict)
  ```

  ```python
        # Show 2d-histogram of number of point clouds in each class
        visualizer.hist2d_object_num_points_for_each_class(class_names)

        # Get pandas profiling report
        visualizer.get_pandas_profiling(class_names, "profiling_" + objects_name)
  ```
