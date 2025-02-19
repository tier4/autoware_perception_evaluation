from __future__ import annotations

import argparse
from glob import glob
import os.path as osp

import numpy as np
import pandas as pd
from perception_eval.tool import PerceptionAnalyzer3D
from tabulate import tabulate


def count_max_consecutive_fn(analyzer: PerceptionAnalyzer3D, distance: tuple[float, float]) -> dict:
    df = analyzer.filter_by_distance(distance=distance)

    max_counts = []
    max_time_spans = []
    # Compute max consecutive FN frame length for each uuid
    for uuid in pd.unique(df.xs("ground_truth", level=1)["uuid"]):
        if uuid is None:
            continue

        max_count = 0
        max_time_span = 0.0
        current_count = 0
        run_start_ts = None

        fn_df = df[np.bitwise_and(df["uuid"] == uuid, df["status"] == "FN")]
        frame_diff = np.diff(fn_df["frame"], append=0)
        timestamp = fn_df["timestamp"].tolist()
        for i, value in enumerate(frame_diff):
            if value == 1.0:
                # Update max_count if current streak is larger than what we've seen before.
                if current_count == 0:
                    run_start_ts = timestamp[i]
                current_count += 1
            else:
                if current_count > 0:
                    if current_count > max_count:
                        max_count = current_count
                    run_end_ts = timestamp[i - 1]
                    time_span = run_end_ts - run_start_ts
                    if time_span > max_time_span:
                        max_time_span = time_span
                # If we see a 0.0, reset the current streak count to 0.
                current_count = 0
                run_start_ts = None

        # Handle the case if the longest run ends at the last element
        if current_count > 0:
            # Check the final run
            if current_count > max_count:
                max_count = current_count
            run_end_ts = timestamp[-1]  # the last timestamp
            time_span = run_end_ts - run_start_ts
            if time_span > max_time_span:
                max_time_span = time_span

        max_counts.append(max_count)
        max_time_spans.append(max_time_span * 1e-6)

    return {
        # "Max Consecutive FN Frames": max(max_counts),
        # "Min Consecutive FN Frames": min(max_counts),
        # "Mean Consecutive FN Frames": f"{np.mean(max_counts):.3f}" if len(max_counts) > 0 else 0.0,
        # "Std Consecutive FN Frames": f"{np.std(max_counts):.3f}" if len(max_counts) > 0 else 1.0,
        # "Median Consecutive FN Frames": np.median(max_counts) if len(max_counts) > 0 else 0,
        "Max Consecutive FN Time": f"{max(max_time_spans):.3f}",
        "Min Consecutive FN Time": min(max_time_spans),
        "Mean Consecutive FN Time": f"{np.mean(max_time_spans):.3f}" if len(max_time_spans) > 0 else 0.0,
        "Std Consecutive FN Time": f"{np.std(max_time_spans):.3f}" if len(max_time_spans) > 0 else 1.0,
        "Median Consecutive FN Time": np.median(max_time_spans) if len(max_time_spans) > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str, help="File path to scenario")
    parser.add_argument("result", type=str, help="Directory path to DLR result")
    parser.add_argument("-o", "--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    result_archives = glob(osp.join(args.result, "**/result_archive/*.pkl"), recursive=True)

    analyzer = PerceptionAnalyzer3D.from_scenario(args.output, args.scenario)

    for archive in result_archives:
        analyzer.add_from_pkl(archive)

    print(analyzer.df)

    # print(analyzer.df["status"].to_string())

    profiles = []

    distances = [(i * 10, (i + 1) * 10) for i in range(0, 13)]
    for distance in distances:
        counts = count_max_consecutive_fn(analyzer, distance)
        analysis = analyzer.analyze(distance=distance)
        fp = analysis.score["FP"]["ALL"]
        fn = analysis.score["FN"]["ALL"]
        err_x = analysis.error["average"][("ALL", "x")]
        err_y = analysis.error["average"][("ALL", "y")]
        err_pos = np.linalg.norm([err_x, err_y])
        err_length = analysis.error["average"][("ALL", "length")]
        err_width = analysis.error["average"][("ALL", "width")]
        err_area = abs(err_length * err_width)
        err_speed = analysis.error["average"][("ALL", "speed")]

        df = analyzer.filter_by_distance(distance)
        num_tp = analyzer.get_num_tp(df)
        num_fp = analyzer.get_num_fp(df)
        num_fn = analyzer.get_num_fn(df)

        precision = num_tp / (num_tp + num_fp) if num_tp + num_fp != 0 else 1.0
        recall = num_tp / (num_tp + num_fn) if num_tp + num_fn != 0 else 1.0

        profile = {
            "Distance[m]": distance[1],  # (min, max)
            "FP[%]": f"{fp * 100:.3f}",
            "FN[%]": f"{fn * 100:.3f}",
            "Precision[%]": f"{precision * 100:.3f}",
            "Recall[%]": f"{recall * 100:.3f}",
            "Position Error[m]": f"{err_pos:.3f}",
            "Area Error[m^2]": f"{err_area:.3f}",
            "Speed Error[m/s]": f"{err_speed:.3f}",
        }
        profile.update(counts)

        profiles.append(profile)

    print(tabulate(profiles, headers="keys"))


if __name__ == "__main__":
    main()
