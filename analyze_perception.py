from __future__ import annotations

import argparse
from glob import glob
import os.path as osp

import numpy as np
import pandas as pd
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.tool import PerceptionAnalyzer3D
from tabulate import tabulate


def count_max_consecutive_fn(analyzer: PerceptionAnalyzer3D, distance: tuple[float, float]) -> dict:
    df = analyzer.filter_by_distance(distance=distance)

    if len(df) == 0:
        return {}  # TODO(ktro2828): Temporal

    max_counts = []
    max_time_spans = []
    max_uuids = []
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
        max_uuids.append(uuid)

    # max_uuid = max_uuids[np.argmax(max_time_spans)]
    # max_value = np.max(max_time_spans)
    # print(f"{distance[1]}: {max_uuid}, {max_value}[s]")

    sort_idx = np.argsort(max_time_spans)[::-1]
    sort_uuids = np.asarray(max_uuids)[sort_idx]
    sort_values = np.sort(max_time_spans)[::-1]

    print(f"{distance[1]}[m]:")
    for u, v in zip(sort_uuids, sort_values):
        print(f"{u}: {v}[s]")

    return {
        # "Max Consecutive FN Frames": max(max_counts),
        # "Min Consecutive FN Frames": min(max_counts),
        # "Mean Consecutive FN Frames": f"{np.mean(max_counts):.3f}" if len(max_counts) > 0 else 0.0,
        # "Std Consecutive FN Frames": f"{np.std(max_counts):.3f}" if len(max_counts) > 0 else 1.0,
        # "Median Consecutive FN Frames": np.median(max_counts) if len(max_counts) > 0 else 0,
        "Max Consecutive FN Time": f"{max(max_time_spans):.3f}",
        "Min Consecutive FN Time": min(max_time_spans),
        "Mean Consecutive FN Time": f"{np.mean(max_time_spans):.3f}" if len(max_time_spans) > 0 else np.nan,
        "Std Consecutive FN Time": f"{np.std(max_time_spans):.3f}" if len(max_time_spans) > 0 else np.nan,
        "Median Consecutive FN Time": np.median(max_time_spans) if len(max_time_spans) > 0 else np.nan,
        "50Percentile Consecutive FN Time": np.percentile(max_time_spans, 50) if len(max_time_spans) > 0 else np.nan,
        "99Percentile Consecutive FN Time": np.percentile(max_time_spans, 99) if len(max_time_spans) > 0 else np.nan,
    }


def analyze_prediction(
    analyzer: PerceptionAnalyzer3D,
    distance: tuple[float, float],
    time_steps: list[float],
) -> dict[int, dict]:
    df = analyzer.filter_by_distance(distance=distance)
    time_step_results: dict[int, list[tuple[float, float]]] = {t: [] for t in time_steps}
    for uuid in pd.unique(df["uuid"]):
        scenes = df[df["uuid"] == uuid]["scene"].values.tolist()
        frames = df[df["uuid"] == uuid]["frame"].values.tolist()
        for t in time_steps:
            for scene in scenes:
                for frame in frames:
                    future_df_t = analyzer.future_at(uuid=uuid, t=t, scene=scene, frame=frame)
                    if len(future_df_t) > 0:
                        time_step_results[t].append(future_df_t[["err_x", "err_y"]].values)

    outputs = {}
    for t, results in time_step_results.items():
        n_ret = len(results)
        if n_ret == 0:  # TODO(ktro2828): Temporal
            continue
        results = np.concatenate(results, axis=0)
        outputs_t = {}
        outputs_t["Max Error X[m]"] = np.nanmax(results[:, 0]) if n_ret > 0 else np.nan
        outputs_t["Max Error Y[m]"] = np.nanmax(results[:, 1]) if n_ret > 0 else np.nan
        outputs_t["Min Error X[m]"] = np.nanmin(results[:, 0]) if n_ret > 0 else np.nan
        outputs_t["Min Error Y[m]"] = np.nanmin(results[:, 1]) if n_ret > 0 else np.nan
        outputs_t["Mean Error X[m]"] = np.nanmean(results[:, 0]) if n_ret > 0 else np.nan
        outputs_t["Mean Error Y[m]"] = np.nanmean(results[:, 1]) if n_ret > 0 else np.nan
        outputs_t["Std Error X[m]"] = np.nanstd(results[:, 0]) if n_ret > 0 else np.nan
        outputs_t["Std Error Y[m]"] = np.nanstd(results[:, 1]) if n_ret > 0 else np.nan
        outputs_t["50Percentile Error X[m]"] = np.nanpercentile(results[:, 0], 50) if n_ret > 0 else np.nan
        outputs_t["50Percentile Error Y[m]"] = np.nanpercentile(results[:, 1], 50) if n_ret > 0 else np.nan
        outputs_t["99Percentile Error X[m]"] = np.nanpercentile(results[:, 0], 99) if n_ret > 0 else np.nan
        outputs_t["99Percentile Error Y[m]"] = np.nanpercentile(results[:, 1], 99) if n_ret > 0 else np.nan
        outputs[t] = outputs_t
    return outputs


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
    print(analyzer.future_df.to_string())

    perception_profiles = []

    time_steps = [0, 4, 9]  # [1, 5, 10]
    prediction_profiles = {t: [] for t in time_steps}
    distances = [(i * 10, (i + 1) * 10) for i in range(0, 13)]
    for distance in distances:
        counts = count_max_consecutive_fn(analyzer, distance)
        analysis = analyzer.analyze(distance=distance)
        if analysis.score is None:  # TODO(ktro2828): Temporal
            continue
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

        perception_profile = {
            "Distance[m]": distance[1],  # (min, max)
            "FP[%]": f"{fp * 100:.3f}",
            "FN[%]": f"{fn * 100:.3f}",
            "Precision[%]": f"{precision * 100:.3f}",
            "Recall[%]": f"{recall * 100:.3f}",
            "Position Error[m]": f"{err_pos:.3f}",
            "Area Error[m^2]": f"{err_area:.3f}",
            "Speed Error[m/s]": f"{err_speed:.3f}",
        }
        perception_profile.update(counts)

        perception_profiles.append(perception_profile)

        if analyzer.config.evaluation_task == EvaluationTask.PREDICTION:
            prediction_profile = analyze_prediction(analyzer, distance, time_steps)
            for t, profiles in prediction_profile.items():
                prediction_profiles[t].append({"Distance[m]": distance[1], **profiles})

    print("=== Perception Analysis ===")
    print(tabulate(perception_profiles, headers="keys"))

    if any(len(p) > 0 for _, p in prediction_profiles.items()):
        print("=== Prediction Analysis ===")
        for t, profiles in prediction_profiles.items():
            print(f"===@{t} steps===")
            print(tabulate(profiles, headers="keys"))


if __name__ == "__main__":
    main()
