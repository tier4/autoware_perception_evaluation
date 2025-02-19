from __future__ import annotations

import argparse
from collections import defaultdict
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd


def _read_last_line_of_jsonl(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read the last line of a .jsonl file and parse it as JSON.

    Args:
        file_path: A Path object pointing to the .jsonl file.

    Returns:
        A dictionary representing the last JSON object in the file if successful, None otherwise.
    """
    try:
        with open(file_path, "rb") as file:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b"\n":
                file.seek(-2, os.SEEK_CUR)
            return json.loads(file.readline().decode())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def _get_last_lines_from_jsonl(latest_dir: Path) -> Dict[str, Any]:
    """Process each 'result.jsonl' file in the directories under 'latest'.

    Args:
        latest_dir: A Path object pointing to the 'latest' directory.

    Returns:
        An OrderedDict where keys are scenario names and values are the last line from each 'result.jsonl' file.
    """
    last_lines = OrderedDict()
    for subdir in latest_dir.iterdir():
        # Skip if not a directory
        if not subdir.is_dir():
            continue

        scenario_name = subdir.name
        result_jsonl_path = subdir / "result.jsonl"
        # Skip if 'result.jsonl' does not exist
        if not result_jsonl_path.exists():
            last_lines[scenario_name] = None
            continue

        last_line = _read_last_line_of_jsonl(result_jsonl_path)
        # Skip if the last line is None
        if last_line is None:
            continue

        # Collect last_lines
        last_lines[scenario_name] = last_line

    return last_lines


def _calc_overall_metrics(last_lines: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate the mean, max, and min values for FinalMetrics across all results.

    Args:
        last_lines: A dictionary where keys are scenario names and values are the last line from each 'result.jsonl' file.

    Returns:
        A list of dictionaries with metrics calculated across all scenarios, formatted as {"object_class": class, "metric": metric, "values": {"min": min, "max": max, "mean": mean}}.
    """
    metrics_summary = defaultdict(lambda: {"min": [], "max": [], "mean": []})

    # Iterate through each scenario and its metrics
    for scenario, last_line in last_lines.items():
        if last_line is None:
            continue
        final_metrics = last_line.get("Frame", {}).get("FinalMetrics", {})
        for object_class, metrics in final_metrics.items():
            for metric, values in metrics.items():
                # Append mean, max, and min values for each metric of each class if they exist
                if values.get("mean") is not None:
                    metrics_summary[(object_class, metric)]["mean"].append(values["mean"])
                if values.get("max") is not None:
                    metrics_summary[(object_class, metric)]["max"].append(values["max"])
                if values.get("min") is not None:
                    metrics_summary[(object_class, metric)]["min"].append(values["min"])

    # Calculate mean, max, and min values for each metric-class combination
    result = []
    for (object_class, metric), value_dict in metrics_summary.items():
        result.append(
            {
                "object_class": object_class,
                "metric": metric,
                "values": {
                    "min": min(value_dict["min"]) if value_dict["min"] else None,
                    "max": max(value_dict["max"]) if value_dict["max"] else None,
                    "mean": (sum(value_dict["mean"]) / len(value_dict["mean"]) if value_dict["mean"] else None),
                },
            }
        )

    return result


def _create_metrics_dataframe(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from the calculated metrics.

    Args:
        metrics: A list of dictionaries with metrics calculated across all scenarios, formatted as {"object_class": class, "metric": metric, "values": {"min": min, "max": max, "mean": mean}}.

    Returns:
        A pandas DataFrame with object classes as rows and {metric}_{min/max/mean} as columns.
    """
    # Initialize an empty dict to hold the restructured data
    data = {}

    # Iterate through the metrics to populate the data dict
    for metric_dict in metrics:
        object_class = metric_dict["object_class"]
        metric = metric_dict["metric"]
        values = metric_dict["values"]

        if object_class not in data:
            data[object_class] = {}

        for value_type, value in values.items():
            # Combine metric and value type to form the column name
            column_name = f"{metric}_{value_type}"
            data[object_class][column_name] = value

    # Convert the dict to a DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    # Optionally, sort the columns if needed
    df = df.sort_index(axis=1)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, help="Directory path to DLR results")
    args = parser.parse_args()

    result_dir = Path(args.result)
    last_lines = _get_last_lines_from_jsonl(result_dir)

    mean_values = _calc_overall_metrics(last_lines)

    # Create a DataFrame from the calculated metrics
    metrics_df = _create_metrics_dataframe(mean_values)

    # Extract only predicted path related metrics
    predicted_path_keys: list[str] = [key for key in metrics_df.keys() if "predicted_path" in key]
    time_lengths = set([key.split("_")[-2] for key in predicted_path_keys])
    for time_length in time_lengths:
        key_at = [key for key in predicted_path_keys if time_length in key]
        predicted_path_metrics_df = metrics_df[key_at]

        print(f"=== {time_length}[s] ===")
        print(predicted_path_metrics_df)
        print("\n")


if __name__ == "__main__":
    main()
