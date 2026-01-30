#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def extract_failed_l2_task_ids(records: list[dict]) -> list[str]:
    failed_task_ids: set[str] = set()
    for item in records:
        task_id = item.get("task_id")
        if not task_id:
            continue
        for rec in item.get("trajectory_records", []):
            path = rec.get("path", "")
            if "workarena-l2" not in path:
                continue
            if rec.get("reward") == 0:
                failed_task_ids.add(task_id)
            break
    return sorted(failed_task_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract failed workarena-l2 task_ids from a trajectory journal."
    )
    default_input = Path(__file__).resolve().parent / "longmemevalv2_trajectory_collection_journal.json"
    default_output = Path(__file__).resolve().parent / "failed_l2_task_ids.json"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to the journal JSON file.")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to write failed task IDs JSON.")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list at the top level of the JSON file.")

    failed_task_ids = extract_failed_l2_task_ids(data)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(failed_task_ids, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()
