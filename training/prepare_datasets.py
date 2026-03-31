import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


OUTPUT_FIELDS = ["dataset_name", "text1", "text2", "label"]
TARGET_SPLITS = ["train", "validation", "test"]


def normalize_label(value: Any, scale: float) -> float:
    score = float(value)
    if scale and scale != 1.0:
        score /= scale
    return max(0.0, min(1.0, score))


def append_example(storage: dict[str, list[dict[str, Any]]], split_name: str, dataset_name: str, text1: str, text2: str, label: float) -> None:
    text1 = str(text1).strip()
    text2 = str(text2).strip()
    if not text1 or not text2:
        return

    storage[split_name].append(
        {
            "dataset_name": dataset_name,
            "text1": text1,
            "text2": text2,
            "label": round(float(label), 6),
        }
    )


def process_pair_dataset(dataset_config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    dataset = load_dataset(dataset_config["path"], dataset_config.get("config"))
    output = {split_name: [] for split_name in TARGET_SPLITS}
    label_scale = float(dataset_config.get("label_scale", 1.0))

    for split_name in TARGET_SPLITS:
        if split_name not in dataset:
            continue

        split = dataset[split_name]
        limit = dataset_config.get(f"{split_name}_examples")
        if limit is not None:
            split = split.shuffle(seed=seed).select(range(min(int(limit), len(split))))

        for row in split:
            append_example(
                output,
                split_name,
                dataset_config["name"],
                row[dataset_config["text1_column"]],
                row[dataset_config["text2_column"]],
                normalize_label(row[dataset_config["label_column"]], label_scale),
            )

    return output


def process_triplet_dataset(dataset_config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    split = load_dataset(dataset_config["path"], dataset_config.get("config"), split="train")
    split = split.shuffle(seed=seed)

    requested_limits = {name: int(dataset_config.get(f"{name}_examples", 0) or 0) for name in TARGET_SPLITS}
    negatives_per_query = int(dataset_config.get("negatives_per_query", 1))
    negative_prefix = dataset_config.get("negative_prefix", "negative_")
    negative_columns = sorted([name for name in split.column_names if name.startswith(negative_prefix)])
    if not negative_columns:
        raise ValueError(f"No negative columns found for dataset {dataset_config['name']}")

    examples_per_query = 1 + negatives_per_query
    total_available_examples = len(split) * examples_per_query
    limits = {
        "validation": min(requested_limits["validation"], total_available_examples),
        "test": min(requested_limits["test"], max(total_available_examples - requested_limits["validation"], 0)),
    }
    remaining_for_train = max(total_available_examples - limits["validation"] - limits["test"], 0)
    limits["train"] = min(requested_limits["train"], remaining_for_train)

    output = {split_name: [] for split_name in TARGET_SPLITS}
    split_order = ["train", "validation", "test"]
    current_split_index = 0

    for row in split:
        while current_split_index < len(split_order) and len(output[split_order[current_split_index]]) >= limits[split_order[current_split_index]]:
            current_split_index += 1
        if current_split_index >= len(split_order):
            break

        split_name = split_order[current_split_index]
        append_example(
            output,
            split_name,
            dataset_config["name"],
            row[dataset_config["query_column"]],
            row[dataset_config["positive_column"]],
            1.0,
        )

        sampled_negative_columns = negative_columns[:negatives_per_query]
        for negative_column in sampled_negative_columns:
            if len(output[split_name]) >= limits[split_name]:
                break
            append_example(
                output,
                split_name,
                dataset_config["name"],
                row[dataset_config["query_column"]],
                row[negative_column],
                0.0,
            )

    return output


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare configurable multilingual similarity datasets.")
    parser.add_argument("--config", default="training/configs/multitask_default.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    random.seed(seed)

    output_dir = Path(args.output_dir or config.get("data_output_dir", "artifacts/data/multitask"))
    merged_rows = {split_name: [] for split_name in TARGET_SPLITS}
    metadata = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "datasets": [],
        "splits": {},
    }

    for dataset_config in config.get("datasets", []):
        if not dataset_config.get("enabled", True):
            continue

        if dataset_config["type"] == "pairs":
            dataset_rows = process_pair_dataset(dataset_config, seed)
        elif dataset_config["type"] == "triplets":
            dataset_rows = process_triplet_dataset(dataset_config, seed)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

        dataset_counts = {}
        for split_name, rows in dataset_rows.items():
            merged_rows[split_name].extend(rows)
            dataset_counts[split_name] = len(rows)

        metadata["datasets"].append(
            {
                "name": dataset_config["name"],
                "type": dataset_config["type"],
                "path": dataset_config["path"],
                "config": dataset_config.get("config"),
                "counts": dataset_counts,
            }
        )

    for split_name, rows in merged_rows.items():
        random.Random(seed).shuffle(rows)
        write_split(rows, output_dir / f"{split_name}.tsv")
        counts_by_dataset = defaultdict(int)
        for row in rows:
            counts_by_dataset[row["dataset_name"]] += 1

        metadata["splits"][split_name] = {
            "total_rows": len(rows),
            "by_dataset": dict(sorted(counts_by_dataset.items())),
        }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
