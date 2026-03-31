import argparse
import csv
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset


SPLIT_ALIASES = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "valid": "validation",
    "test": "test",
}


def normalize_split_name(name: str) -> str:
    return SPLIT_ALIASES.get(name.lower(), name.lower())


def detect_columns(split) -> tuple[str, str, str]:
    column_names = set(split.column_names)
    text1_candidates = ["sentence1", "query", "text1", "question1", "sentence_a"]
    text2_candidates = ["sentence2", "target", "text2", "question2", "sentence_b"]
    label_candidates = ["label", "score", "labels"]

    text1_col = next((name for name in text1_candidates if name in column_names), None)
    text2_col = next((name for name in text2_candidates if name in column_names), None)
    label_col = next((name for name in label_candidates if name in column_names), None)

    if not text1_col or not text2_col or not label_col:
        raise ValueError(f"Unsupported LCQMC schema: {split.column_names}")

    return text1_col, text2_col, label_col


def load_lcqmc(dataset_name: str, dataset_config: str | None) -> DatasetDict:
    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def export_split(split, output_path: Path) -> int:
    text1_col, text2_col, label_col = detect_columns(split)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_rows = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["text1", "text2", "label"])

        for row in split:
            text1 = str(row[text1_col]).strip()
            text2 = str(row[text2_col]).strip()
            if not text1 or not text2:
                continue

            label = int(row[label_col])
            if label not in (0, 1):
                raise ValueError(f"Expected binary labels, got {label}")

            writer.writerow([text1, text2, label])
            kept_rows += 1

    return kept_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LCQMC train/dev/test files.")
    parser.add_argument("--dataset-name", default="C-MTEB/LCQMC")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--output-dir", default="artifacts/data/lcqmc")
    args = parser.parse_args()

    dataset = load_lcqmc(args.dataset_name, args.dataset_config)
    output_dir = Path(args.output_dir)
    counts = {}

    for split_name, split in dataset.items():
        normalized = normalize_split_name(split_name)
        output_file = output_dir / f"{normalized}.tsv"
        counts[normalized] = export_split(split, output_file)

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "splits": counts,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
