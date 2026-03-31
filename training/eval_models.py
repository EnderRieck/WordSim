import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import CrossEncoder


DEFAULT_MODELS = {
    "bge_zh": "BAAI/bge-reranker-large",
    "minilm_zero_shot": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm_finetuned": "artifacts/models/msmarco-minilm-l6-multitask",
}


def read_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "dataset_name": row.get("dataset_name", "unknown"),
                    "text1": row["text1"],
                    "text2": row["text2"],
                    "label": float(row["label"]),
                }
            )
    return rows


def build_pairs_and_labels(rows: list[dict]) -> tuple[list[tuple[str, str]], list[float]]:
    pairs = []
    labels = []
    for row in rows:
        pairs.append((row["text1"], row["text2"]))
        labels.append(float(row["label"]))
    return pairs, labels


def normalize_scores(raw_scores) -> list[float]:
    normalized = []
    for score in raw_scores:
        score = float(score)
        if 0.0 <= score <= 1.0:
            normalized.append(score)
        else:
            normalized.append(1.0 / (1.0 + math.exp(-score)))
    return normalized


def evaluate_model(model_name_or_path: str, pairs, labels, cache_dir: str, batch_size: int) -> dict:
    model = CrossEncoder(model_name_or_path, cache_folder=cache_dir)
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    scores = normalize_scores(raw_scores)
    return compute_metrics(labels, scores)


def is_binary_label(value: float) -> bool:
    return value in (0.0, 1.0)


def safe_round(value):
    if value is None:
        return None
    return round(float(value), 4)


def format_metric(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def compute_metrics(labels: list[float], scores: list[float]) -> dict:
    metrics = {}

    if len(labels) >= 2 and len(set(labels)) > 1:
        metrics["pearson"] = safe_round(pearsonr(labels, scores).statistic)
        metrics["spearman"] = safe_round(spearmanr(labels, scores).statistic)
    else:
        metrics["pearson"] = None
        metrics["spearman"] = None

    binary_indices = [index for index, label in enumerate(labels) if is_binary_label(label)]
    if binary_indices:
        binary_labels = [int(labels[index]) for index in binary_indices]
        binary_scores = [scores[index] for index in binary_indices]
        predictions = [1 if score >= 0.5 else 0 for score in binary_scores]
        metrics["binary_count"] = len(binary_indices)
        metrics["accuracy"] = safe_round(accuracy_score(binary_labels, predictions))
        metrics["f1"] = safe_round(f1_score(binary_labels, predictions))
        metrics["auc"] = safe_round(roc_auc_score(binary_labels, binary_scores)) if len(set(binary_labels)) > 1 else None
    else:
        metrics["binary_count"] = 0
        metrics["accuracy"] = None
        metrics["f1"] = None
        metrics["auc"] = None

    return metrics


def validate_model_path(model_name_or_path: str) -> None:
    path = Path(model_name_or_path)
    if not path.exists() or not path.is_dir():
        return

    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_name_or_path}' is incomplete: missing config.json. "
            "Re-run training so the fine-tuned model is exported correctly."
        )


def parse_model_args(model_args: list[str]) -> dict[str, str]:
    if not model_args:
        return DEFAULT_MODELS

    parsed = {}
    for item in model_args:
        model_id, model_path = item.split("=", 1)
        parsed[model_id] = model_path
    return parsed


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_arg(cli_value, config_value, fallback):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback


def derive_model_output_dir(model_name: str) -> str:
    return str(Path("artifacts/models") / model_name)


def derive_report_name(model_name: str) -> str:
    return f"{model_name}_comparison"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple CrossEncoder models on configurable test sets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--test-file", default="artifacts/data/lcqmc/test.tsv")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--cache-dir", default="model_cache")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation_config = config.get("evaluation", {})
    model_name = resolve_arg(None, config.get("model_name"), "msmarco-minilm-l6-multitask")
    models = parse_model_args(args.model)
    if not args.model and "minilm_finetuned" in models:
        models["minilm_finetuned"] = derive_model_output_dir(model_name)
    test_file = Path(args.test_file)
    dataset_dir_arg = resolve_arg(args.dataset_dir, config.get("data_output_dir"), None)
    if dataset_dir_arg:
        test_file = Path(dataset_dir_arg) / "test.tsv"
    report_dir_arg = resolve_arg(args.report_dir, evaluation_config.get("report_dir"), "artifacts/reports")
    report_name = resolve_arg(args.report_name, None, derive_report_name(model_name))

    rows = read_dataset(test_file)
    pairs, labels = build_pairs_and_labels(rows)
    dataset_names = sorted({row["dataset_name"] for row in rows})
    report = {
        "test_file": str(test_file),
        "models": {},
    }

    for model_id, model_path in models.items():
        validate_model_path(model_path)
        model = CrossEncoder(model_path, cache_folder=args.cache_dir)
        raw_scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
        scores = normalize_scores(raw_scores)
        per_dataset = {}
        for dataset_name in dataset_names:
            dataset_rows = [row for row in rows if row["dataset_name"] == dataset_name]
            _, dataset_labels = build_pairs_and_labels(dataset_rows)
            dataset_scores = [scores[index] for index, row in enumerate(rows) if row["dataset_name"] == dataset_name]
            per_dataset[dataset_name] = {
                "count": len(dataset_rows),
                "metrics": compute_metrics(dataset_labels, dataset_scores),
            }

        report["models"][model_id] = {
            "path": model_path,
            "metrics": compute_metrics(labels, scores),
            "per_dataset": per_dataset,
        }

    report_dir = Path(report_dir_arg)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{report_name}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Dataset Comparison",
        "",
        "| Model ID | Model Path | Pearson | Spearman | Accuracy | F1 | AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id, payload in report["models"].items():
        metrics = payload["metrics"]
        markdown_lines.append(
            f"| {model_id} | {payload['path']} | {format_metric(metrics['pearson'])} | {format_metric(metrics['spearman'])} | {format_metric(metrics['accuracy'])} | {format_metric(metrics['f1'])} | {format_metric(metrics['auc'])} |"
        )

    markdown_lines.extend(["", "## Per Dataset", ""])
    for model_id, payload in report["models"].items():
        markdown_lines.append(f"### {model_id}")
        markdown_lines.append("")
        markdown_lines.append("| Dataset | Count | Pearson | Spearman | Accuracy | F1 | AUC |")
        markdown_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for dataset_name, dataset_payload in payload["per_dataset"].items():
            metrics = dataset_payload["metrics"]
            markdown_lines.append(
                f"| {dataset_name} | {dataset_payload['count']} | {format_metric(metrics['pearson'])} | {format_metric(metrics['spearman'])} | {format_metric(metrics['accuracy'])} | {format_metric(metrics['f1'])} | {format_metric(metrics['auc'])} |"
            )
        markdown_lines.append("")

    markdown_path = report_dir / f"{report_name}.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
