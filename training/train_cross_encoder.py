import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CECorrelationEvaluator
from torch.utils.data import DataLoader


def read_examples(path: Path) -> list[InputExample]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            examples.append(
                InputExample(
                    texts=[row["text1"], row["text2"]],
                    label=float(row["label"]),
                )
            )
    return examples


def export_cross_encoder(model: CrossEncoder, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_method = getattr(model, "save", None)
    if callable(save_method):
        try:
            save_method(str(output_dir))
            return
        except Exception:
            pass

    model.model.save_pretrained(str(output_dir))
    model.tokenizer.save_pretrained(str(output_dir))


def is_binary_labels(examples: list[InputExample]) -> bool:
    return all(example.label in (0.0, 1.0) for example in examples)


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


def persist_experiment_config(config_path: str | None, config_payload: dict, output_dir: Path) -> None:
    destination = output_dir / "experiment_config.json"
    if config_path:
        shutil.copyfile(config_path, destination)
        return
    destination.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_active_model_name(model_name: str) -> None:
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "active_model_name.txt").write_text(f"{model_name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a multilingual CrossEncoder on configurable datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-file", default="artifacts/data/lcqmc/train.tsv")
    parser.add_argument("--dev-file", default="artifacts/data/lcqmc/validation.tsv")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="model_cache")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    model_name = resolve_arg(None, config.get("model_name"), "msmarco-minilm-l6-multitask")

    dataset_dir_arg = args.dataset_dir
    if dataset_dir_arg is None:
        dataset_dir_arg = config.get("data_output_dir")

    base_model = resolve_arg(args.base_model, training_config.get("base_model"), "cross-encoder/ms-marco-MiniLM-L-6-v2")
    output_dir_arg = resolve_arg(args.output_dir, None, derive_model_output_dir(model_name))
    epochs = int(resolve_arg(args.epochs, training_config.get("epochs"), 3))
    train_batch_size = int(resolve_arg(args.train_batch_size, training_config.get("train_batch_size"), 32))
    eval_batch_size = int(resolve_arg(args.eval_batch_size, training_config.get("eval_batch_size"), 64))
    max_length = int(resolve_arg(args.max_length, training_config.get("max_length"), 256))
    learning_rate = float(resolve_arg(args.learning_rate, training_config.get("learning_rate"), 2e-5))
    warmup_ratio = float(resolve_arg(args.warmup_ratio, training_config.get("warmup_ratio"), 0.1))

    train_file = Path(args.train_file)
    dev_file = Path(args.dev_file)
    if dataset_dir_arg:
        dataset_dir = Path(dataset_dir_arg)
        train_file = dataset_dir / "train.tsv"
        dev_file = dataset_dir / "validation.tsv"

    train_examples = read_examples(train_file)
    dev_examples = read_examples(dev_file)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

    if is_binary_labels(dev_examples):
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            dev_examples,
            name="binary-dev",
            batch_size=eval_batch_size,
        )
        evaluator_name = "CEBinaryClassificationEvaluator"
    else:
        evaluator = CECorrelationEvaluator.from_input_examples(
            dev_examples,
            name="correlation-dev",
            batch_size=eval_batch_size,
        )
        evaluator_name = "CECorrelationEvaluator"

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=max_length,
        cache_folder=args.cache_dir,
    )

    output_dir = Path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "config": args.config,
        "model_name": model_name,
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "dataset_dir": dataset_dir_arg,
        "base_model": base_model,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "evaluator": evaluator_name,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    persist_experiment_config(args.config, config, output_dir)
    persist_active_model_name(model_name)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        save_best_model=True,
        use_amp=torch.cuda.is_available(),
        show_progress_bar=True,
    )

    # Some sentence-transformers versions do not persist the final/best model to
    # output_path when using CrossEncoder.fit. Export explicitly for downstream evaluation.
    export_cross_encoder(model, output_dir)


if __name__ == "__main__":
    main()
