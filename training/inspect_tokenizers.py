import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder


DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-large",
]

DEFAULT_SENTENCES = [
    "你太好了",
    "你真好",
    "今天天气不错",
    "我喜欢吃苹果",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tokenizer behavior for Chinese text across reranker models.")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--sentence", action="append", default=[])
    parser.add_argument("--cache-dir", default="model_cache")
    return parser.parse_args()


def inspect_model(model_name: str, sentences: list[str], cache_dir: str) -> None:
    print(f"=== {model_name} ===")
    model = CrossEncoder(model_name, cache_folder=cache_dir)
    tokenizer = model.tokenizer

    vocab = tokenizer.get_vocab()
    chinese_sentence_ids = []

    for sentence in sentences:
        encoded = tokenizer(sentence, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        chinese_sentence_ids.extend(input_ids)

        print(f"sentence: {sentence}")
        print(f"token_count: {len(tokens)}")
        print(f"tokens: {tokens}")
        print(f"ids: {input_ids}")
        print()

    unique_ids = len(set(chinese_sentence_ids))
    chinese_tokens_in_vocab = sum(1 for token in vocab if contains_cjk(token))
    print(f"summary.unique_token_ids: {unique_ids}")
    print(f"summary.vocab_tokens_with_cjk: {chinese_tokens_in_vocab}")
    print()


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def main() -> None:
    args = parse_args()
    models = args.model or DEFAULT_MODELS
    sentences = args.sentence or DEFAULT_SENTENCES

    for model_name in models:
        inspect_model(model_name, sentences, args.cache_dir)


if __name__ == "__main__":
    main()
