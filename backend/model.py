from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    model_name_or_path: str
    description: str
    is_local: bool = False
    default: bool = False
    max_length: int = 256


class ModelNotReadyError(RuntimeError):
    pass


class ModelNotFoundError(KeyError):
    pass


class SimilarityModel:
    _instance = None

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.cache_dir = self.repo_root / "model_cache"
        self.models_dir = self.repo_root / "artifacts" / "models"
        self.cache_dir.mkdir(exist_ok=True)
        self.registry = self._build_registry()
        self.models: dict[str, CrossEncoder] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_registry(self) -> dict[str, ModelConfig]:
        finetuned_path = self._resolve_active_finetuned_path()
        return {
            "bge_zh": ModelConfig(
                model_id="bge_zh",
                name="BGE Chinese Baseline",
                model_name_or_path="BAAI/bge-reranker-large",
                description="当前中文强基线模型，作为服务默认模型。",
                default=True,
            ),
            "minilm_zero_shot": ModelConfig(
                model_id="minilm_zero_shot",
                name="MiniLM Zero-Shot",
                model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
                description="多语言 CrossEncoder 基座，未经过 LCQMC 中文微调。",
            ),
            "minilm_finetuned": ModelConfig(
                model_id="minilm_finetuned",
                name="MiniLM Fine-Tuned",
                model_name_or_path=str(finetuned_path),
                description="在可配置的中文/英文相似度数据集上混合微调后的多语言 CrossEncoder。",
                is_local=True,
            ),
        }

    def _resolve_active_finetuned_path(self) -> Path:
        active_model_file = self.models_dir / "active_model_name.txt"
        if active_model_file.exists():
            active_model_name = active_model_file.read_text(encoding="utf-8").strip()
            if active_model_name:
                return self.models_dir / active_model_name
        return self.models_dir / "msmarco-minilm-l6-multitask"

    def warmup_default_model(self) -> None:
        self._load_model(self.default_model_id)

    @property
    def default_model_id(self) -> str:
        for model_id, config in self.registry.items():
            if config.default:
                return model_id
        raise RuntimeError("No default model configured")

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        for config in self.registry.values():
            models.append(
                {
                    "model_id": config.model_id,
                    "name": config.name,
                    "description": config.description,
                    "default": config.default,
                    "supports_detailed": True,
                    "available": self._is_available(config),
                }
            )
        return models

    def predict_similarity(self, sentence1: str, sentence2: str, model_id: str | None = None) -> dict[str, Any]:
        config = self._get_config(model_id)
        model = self._load_model(config.model_id)
        score = model.predict([(sentence1, sentence2)])
        raw_score = float(score[0])
        return {
            "model_id": config.model_id,
            "score": raw_score,
            "normalized_score": self._normalize_score(raw_score),
        }

    def predict_with_attention(self, sentence1: str, sentence2: str, model_id: str | None = None) -> dict[str, Any]:
        config = self._get_config(model_id)
        model = self._load_model(config.model_id)

        inputs = model.tokenizer(
            [sentence1],
            [sentence2],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

        if hasattr(model.model, "device"):
            inputs = {key: value.to(model.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.model(**inputs, output_attentions=True, output_hidden_states=True)

        attentions = [att[0].mean(dim=0).detach().cpu().numpy().tolist() for att in outputs.attentions]
        layer_scores = [self._normalize_score(self._score_hidden_state(model, hidden_state)) for hidden_state in outputs.hidden_states[1:]]

        tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
        tokens = [token.replace("▁", "_") for token in tokens]

        raw_score = self._extract_logits(outputs.logits)
        return {
            "model_id": config.model_id,
            "score": raw_score,
            "normalized_score": self._normalize_score(raw_score),
            "tokens": tokens,
            "attentions": attentions,
            "layer_scores": layer_scores,
            "seq_length": len(tokens),
        }

    def _get_config(self, model_id: str | None) -> ModelConfig:
        resolved_model_id = model_id or self.default_model_id
        if resolved_model_id not in self.registry:
            raise ModelNotFoundError(resolved_model_id)
        return self.registry[resolved_model_id]

    def _is_available(self, config: ModelConfig) -> bool:
        if not config.is_local:
            return True
        model_dir = Path(config.model_name_or_path)
        return model_dir.exists() and (model_dir / "config.json").exists()

    def _load_model(self, model_id: str) -> CrossEncoder:
        if model_id in self.models:
            return self.models[model_id]

        config = self.registry[model_id]
        if config.is_local and not self._is_available(config):
            raise ModelNotReadyError(
                f"Model '{model_id}' is not available yet. Train it first at {config.model_name_or_path}."
            )

        model = CrossEncoder(config.model_name_or_path, cache_folder=str(self.cache_dir), max_length=config.max_length)
        model.predict([("warmup", "warmup")])
        self.models[model_id] = model
        return model

    @staticmethod
    def _extract_logits(logits: Any) -> float:
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu()
        if hasattr(logits, "reshape"):
            logits = logits.reshape(-1)
        return float(logits[0])

    def _score_hidden_state(self, model: CrossEncoder, hidden_state: torch.Tensor) -> float:
        classifier = getattr(model.model, "classifier", None)
        if classifier is None:
            return self._extract_logits(hidden_state[:, 0, 0])

        if isinstance(classifier, torch.nn.Linear):
            pooled_state = hidden_state[:, 0, :]
            dropout = getattr(model.model, "dropout", None)
            if dropout is not None:
                pooled_state = dropout(pooled_state)
            return self._extract_logits(classifier(pooled_state))

        try:
            logits = classifier(hidden_state)
        except Exception:
            logits = classifier(hidden_state[:, 0, :])

        return self._extract_logits(logits)

    @staticmethod
    def _normalize_score(score: float) -> float:
        if 0.0 <= score <= 1.0:
            return score
        return float(torch.sigmoid(torch.tensor(score)).item())
