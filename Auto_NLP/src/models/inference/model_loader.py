
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.models.base import MultiTaskModel
from src.training.configs.config import CommandConfig, EntityConfig, IntentConfig, ModelConfig


class ModelLoader:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        return torch.load(checkpoint_path, map_location=self.device)

    def load_tokenizer(self, tokenizer_dir: str, fallback_model: Optional[str] = None) -> AutoTokenizer:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception:
            if not fallback_model:
                raise
            return AutoTokenizer.from_pretrained(fallback_model)


class MultiTaskInference:
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str):
        self.loader = ModelLoader(Path(model_path).parent)
        self.checkpoint = self.loader.load_checkpoint(model_path)
        self.config = self.checkpoint.get("config") or {}

        fallback_model_name = self.config.get("model_name", ModelConfig.model_name)
        self.tokenizer: Any = self.loader.load_tokenizer(
            tokenizer_path, fallback_model=fallback_model_name
        )
        if not self.config:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                self.config = {"model_name": fallback_model_name}

        self.intent_labels = self.config.get("intent_labels", IntentConfig().intent_labels)
        self.entity_labels = self.config.get("entity_labels", EntityConfig().entity_labels)
        self.command_labels = self.config.get("command_labels", CommandConfig().command_labels)

        self.id_to_intent = dict(enumerate(self.intent_labels))
        self.id_to_entity = dict(enumerate(self.entity_labels))
        self.id_to_command = dict(enumerate(self.command_labels))

        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> MultiTaskModel:
        model_name = self.config.get("model_name", ModelConfig.model_name)
        dropout = self.config.get("dropout", ModelConfig.dropout)

        model = MultiTaskModel(
            model_name=model_name,
            num_intents=len(self.intent_labels),
            num_entity_labels=len(self.entity_labels),
            num_commands=len(self.command_labels),
            dropout=dropout,
            use_safetensors=self.config.get("use_safetensors", True),
        )

        state_dict = self.checkpoint.get("model_state") or self.checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = self.checkpoint
        if state_dict is None:
            raise KeyError("Checkpoint không chứa model_state hoặc model_state_dict.")

        keys_to_drop = [k for k in list(state_dict.keys()) if k.startswith("entity_mask")]
        for k in keys_to_drop:
            state_dict.pop(k, None)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"⚠️ Unexpected keys ignored when loading model: {unexpected}")
        if missing:
            print(f"⚠️ Missing keys when loading model (using default init): {missing}")
        model.to(self.loader.device)
        return model

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.get("max_length", ModelConfig.max_length),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.loader.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

        intent_logits = outputs["intent_logits"]
        intent_probabilities = torch.softmax(intent_logits, dim=-1)[0]
        intent_id = int(torch.argmax(intent_probabilities).item())
        intent_confidence = intent_probabilities[intent_id].item()

        command_logits = outputs["command_logits"]
        command_probabilities = torch.softmax(command_logits, dim=-1)[0]
        command_id = int(torch.argmax(command_probabilities).item())
        command_confidence = command_probabilities[command_id].item()

        entity_predictions = outputs["entity_logits"].argmax(dim=-1)[0].cpu().tolist()
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        entity_labels = [self.id_to_entity[idx] for idx in entity_predictions]
        entities = self._decode_entities(tokens, entity_labels, None, text)

        return {
            "text": text,
            "intent": self.id_to_intent.get(intent_id, f"unknown_{intent_id}"),
            "intent_confidence": intent_confidence,
            "command": self.id_to_command.get(command_id, f"unknown_{command_id}"),
            "command_confidence": command_confidence,
            "entities": entities,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(text) for text in texts]

    def _decode_entities(
        self,
        tokens: List[str],
        labels: List[str],
        offsets: Optional[List[Tuple[int, int]]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        current_tokens: List[str] = []
        current_label: Optional[str] = None

        special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "[PAD]", "[CLS]", "[SEP]"}

        for idx, (token, label) in enumerate(zip(tokens, labels)):
            if token in special_tokens:
                if current_label is not None:
                    text = self._tokens_to_text(current_tokens, original_text, offsets, tokens, idx)
                    if text:
                        entities.append({"label": current_label, "text": text})
                current_tokens = []
                current_label = None
                continue
            if label == "O":
                if current_label is not None:
                    text = self._tokens_to_text(current_tokens, original_text, offsets, tokens, idx)
                    if text:
                        entities.append({"label": current_label, "text": text})
                    current_tokens = []
                    current_label = None
                continue

            prefix, _, entity_type = label.partition("-")

            if prefix == "B" or (prefix == "I" and current_label != entity_type):
                if current_label is not None:
                    text = self._tokens_to_text(current_tokens, original_text, offsets, tokens, idx)
                    if text:
                        entities.append({"label": current_label, "text": text})
                current_label = entity_type
                current_tokens = [token]
            elif prefix == "I" and current_label == entity_type:
                current_tokens.append(token)
            else:
                if current_label is not None:
                    text = self._tokens_to_text(current_tokens, original_text, offsets, tokens, idx)
                    if text:
                        entities.append({"label": current_label, "text": text})
                current_label = None
                current_tokens = []

        if current_label is not None and current_tokens:
            text = self._tokens_to_text(current_tokens, original_text, offsets, tokens, len(tokens))
            if text:
                entities.append({"label": current_label, "text": text})

        return entities

    @staticmethod
    def _tokens_to_text(
        entity_tokens: List[str],
        original_text: str,
        offsets: Optional[List[Tuple[int, int]]],
        all_tokens: List[str],
        end_idx: int,
    ) -> str:
        if not entity_tokens:
            return ""

        if offsets is not None:
            start_char = None
            end_char = None
            start_token_idx = max(0, end_idx - len(entity_tokens))
            end_token_idx = end_idx - 1
            for i in range(start_token_idx, end_token_idx + 1):
                s, e = offsets[i]
                if start_char is None or s < start_char:
                    start_char = s
                if end_char is None or e > end_char:
                    end_char = e
            if start_char is None or end_char is None or start_char >= end_char:
                return ""
            return original_text[start_char:end_char].strip()

        cleaned_tokens: List[str] = []
        for tok in entity_tokens:
            if tok in {"<s>", "</s>", "<pad>", "<unk>", "[PAD]", "[CLS]", "[SEP]"}:
                continue
            cleaned_tokens.append(tok.replace("▁", " ").replace("@@", "").strip())

        if not cleaned_tokens:
            return ""

        text = " ".join(cleaned_tokens)
        text = " ".join(text.split())
        return text


def _select_checkpoint(model_dir: Path) -> Path:
    model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No checkpoint file found in {model_dir}")

    for model_file in model_files:
        if "best" in model_file.name:
            return model_file
    return model_files[0]


def load_multi_task_model(model_name: str = "phobert_large_intent_model") -> MultiTaskInference:
    model_dir = Path("models/trained") / model_name
    if not model_dir.exists():
        fallback_dir = Path("models") / model_name
        if fallback_dir.exists():
            model_dir = fallback_dir
        else:
            raise FileNotFoundError(f"Model directory not found in models/trained or models: {model_name}")

    checkpoint_path = _select_checkpoint(model_dir)
    tokenizer_path = str(model_dir)
    config_path = model_dir / "config.json"

    return MultiTaskInference(str(checkpoint_path), tokenizer_path, str(config_path))


def load_trained_model(model_name: str = "phobert_large_intent_model") -> MultiTaskInference:
    return load_multi_task_model(model_name)

