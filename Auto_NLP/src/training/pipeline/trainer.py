from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


@dataclass
class TrainingOutcome:
    best_epoch: int
    best_val_metric: float
    history: List[Dict[str, float]] = field(default_factory=list)
    best_checkpoint_path: Optional[Path] = None


class MultitaskTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        config,
        processor,
        output_dir: str | os.PathLike[str],
        gradient_accumulation_steps: Optional[int] = None,
        freeze_encoder_epochs: Optional[int] = None,
        entity_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and getattr(config, "use_cuda", True) else "cpu")
        self.model = model.to(self.device)
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        self.intent_id2label = processor.intent_id2label
        self.command_id2label = processor.command_id2label
        self.entity_id2label = processor.entity_id2label
        self.entity_label2id = processor.entity_label2id

        self.grad_accum = gradient_accumulation_steps or getattr(config, "gradient_accumulation_steps", 1)
        self.lambda_intent = getattr(config, "LAMBDA_INTENT", 0.45)
        self.lambda_entity = getattr(config, "LAMBDA_ENTITY", 0.25)
        self.lambda_command = getattr(config, "LAMBDA_COMMAND", 0.2)
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)

        self.intent_criterion = nn.CrossEntropyLoss()
        self.command_criterion = nn.CrossEntropyLoss()
        if entity_class_weights is not None:
            self.entity_criterion = nn.CrossEntropyLoss(weight=entity_class_weights.to(self.device), ignore_index=-100)
        else:
            self.entity_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.scaler: Optional[GradScaler] = None
        self.autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        use_mixed_precision = getattr(config, "use_mixed_precision", True)
        if use_mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
            self.logger.info("✅ Mixed Precision (FP16) enabled - using GradScaler")
        elif self.device.type == "cuda":
            self.logger.info("⚠️ Mixed Precision (FP16) disabled - using FP32")
        else:
            self.logger.info("ℹ️ CPU mode - Mixed Precision not available")

        self.freeze_encoder_epochs = max(0, freeze_encoder_epochs or getattr(config, "freeze_encoder_epochs", 0) or 0)
        self._encoder_frozen = False
        if self.freeze_encoder_epochs > 0:
            self._set_encoder_trainable(False)
            self._encoder_frozen = True
            self.logger.info("🔒 Đóng băng encoder trong %d epoch đầu.", self.freeze_encoder_epochs)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return
        for param in encoder.parameters():
            param.requires_grad = trainable
        encoder.train(trainable and self.model.training)

    def _maybe_update_encoder_freeze(self, epoch: int) -> None:
        if self.freeze_encoder_epochs <= 0:
            return
        if epoch <= self.freeze_encoder_epochs:
            if not self._encoder_frozen:
                self._set_encoder_trainable(False)
                self._encoder_frozen = True
                self.logger.info("🔒 Giữ encoder đóng băng ở epoch %d.", epoch)
        elif self._encoder_frozen:
            self._set_encoder_trainable(True)
            self._encoder_frozen = False
            self.logger.info("🔓 Mở khóa encoder từ epoch %d.", epoch)

    def _setup_optimizer(self) -> Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": getattr(self.config, "weight_decay", 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=getattr(self.config, "learning_rate", 2e-5),
            betas=getattr(self.config, "adam_betas", (0.9, 0.999)),
            eps=getattr(self.config, "adam_epsilon", 1e-8),
        )
        return optimizer

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, train_loader: DataLoader, num_epochs: int
    ) -> LambdaLR:
        total_update_steps = math.ceil(len(train_loader) / self.grad_accum) * num_epochs
        configured_warmup = getattr(self.config, "warmup_steps", 500)
        warmup_steps = min(configured_warmup, max(1, total_update_steps // 10))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
        return scheduler

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        evaluation_loader: Optional[DataLoader] = None,
    ) -> TrainingOutcome:
        epochs = num_epochs or getattr(self.config, "num_epochs", 6)
        optimizer = self._setup_optimizer()
        scheduler = self._build_scheduler(optimizer, train_loader, epochs)

        history: List[Dict[str, float]] = []
        best_val_metric = float("-inf")
        best_epoch = -1
        best_path: Optional[Path] = None

        for epoch in range(1, epochs + 1):
            self._maybe_update_encoder_freeze(epoch)
            train_metrics = self._train_one_epoch(train_loader, optimizer, scheduler, epoch)
            val_metrics = self.evaluate(val_loader)
            epoch_record = {
                "epoch": float(epoch),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_record)

            intent_score = val_metrics.get("intent_macro_f1", val_metrics.get("intent_weighted_f1", 0.0))
            command_score = val_metrics.get("command_macro_f1", val_metrics.get("command_weighted_f1", 0.0))
            entity_score = val_metrics.get("entity_f1", 0.0)

            val_score = (
                intent_score * self.lambda_intent
                + entity_score * self.lambda_entity
                + command_score * self.lambda_command
            )

            self.logger.info(
                "Epoch %d - Val score (weighted): %.4f | intent=%.4f, entity=%.4f, command=%.4f",
                epoch,
                val_score,
                intent_score,
                entity_score,
                command_score,
            )

            if val_score > best_val_metric:
                best_val_metric = val_score
                best_epoch = epoch
                best_path = self._save_checkpoint(epoch, val_metrics, is_best=True)

            else:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        if evaluation_loader is not None:
            self.evaluate(evaluation_loader, split="test")

        return TrainingOutcome(best_epoch=best_epoch, best_val_metric=best_val_metric, history=history, best_checkpoint_path=best_path)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_intent_loss = 0.0
        total_entity_loss = 0.0
        total_command_loss = 0.0
        total_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            total=len(train_loader),
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            loss_dict = self._compute_loss(batch, training=True, return_outputs=False)
            loss = loss_dict["total_loss"] / self.grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss_dict["total_loss"].item()
            total_intent_loss += loss_dict["intent_loss"].item()
            total_entity_loss += loss_dict["entity_loss"].item()
            total_command_loss += loss_dict["command_loss"].item()
            total_steps += 1

            if step % 20 == 0:
                progress.set_postfix(
                    loss=f"{total_loss / max(total_steps, 1):.4f}",
                    intent=f"{total_intent_loss / max(total_steps, 1):.4f}",
                    entity=f"{total_entity_loss / max(total_steps, 1):.4f}",
                    command=f"{total_command_loss / max(total_steps, 1):.4f}",
                )

        return {
            "loss": total_loss / max(total_steps, 1),
            "intent_loss": total_intent_loss / max(total_steps, 1),
            "entity_loss": total_entity_loss / max(total_steps, 1),
            "command_loss": total_command_loss / max(total_steps, 1),
        }

    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool,
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        intent_labels = batch["intent_labels"].to(self.device)
        entity_labels = batch["entity_labels"].to(self.device)
        command_labels = batch["command_labels"].to(self.device)

        with autocast(device_type=self.autocast_device, enabled=self.scaler is not None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            intent_logits = outputs["intent_logits"]
            entity_logits = outputs["entity_logits"]
            command_logits = outputs["command_logits"]

            intent_loss = self.intent_criterion(intent_logits, intent_labels)
            command_loss = self.command_criterion(command_logits, command_labels)

            entity_loss = self._entity_loss(entity_logits, entity_labels, attention_mask)

            total_loss = (
                self.lambda_intent * intent_loss
                + self.lambda_entity * entity_loss
                + self.lambda_command * command_loss
            )

        result: Dict[str, torch.Tensor] = {
            "total_loss": total_loss,
            "intent_loss": intent_loss.detach(),
            "entity_loss": entity_loss.detach(),
            "command_loss": command_loss.detach(),
        }

        if return_outputs:
            result["intent_logits"] = intent_logits.detach()
            result["entity_logits"] = entity_logits.detach()
            result["command_logits"] = command_logits.detach()

        return result

    def _entity_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, num_labels = logits.shape
        logits = logits.view(-1, num_labels)
        labels = labels.view(-1)
        loss = self.entity_criterion(logits, labels)
        return loss

    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_intent_loss = 0.0
        total_entity_loss = 0.0
        total_command_loss = 0.0
        total_steps = 0

        all_intent_preds: List[int] = []
        all_intent_labels: List[int] = []
        all_command_preds: List[int] = []
        all_command_labels: List[int] = []
        entity_pred_labels: List[List[str]] = []
        entity_true_labels: List[List[str]] = []

        with torch.no_grad():
            for batch in loader:
                loss_dict = self._compute_loss(batch, training=False, return_outputs=True)
                total_loss += loss_dict["total_loss"].item()
                total_intent_loss += loss_dict["intent_loss"].item()
                total_entity_loss += loss_dict["entity_loss"].item()
                total_command_loss += loss_dict["command_loss"].item()
                total_steps += 1

                intent_logits = loss_dict["intent_logits"].cpu()
                command_logits = loss_dict["command_logits"].cpu()
                entity_logits = loss_dict["entity_logits"].cpu()

                intent_preds = intent_logits.argmax(dim=-1).cpu().tolist()
                command_preds = command_logits.argmax(dim=-1).cpu().tolist()
                intents = batch["intent_labels"].tolist()
                commands = batch["command_labels"].tolist()

                all_intent_preds.extend(intent_preds)
                all_intent_labels.extend(intents)
                all_command_preds.extend(command_preds)
                all_command_labels.extend(commands)

                pred_entities, true_entities = self._collect_entity_sequences(
                    entity_logits, batch["entity_labels"], batch["attention_mask"]
                )
                entity_pred_labels.extend(pred_entities)
                entity_true_labels.extend(true_entities)

        metrics = {
            "loss": total_loss / max(total_steps, 1),
            "intent_loss": total_intent_loss / max(total_steps, 1),
            "entity_loss": total_entity_loss / max(total_steps, 1),
            "command_loss": total_command_loss / max(total_steps, 1),
        }

        if all_intent_labels:
            metrics.update(self._classification_metrics(all_intent_labels, all_intent_preds, prefix="intent"))

        if all_command_labels:
            metrics.update(self._classification_metrics(all_command_labels, all_command_preds, prefix="command"))

        if entity_true_labels:
            metrics.update(self._entity_metrics(entity_true_labels, entity_pred_labels))
            total_tokens = sum(len(seq) for seq in entity_true_labels)
            pred_non_o = sum(
                sum(1 for label in pred_seq if label != "O") for pred_seq in entity_pred_labels
            )
            true_non_o = sum(
                sum(1 for label in true_seq if label != "O") for true_seq in entity_true_labels
            )
            metrics["entity_pred_non_o_ratio"] = (
                pred_non_o / max(total_tokens, 1)
            )
            metrics["entity_true_non_o_ratio"] = (
                true_non_o / max(total_tokens, 1)
            )

        return metrics

    def _classification_metrics(
        self, labels: List[int], preds: List[int], prefix: str
    ) -> Dict[str, float]:
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        precision = precision_score(labels, preds, average="macro", zero_division="warn")
        recall = recall_score(labels, preds, average="macro", zero_division="warn")

        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_weighted_f1": weighted_f1,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
        }

    def _entity_metrics(
        self, true_labels: List[List[str]], pred_labels: List[List[str]]
    ) -> Dict[str, float]:
        precision = seqeval_precision_score(true_labels, pred_labels)
        recall = seqeval_recall_score(true_labels, pred_labels)
        f1 = seqeval_f1_score(true_labels, pred_labels)
        # seqeval metrics can return float or List[float], ensure we return float
        precision_val = float(precision) if isinstance(precision, (int, float)) else float(precision[0]) if isinstance(precision, list) and len(precision) > 0 else 0.0
        recall_val = float(recall) if isinstance(recall, (int, float)) else float(recall[0]) if isinstance(recall, list) and len(recall) > 0 else 0.0
        f1_val = float(f1) if isinstance(f1, (int, float)) else float(f1[0]) if isinstance(f1, list) and len(f1) > 0 else 0.0
        return {
            "entity_precision": precision_val,
            "entity_recall": recall_val,
            "entity_f1": f1_val,
        }

    def _collect_entity_sequences(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        pred_ids = logits.argmax(dim=-1).tolist()
        true_ids = labels.tolist()
        masks = attention_mask.tolist()
        pred_sequences: List[List[str]] = []
        true_sequences: List[List[str]] = []

        for pred_seq, true_seq, mask_seq in zip(pred_ids, true_ids, masks):
            length = int(sum(mask_seq))
            if length <= 2:
                continue
            start = 1
            end = max(length - 1, start)

            pred_labels = [self.entity_id2label[pred_seq[i]] for i in range(start, end)]
            true_labels = [self.entity_id2label[true_seq[i]] for i in range(start, end)]

            pred_sequences.append(pred_labels)
            true_sequences.append(true_labels)

        return pred_sequences, true_sequences

    def _save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool
    ) -> Optional[Path]:
        filename = f"checkpoint-epoch{epoch}.pt"
        path = self.output_dir / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "metrics": metrics,
                "config": self._export_config(),
            },
            path,
        )

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_path)
            return best_path

        return None

    def _export_config(self) -> Dict[str, object]:
        items: Dict[str, object] = {}
        for attr in dir(self.config):
            if attr.startswith("_"):
                continue
            value = getattr(self.config, attr)
            if callable(value):
                continue
            items[attr] = value
        return items

