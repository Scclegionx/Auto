from __future__ import annotations

import argparse
import json
import logging
import sys
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.processed.data_processor import DataProcessor
from models.base.multitask_model import MultiTaskModel
from training.configs.config import CommandConfig, EntityConfig, IntentConfig, ModelConfig
from training.datasets import MultiTaskDataset, create_dataloader
from training.pipeline import MultitaskTrainer

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "training.log"

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def load_datasets(processor: DataProcessor, config: ModelConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_raw = processor.load_dataset(config.train_data_path)
    val_raw = processor.load_dataset(config.val_data_path)
    test_raw = processor.load_dataset(config.test_data_path)
    return train_raw, val_raw, test_raw


def sample_subset(
    samples: List[Dict],
    ratio: float,
    max_samples: int | None,
    seed: int,
    label_key: str = "command",
) -> List[Dict]:
    if not samples:
        return samples

    ratio = max(0.0, min(ratio, 1.0))
    total = len(samples)
    if ratio >= 1.0 and (max_samples is None or max_samples >= total):
        return samples

    rng = random.Random(seed)
    target = total
    if ratio > 0:
        target = max(1, int(total * ratio))
    if max_samples is not None:
        target = min(target, max_samples)
    target = max(1, target)

    if target >= total:
        return samples

    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        grouped.setdefault(sample.get(label_key, "unknown"), []).append(sample)

    selected: List[Dict] = []
    for label, group in grouped.items():
        take = max(1, int(len(group) * target / total))
        if take >= len(group):
            selected.extend(group)
        else:
            selected.extend(rng.sample(group, take))

    if len(selected) > target:
        rng.shuffle(selected)
        selected = selected[:target]
    elif len(selected) < target:
        remaining = [item for group in grouped.values() for item in group if item not in selected]
        rng.shuffle(remaining)
        needed = target - len(selected)
        selected.extend(remaining[:needed])

    return selected


def summarize_commands(name: str, samples: List[Dict]) -> None:
    counter = Counter(sample.get("command", "unknown") for sample in samples)
    summary = ", ".join(f"{cmd}: {count}" for cmd, count in counter.most_common())
    logger = logging.getLogger("training")
    logger.info("%s (%d): %s", name, len(samples), summary)


def _format_top_counts(counter: Counter, top_k: int = 5) -> str:
    if not counter:
        return "∅"
    return ", ".join(f"{label}:{count}" for label, count in counter.most_common(top_k))


def log_dataset_statistics(name: str, samples: List[Dict], processor: DataProcessor) -> None:
    logger = logging.getLogger("training")
    if not samples:
        logger.warning("%s rỗng.", name)
        return

    intent_counter: Counter = Counter()
    command_counter: Counter = Counter()
    entity_counter: Counter = Counter()

    total_tokens = 0
    total_non_o = 0
    o_label_id = processor.entity_label2id.get("O", 0)

    for sample in samples:
        intent_id = sample.get("intent_label")
        command_id = sample.get("command_label")
        if intent_id is not None:
            intent_label = processor.intent_id2label.get(int(intent_id), f"unknown_{intent_id}")
            intent_counter[intent_label] += 1
        if command_id is not None:
            command_label = processor.command_id2label.get(int(command_id), f"unknown_{command_id}")
            command_counter[command_label] += 1

        raw_entity_labels = sample.get("entity_labels", [])
        valid_labels = [label for label in raw_entity_labels if label != IGNORE_INDEX]
        total_tokens += len(valid_labels)
        non_o = sum(1 for label in valid_labels if label != o_label_id)
        total_non_o += non_o
        for label in valid_labels:
            entity_label = processor.entity_id2label.get(int(label), f"unknown_{label}")
            entity_counter[entity_label] += 1

    avg_tokens = total_tokens / len(samples)
    avg_non_o = total_non_o / len(samples)
    non_o_ratio = (total_non_o / total_tokens) if total_tokens else 0.0

    logger.info(
        "%s - Intent phân bố: %s",
        name,
        _format_top_counts(intent_counter),
    )
    logger.info(
        "%s - Command phân bố: %s",
        name,
        _format_top_counts(command_counter),
    )
    logger.info(
        "%s - Entity tokens (avg_len=%.2f | avg_non_O=%.2f | non_O_ratio=%.4f): %s",
        name,
        avg_tokens,
        avg_non_o,
        non_o_ratio,
        _format_top_counts(entity_counter),
    )


def prepare_multitask_samples(processor: DataProcessor, dataset: Iterable[Dict]) -> List[Dict]:
    return processor.prepare_multi_task_data(list(dataset))


def compute_entity_class_weights(
    samples: Iterable[Dict],
    num_labels: int,
    o_label_id: int,
    min_weight: float = 0.3,
    max_weight: float = 3.0,
    o_weight: float = 0.2,
) -> torch.Tensor | None:
    counts = np.zeros(num_labels, dtype=np.float64)

    for sample in samples:
        for label in sample.get("entity_labels", []):
            if label == IGNORE_INDEX or not 0 <= label < num_labels:
                continue
            counts[label] += 1.0

    valid_mask = counts > 0
    if not valid_mask.any():
        return None

    min_positive = counts[valid_mask].min()
    counts[~valid_mask] = min_positive

    total = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    if 0 <= o_label_id < len(weights):
        weights[o_label_id] = np.clip(o_weight, min_weight, max_weight)

    return torch.tensor(weights, dtype=torch.float)


def compute_entity_density_sampler_weights(
    samples: Iterable[Dict],
    o_label_id: int,
    min_weight: float = 0.9,
    max_weight: float = 2.8,
) -> torch.Tensor | None:
    weights: List[float] = []

    for sample in samples:
        labels = [label for label in sample.get("entity_labels", []) if label != IGNORE_INDEX]
        if not labels:
            density = 0.0
        else:
            non_o = sum(1 for label in labels if label != o_label_id)
            density = non_o / max(1, len(labels))

        weight = min_weight + density * (max_weight - min_weight)
        weights.append(weight)

    if not weights:
        return None

    weights_arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_arr).all():
        return None

    weights_arr = np.clip(weights_arr, min_weight, max_weight)

    if weights_arr.sum() == 0:
        return None

    return torch.tensor(weights_arr, dtype=torch.double)


def build_model(
    config: ModelConfig,
    intent_cfg: IntentConfig,
    entity_cfg: EntityConfig,
    command_cfg: CommandConfig,
) -> MultiTaskModel:
    return MultiTaskModel(
        model_name=config.model_name,
        num_intents=len(intent_cfg.intent_labels),
        num_entity_labels=len(entity_cfg.entity_labels),
        num_commands=len(command_cfg.command_labels),
        dropout=config.dropout,
        use_safetensors=getattr(config, "use_safetensors", True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT đa tác vụ trên GPU.")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch (override config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (override config).")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên.")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục lưu checkpoint.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker DataLoader.")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử 1 batch để kiểm tra nhanh.")
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Tỷ lệ mẫu train sử dụng (0<ratio<=1).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Giới hạn số mẫu train (ưu tiên hơn subset-ratio).",
    )
    parser.add_argument("--subset-seed", type=int, default=42, help="Seed chọn subset.")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="Số epoch đầu chỉ train head (đóng băng encoder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig()
    intent_cfg = IntentConfig()
    entity_cfg = EntityConfig()
    command_cfg = CommandConfig()

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.freeze_epochs is not None:
        config.freeze_encoder_epochs = max(0, args.freeze_epochs)

    set_seed(getattr(config, "seed", 42))
    output_dir = Path(config.output_dir)
    log_dir = Path(getattr(config, "log_dir", "logs/vitext_training"))
    logger = configure_logger(log_dir)

    logger.info("Bắt đầu huấn luyện đa tác vụ PhoBERT.")
    processor = DataProcessor()
    train_raw, val_raw, test_raw = load_datasets(processor, config)
    logger.info("Loaded datasets - train: %d, val: %d, test: %d", len(train_raw), len(val_raw), len(test_raw))

    train_raw = sample_subset(
        train_raw,
        ratio=args.subset_ratio,
        max_samples=args.subset_size,
        seed=args.subset_seed,
    )
    if not train_raw:
        logger.error("Subset train rỗng - kiểm tra subset-ratio/subset-size.")
        return

    summarize_commands("Train subset (raw)", train_raw)
    summarize_commands("Validation (raw)", val_raw)
    summarize_commands("Test (raw)", test_raw)

    train_samples = prepare_multitask_samples(processor, train_raw)
    val_samples = prepare_multitask_samples(processor, val_raw)
    test_samples = prepare_multitask_samples(processor, test_raw)

    log_dataset_statistics("Train subset (processed)", train_samples, processor)
    log_dataset_statistics("Validation (processed)", val_samples, processor)
    log_dataset_statistics("Test (processed)", test_samples, processor)

    o_label_id = processor.entity_label2id.get("O", 0)
    entity_class_weights = compute_entity_class_weights(
        train_samples,
        len(entity_cfg.entity_labels),
        o_label_id=o_label_id,
    )
    if entity_class_weights is not None:
        o_index = processor.entity_label2id.get("O", 0)
        logger.info(
            "Entity class weights: O=%.3f | min=%.3f | max=%.3f",
            float(entity_class_weights[o_index].item()),
            float(entity_class_weights.min().item()),
            float(entity_class_weights.max().item()),
        )
    else:
        logger.warning("Không tính được entity class weights - giữ nguyên CrossEntropyLoss mặc định.")

    sampler_weights = compute_entity_density_sampler_weights(train_samples, o_label_id)
    train_sampler = None
    if sampler_weights is not None:
        sampler_weight_list = sampler_weights.tolist()
        train_sampler = WeightedRandomSampler(
            weights=sampler_weight_list,
            num_samples=len(train_samples),
            replacement=False,
        )
        logger.info(
            "Áp dụng WeightedRandomSampler theo mật độ entity (min=%.3f | max=%.3f).",
            float(sampler_weights.min().item()),
            float(sampler_weights.max().item()),
        )
    else:
        logger.warning("Không thể tạo sampler dựa trên mật độ entity - fallback shuffle ngẫu nhiên.")

    train_dataset = MultiTaskDataset(train_samples)
    val_dataset = MultiTaskDataset(val_samples)
    test_dataset = MultiTaskDataset(test_samples)

    batch_size = getattr(config, "batch_size", 16)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(config, intent_cfg, entity_cfg, command_cfg)
    trainer = MultitaskTrainer(
        model=model,
        config=config,
        processor=processor,
        output_dir=output_dir,
        gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
        freeze_encoder_epochs=getattr(config, "freeze_encoder_epochs", 0),
        entity_class_weights=entity_class_weights,
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        loss_dict = trainer._compute_loss(batch, training=True, return_outputs=False)
        logger.info("Dry-run loss (1 batch): %.4f", float(loss_dict["total_loss"]))
        return

    outcome = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=getattr(config, "num_epochs", 4),
        evaluation_loader=test_loader,
    )

    logger.info("Huấn luyện xong. Best epoch = %d, best metric = %.4f", outcome.best_epoch, outcome.best_val_metric)
    if outcome.best_checkpoint_path:
        logger.info("Best checkpoint: %s", outcome.best_checkpoint_path)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(outcome.history, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu training history tại %s", history_path)


if __name__ == "__main__":
    main()

