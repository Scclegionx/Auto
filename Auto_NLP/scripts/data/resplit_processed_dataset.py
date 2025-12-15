#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resplit processed dataset (train/val/test) theo phân bố cân bằng intent.

Giữ nguyên cấu trúc mẫu đã được chuẩn hóa: input, command, entities, spans, bio_labels...
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "src" / "data" / "processed"


def load_split(filename: str) -> List[Dict]:
    path = PROCESSED_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_split(samples: List[Dict], filename: str) -> None:
    path = PROCESSED_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def summarize_split(name: str, samples: List[Dict]) -> None:
    counter = Counter(sample.get("command", "unknown") for sample in samples)
    total = len(samples)
    summary = ", ".join(f"{cmd}: {count}" for cmd, count in sorted(counter.items()))
    print(f"{name}: {total} samples")
    print(f"  {summary}")


def resplit_dataset(train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_samples = load_split("train.json")
    val_samples = load_split("val.json")
    test_samples = load_split("test.json")

    all_samples = train_samples + val_samples + test_samples
    if not all_samples:
        raise RuntimeError("Dataset trống - không thể resplit.")

    commands = [sample.get("command", "unknown") for sample in all_samples]

    train_samples, temp_samples = train_test_split(
        all_samples,
        test_size=1 - train_ratio,
        stratify=commands,
        random_state=seed,
    )

    temp_commands = [sample.get("command", "unknown") for sample in temp_samples]

    val_size = val_ratio / (1 - train_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=1 - val_size,
        stratify=temp_commands,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


def main() -> None:
    print(f"Resplitting dataset trong {PROCESSED_DIR} ...")
    train_samples, val_samples, test_samples = resplit_dataset(train_ratio=0.8, val_ratio=0.1, seed=42)

    save_split(train_samples, "train.json")
    save_split(val_samples, "val.json")
    save_split(test_samples, "test.json")

    summarize_split("Train", train_samples)
    summarize_split("Val", val_samples)
    summarize_split("Test", test_samples)


if __name__ == "__main__":
    main()

