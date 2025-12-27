#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebalance dataset theo target cố định mỗi intent.

Logic:
  - Nếu số mẫu > target: downsample ngẫu nhiên.
  - Nếu số mẫu < target: sinh thêm bằng augment_vi_asr (giữ entity đúng).
  - Gộp lại và shuffle.

Ví dụ:
    python scripts/data/rebalance_dataset.py \
        --input artifacts/cleaned/train_clean.json \
        --output artifacts/balanced/train_balanced.json \
        --target 3300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.processed.data_processor import DataProcessor  # noqa: E402
from scripts.augment import augment_vi_asr  # noqa: E402

AUGMENT_OPERATIONS = ["filler", "number_swap", "synonym", "noise"]


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def rebalance(samples: List[Dict], target: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    processor = DataProcessor()
    by_command: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_command[sample.get("command", sample.get("intent", "unknown"))].append(sample)

    balanced: List[Dict] = []

    for command, command_samples in by_command.items():
        current = len(command_samples)
        if current >= target:
            selected = rng.sample(command_samples, target)
            balanced.extend(selected)
            continue

        deficit = target - current
        balanced.extend(command_samples)

        pool = command_samples.copy()
        idx = 0
        generated = 0
        while generated < deficit and pool:
            sample = pool[idx % len(pool)]
            augmented = augment_vi_asr.augment_sample(sample, processor, AUGMENT_OPERATIONS)
            for aug in augmented:
                aug["split"] = sample.get("split", "train")
                balanced.append(aug)
                generated += 1
                if generated >= deficit:
                    break
            idx += 1

    rng.shuffle(balanced)

    # Clip to exact target per command
    final_by_command: Dict[str, List[Dict]] = defaultdict(list)
    for sample in balanced:
        command = sample.get("command", sample.get("intent", "unknown"))
        final_by_command[command].append(sample)

    final_dataset: List[Dict] = []
    for command, items in final_by_command.items():
        if len(items) > target:
            final_dataset.extend(items[:target])
        else:
            final_dataset.extend(items)

    rng.shuffle(final_dataset)
    return final_dataset


def main():
    parser = argparse.ArgumentParser(description="Rebalance dataset bằng augment/downsample.")
    parser.add_argument("--input", type=Path, required=True, help="Đường dẫn dataset (đã clean).")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn xuất dataset đã rebalance.")
    parser.add_argument("--target", type=int, default=3300, help="Số mẫu mục tiêu mỗi intent.")
    args = parser.parse_args()

    samples = load_dataset(args.input)
    balanced = rebalance(samples, args.target)
    save_dataset(args.output, balanced)
    counts = defaultdict(int)
    for sample in balanced:
        counts[sample.get("command", sample.get("intent", "unknown"))] += 1
    print("Intent counts sau rebalance:")
    for command, count in sorted(counts.items()):
        print(f"- {command:18s}: {count}")
    print(f"Tổng mẫu: {len(balanced)}")


if __name__ == "__main__":
    main()

