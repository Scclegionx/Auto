#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thống kê đầy đủ master dataset.
Usage:
    python scripts/data/analyze_master_dataset.py --input src/data/raw/master_dataset_35609.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset từ file JSON."""
    print(f"Đang đọc: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset phải là list, nhận được: {type(data)}")
    return data


def analyze_dataset(samples: List[Dict]) -> Dict:
    """Phân tích toàn diện dataset."""
    print("\n" + "=" * 60)
    print("THỐNG KÊ DATASET")
    print("=" * 60)
    
    # 1. Tổng số mẫu
    total = len(samples)
    print(f"\n📊 Tổng số mẫu: {total:,}")
    
    # 2. Phân bố Intent
    intent_counter = Counter()
    command_counter = Counter()
    
    for sample in samples:
        intent = sample.get("intent", sample.get("command", "unknown"))
        command = sample.get("command", sample.get("intent", "unknown"))
        intent_counter[intent] += 1
        command_counter[command] += 1
    
    print(f"\n📌 Phân bố Intent ({len(intent_counter)} loại):")
    for intent, count in intent_counter.most_common():
        pct = count / total * 100
        print(f"  {intent:20s}: {count:6,} mẫu ({pct:5.2f}%)")
    
    print(f"\n📌 Phân bố Command ({len(command_counter)} loại):")
    for command, count in command_counter.most_common():
        pct = count / total * 100
        print(f"  {command:20s}: {count:6,} mẫu ({pct:5.2f}%)")
    
    # 3. Thống kê Entity
    entity_label_counter = Counter()
    total_entities = 0
    samples_with_entities = 0
    
    for sample in samples:
        entities = sample.get("entities", [])
        if entities:
            samples_with_entities += 1
            for ent in entities:
                if isinstance(ent, dict):
                    label = ent.get("label", "UNKNOWN")
                    entity_label_counter[label] += 1
                    total_entities += 1
    
    print(f"\n📌 Thống kê Entity:")
    print(f"  Tổng số entity: {total_entities:,}")
    print(f"  Mẫu có entity: {samples_with_entities:,} / {total:,} ({samples_with_entities/total*100:.2f}%)")
    print(f"  Trung bình entity/mẫu: {total_entities/total:.2f}")
    
    print(f"\n  Top 15 entity labels:")
    for label, count in entity_label_counter.most_common(15):
        pct = count / total_entities * 100
        print(f"    {label:20s}: {count:6,} ({pct:5.2f}%)")
    
    # 4. Độ dài câu (input)
    lengths = []
    for sample in samples:
        text = sample.get("input", "")
        lengths.append(len(text.split()))
    
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        
        print(f"\n📌 Độ dài câu (số từ):")
        print(f"  Trung bình: {avg_len:.2f}")
        print(f"  Min: {min_len}")
        print(f"  Max: {max_len}")
    
    # 5. Thống kê BIO labels (nếu có)
    bio_counter = Counter()
    samples_with_bio = 0
    
    for sample in samples:
        bio_labels = sample.get("bio_labels", [])
        if bio_labels:
            samples_with_bio += 1
            for label in bio_labels:
                bio_counter[label] += 1
    
    print(f"\n📌 Thống kê BIO labels:")
    print(f"  Mẫu có bio_labels: {samples_with_bio:,} / {total:,} ({samples_with_bio/total*100:.2f}%)")
    if bio_counter:
        print(f"  Top 10 BIO tags:")
        for label, count in bio_counter.most_common(10):
            print(f"    {label:20s}: {count:6,}")
    
    # 6. Thống kê split (nếu có)
    split_counter = Counter()
    for sample in samples:
        split = sample.get("split", "unknown")
        split_counter[split] += 1
    
    if len(split_counter) > 1:
        print(f"\n📌 Phân bố Split:")
        for split, count in split_counter.most_common():
            pct = count / total * 100
            print(f"  {split:15s}: {count:6,} mẫu ({pct:5.2f}%)")
    
    print(f"\n{'='*60}")
    print("✅ Phân tích hoàn tất!")
    print("=" * 60)
    
    return {
        "total_samples": total,
        "intent_distribution": dict(intent_counter),
        "command_distribution": dict(command_counter),
        "entity_stats": {
            "total_entities": total_entities,
            "samples_with_entities": samples_with_entities,
            "label_distribution": dict(entity_label_counter.most_common(20)),
        },
        "text_length": {
            "avg": avg_len if lengths else 0,
            "min": min_len if lengths else 0,
            "max": max_len if lengths else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phân tích master dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/data/raw/master_dataset_35609.json"),
        help="Đường dẫn tới master dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Đường dẫn lưu kết quả phân tích (JSON), nếu muốn.",
    )
    args = parser.parse_args()
    
    samples = load_dataset(args.input)
    stats = analyze_dataset(samples)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu kết quả phân tích tại: {args.output}")


if __name__ == "__main__":
    main()





