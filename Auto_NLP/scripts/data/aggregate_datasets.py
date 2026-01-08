#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tổng hợp thông tin từ master dataset và train/val/test splits.
Tạo báo cáo chi tiết về toàn bộ dataset.

Usage:
    python scripts/data/aggregate_datasets.py
    python scripts/data/aggregate_datasets.py --output reports/dataset_aggregate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent.parent))

def load_json(path: Path) -> List[Dict]:
    """Load JSON file."""
    if not path.exists():
        print(f"[WARN] File khong ton tai: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def analyze_samples(samples: List[Dict], split_name: str = "all") -> Dict[str, Any]:
    """Phân tích một tập samples."""
    if not samples:
        return {}
    
    # Basic stats
    total_samples = len(samples)
    
    # Intent distribution
    intent_counter = Counter()
    command_counter = Counter()
    entity_counter = Counter()
    entity_type_counter = Counter()
    
    # Entity stats
    entity_counts_per_sample = []
    sentence_lengths = []
    entities_by_intent = defaultdict(Counter)
    
    for sample in samples:
        # Intent/Command
        intent = sample.get("intent", sample.get("command", "unknown"))
        command = sample.get("command", "unknown")
        intent_counter[intent] += 1
        command_counter[command] += 1
        
        # Sentence length
        text = sample.get("input", "")
        words = text.split()
        sentence_lengths.append(len(words))
        
        # Entities
        entities = sample.get("entities", [])
        entity_counts_per_sample.append(len(entities))
        
        # Entity labels
        for ent in entities:
            if isinstance(ent, dict):
                label = ent.get("label", ent.get("type", "UNKNOWN"))
                entity_counter[label] += 1
                entity_type_counter[label] += 1
                entities_by_intent[intent][label] += 1
    
    # Calculate statistics
    avg_entities = sum(entity_counts_per_sample) / len(entity_counts_per_sample) if entity_counts_per_sample else 0
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    min_sentence_length = min(sentence_lengths) if sentence_lengths else 0
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
    
    return {
        "split": split_name,
        "total_samples": total_samples,
        "intent_distribution": dict(intent_counter),
        "command_distribution": dict(command_counter),
        "entity_distribution": dict(entity_counter.most_common(20)),
        "entity_type_distribution": dict(entity_type_counter.most_common(20)),
        "statistics": {
            "avg_entities_per_sample": round(avg_entities, 2),
            "max_entities_per_sample": max(entity_counts_per_sample) if entity_counts_per_sample else 0,
            "min_entities_per_sample": min(entity_counts_per_sample) if entity_counts_per_sample else 0,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "min_sentence_length": min_sentence_length,
            "max_sentence_length": max_sentence_length,
        },
        "entities_by_intent": {intent: dict(counter) for intent, counter in entities_by_intent.items()},
    }

def aggregate_datasets(
    master_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> Dict[str, Any]:
    """Tổng hợp thông tin từ tất cả datasets."""
    print("[INFO] Dang tong hop dataset...")
    print(f"  - Master: {master_path}")
    print(f"  - Train: {train_path}")
    print(f"  - Val: {val_path}")
    print(f"  - Test: {test_path}")
    print()
    
    # Load datasets
    master_data = load_json(master_path)
    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)
    
    # Analyze each split
    master_analysis = analyze_samples(master_data, "master")
    train_analysis = analyze_samples(train_data, "train")
    val_analysis = analyze_samples(val_data, "val")
    test_analysis = analyze_samples(test_data, "test")
    
    # Combined analysis
    all_data = train_data + val_data + test_data
    combined_analysis = analyze_samples(all_data, "combined")
    
    # Verify consistency
    total_split = len(train_data) + len(val_data) + len(test_data)
    consistency_check = {
        "master_count": len(master_data),
        "split_total": total_split,
        "matches": len(master_data) == total_split,
        "train_ratio": round(len(train_data) / total_split * 100, 2) if total_split > 0 else 0,
        "val_ratio": round(len(val_data) / total_split * 100, 2) if total_split > 0 else 0,
        "test_ratio": round(len(test_data) / total_split * 100, 2) if total_split > 0 else 0,
    }
    
    return {
        "metadata": {
            "master_file": str(master_path),
            "train_file": str(train_path),
            "val_file": str(val_path),
            "test_file": str(test_path),
        },
        "consistency_check": consistency_check,
        "master": master_analysis,
        "train": train_analysis,
        "val": val_analysis,
        "test": test_analysis,
        "combined": combined_analysis,
        "summary": {
            "total_samples": {
                "master": len(master_data),
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "total_splits": total_split,
            },
            "unique_intents": len(set(master_analysis.get("intent_distribution", {}).keys())),
            "unique_commands": len(set(master_analysis.get("command_distribution", {}).keys())),
            "unique_entity_types": len(master_analysis.get("entity_type_distribution", {})),
        }
    }

def save_report(data: Dict[str, Any], output_path: Path) -> None:
    """Lưu báo cáo JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu báo cáo: {output_path}")

def print_summary(data: Dict[str, Any]) -> None:
    """In tóm tắt ra console."""
    print("\n" + "="*70)
    print("TONG HOP DATASET")
    print("="*70)
    
    summary = data["summary"]
    consistency = data["consistency_check"]
    
    print(f"\n[TONG QUAN]")
    print(f"  - Master samples: {summary['total_samples']['master']:,}")
    print(f"  - Train samples: {summary['total_samples']['train']:,} ({consistency['train_ratio']}%)")
    print(f"  - Val samples: {summary['total_samples']['val']:,} ({consistency['val_ratio']}%)")
    print(f"  - Test samples: {summary['total_samples']['test']:,} ({consistency['test_ratio']}%)")
    print(f"  - Unique intents: {summary['unique_intents']}")
    print(f"  - Unique commands: {summary['unique_commands']}")
    print(f"  - Unique entity types: {summary['unique_entity_types']}")
    
    print(f"\n[CONSISTENCY CHECK]")
    print(f"  - Master = Train+Val+Test: {consistency['matches']}")
    if not consistency['matches']:
        print(f"    [WARN] Chenh lech: {abs(consistency['master_count'] - consistency['split_total'])} samples")
    
    # Intent distribution
    master = data["master"]
    if "intent_distribution" in master:
        print(f"\n[INTENT DISTRIBUTION] (Master):")
        for intent, count in sorted(master["intent_distribution"].items(), key=lambda x: -x[1])[:10]:
            pct = count / master["total_samples"] * 100
            print(f"  - {intent:20s}: {count:5,} ({pct:5.2f}%)")
    
    # Entity distribution
    if "entity_type_distribution" in master:
        print(f"\n[TOP ENTITY TYPES] (Master):")
        for entity, count in list(master["entity_type_distribution"].items())[:10]:
            print(f"  - {entity:20s}: {count:6,}")
    
    # Statistics
    if "statistics" in master:
        stats = master["statistics"]
        print(f"\n[STATISTICS] (Master):")
        print(f"  - Avg entities/sample: {stats.get('avg_entities_per_sample', 0):.2f}")
        print(f"  - Avg sentence length: {stats.get('avg_sentence_length', 0):.2f} words")
        print(f"  - Sentence length range: {stats.get('min_sentence_length', 0)} - {stats.get('max_sentence_length', 0)} words")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description="Tổng hợp thông tin từ master và train/val/test datasets.")
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("src/data/raw/elderly_commands_master.json"),
        help="Đường dẫn master dataset.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("src/data/processed/train.json"),
        help="Đường dẫn train dataset.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("src/data/processed/val.json"),
        help="Đường dẫn val dataset.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("src/data/processed/test.json"),
        help="Đường dẫn test dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_aggregate_report.json"),
        help="Đường dẫn file output JSON.",
    )
    args = parser.parse_args()
    
    # Aggregate
    report = aggregate_datasets(args.master, args.train, args.val, args.test)
    
    # Save
    save_report(report, args.output)
    
    # Print summary
    print_summary(report)
    
    print(f"\n[OK] Hoan tat! Bao cao da duoc luu tai: {args.output}")

if __name__ == "__main__":
    main()

