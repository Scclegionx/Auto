#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sửa lệch nhãn BIO sau khi chuẩn hóa text:
- Đồng bộ entities/spans dựa trên substring của input.
- Regenerate BIO labels bằng PhoBERT tokenizer (DataProcessor.align_labels).

Ví dụ:
    python scripts/data/repair_iob2_labels.py \
        --paths src/data/processed/train.json src/data/processed/val.json src/data/processed/test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.entity_schema import canonicalize_entity_dict  # noqa: E402
from data.processed.data_processor import DataProcessor  # noqa: E402


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(path: Path, data: List[Dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_entities(sample: Dict[str, object]) -> List[Dict[str, object]]:
    """Đồng bộ text/start/end của entity với input hiện tại."""
    text = sample.get("input", "")
    normalized: List[Dict[str, object]] = []
    for entity in sample.get("entities", []):
        if not isinstance(entity, dict):
            continue
        canonical = canonicalize_entity_dict(entity)
        start = canonical.get("start")
        end = canonical.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            canonical["text"] = text[start:end]
        normalized.append(canonical)
    return normalized


def repair_sample(processor: DataProcessor, sample: Dict[str, object]) -> Dict[str, object]:
    sample = dict(sample)
    entities = rebuild_entities(sample)
    sample["entities"] = entities
    sample["spans"] = entities
    sample["bio_labels"] = processor.align_labels(sample.get("input", ""), entities)
    return sample


def process_file(path: Path, processor: DataProcessor) -> None:
    data = load_dataset(path)
    repaired = [repair_sample(processor, sample) for sample in data]
    save_dataset(path, repaired)
    print(f"Repaired {len(repaired)} samples in {path}")


def main():
    parser = argparse.ArgumentParser(description="Repair BIO labels after text normalization.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Danh sách file JSON cần sửa (ví dụ train/val/test).",
    )
    args = parser.parse_args()

    processor = DataProcessor()

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: File không tồn tại: {path}")
            continue
        process_file(path, processor)


if __name__ == "__main__":
    main()


