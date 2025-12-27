#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiểm tra tính hợp lệ của nhãn NER IOB2 trong bộ dữ liệu processed.

Các kiểm tra:
- Tất cả nhãn phải thuộc tập {O, B-*, I-*} theo schema hiện hành.
- Nhãn I-* không được xuất hiện nếu trước đó không có B-* hoặc I-* cùng loại.
- Số lượng entity theo danh sách spans phải trùng với số lượng nhãn B-* tương ứng.
- So sánh entities và spans (label/text/start/end) để phát hiện lệch.

Sử dụng:
    python scripts/data/validate_iob2_labels.py \
        --paths src/data/processed/train.json src/data/processed/val.json src/data/processed/test.json \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.entity_schema import (  # noqa: E402
    canonicalize_entity_dict,
    canonicalize_entity_label,
    generate_entity_labels,
)

VALID_LABELS = set(generate_entity_labels())
VALID_LABELS.add("O")


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_sequence(labels: List[str]) -> List[str]:
    issues: List[str] = []
    prev_tag = "O"
    prev_type: str | None = None

    for idx, label in enumerate(labels):
        if label not in VALID_LABELS:
            issues.append(f"Tag '{label}' ngoài schema tại vị trí {idx}")
            prev_tag = "O"
            prev_type = None
            continue

        if label == "O":
            prev_tag = "O"
            prev_type = None
            continue

        tag, _, ent_type = label.partition("-")

        if tag == "B":
            prev_tag = "B"
            prev_type = ent_type
            continue

        if tag == "I":
            if prev_type != ent_type or prev_tag == "O":
                issues.append(
                    f"I-{ent_type} tại vị trí {idx} không nối tiếp B-{ent_type}"
                )
            prev_tag = "I"
            prev_type = ent_type

    return issues


def count_entities(items: List[Dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for entity in items:
        canonical = canonicalize_entity_dict(entity)
        label = canonical.get("label")
        if label:
            counter[canonicalize_entity_label(label)] += 1
    return counter


def compare_counts(bio_labels: List[str], entities: List[Dict]) -> List[str]:
    issues: List[str] = []
    b_counter: Counter[str] = Counter()
    for label in bio_labels:
        if label.startswith("B-"):
            b_counter[label[2:]] += 1

    entity_counter = count_entities(entities)

    all_labels = set(b_counter.keys()) | set(entity_counter.keys())
    for label in sorted(all_labels):
        bio_count = b_counter.get(label, 0)
        ent_count = entity_counter.get(label, 0)
        if bio_count != ent_count:
            issues.append(
                f"Lệch số lượng entity '{label}': BIO={bio_count}, spans={ent_count}"
            )
    return issues


def compare_entities_spans(entities: List[Dict], spans: List[Dict]) -> List[str]:
    issues: List[str] = []
    if len(entities) != len(spans):
        issues.append(f"Số entity ({len(entities)}) khác số spans ({len(spans)})")
        return issues

    for idx, (entity, span) in enumerate(zip(entities, spans)):
        ent_can = canonicalize_entity_dict(entity)
        span_can = canonicalize_entity_dict(span)

        for key in ("label", "text", "start", "end"):
            if ent_can.get(key) != span_can.get(key):
                issues.append(
                    f"Entity #{idx} khác {key}: entity='{ent_can.get(key)}', span='{span_can.get(key)}'"
                )
    return issues


def analyze_file(path: Path, limit: int) -> Tuple[int, Dict[str, List[str]]]:
    data = load_dataset(path)
    problems: Dict[str, List[str]] = defaultdict(list)

    for idx, sample in enumerate(data):
        bio_labels: List[str] = sample.get("bio_labels", [])
        entities: List[Dict] = sample.get("entities", [])
        spans: List[Dict] = sample.get("spans", [])

        seq_issues = check_sequence(bio_labels)
        if seq_issues:
            problems["sequence"].append(
                f"#{idx} {sample.get('command')}: {seq_issues}"
            )

        count_issues = compare_counts(bio_labels, spans or entities)
        if count_issues:
            problems["counts"].append(
                f"#{idx} {sample.get('command')}: {count_issues}"
            )

        entity_issues = compare_entities_spans(entities, spans)
        if entity_issues:
            problems["entities_vs_spans"].append(
                f"#{idx} {sample.get('command')}: {entity_issues}"
            )

        if limit > 0 and all(len(v) >= limit for v in problems.values()):
            break

    return len(data), problems


def main():
    parser = argparse.ArgumentParser(description="Validate IOB2 labels in dataset.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Danh sách file JSON (train/val/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Số lượng vấn đề tối đa in ra cho mỗi loại.",
    )
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: File không tồn tại: {path}")
            continue

        total, problems = analyze_file(path, args.limit)
        print(f"\n=== {path} ({total} samples) ===")
        if not problems:
            print("Không phát hiện vấn đề.")
            continue

        for category, samples in problems.items():
            print(f"- {category}: {len(samples)} mẫu")
            for sample in samples[: args.limit]:
                print(f"    * {sample}")


if __name__ == "__main__":
    main()


