#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validator cho dataset Auto_NLP theo schema entity cuối.

Kiểm tra:
  - Quy mô & phân bố intent/command.
  - Độ dài câu (4–20 token, tối thiểu 10% > 25 token).
  - Số lượt gán entity chủ lực (>= 1500).
  - Ràng buộc entity theo intent/command.

Sử dụng:
    python scripts/dataset/validate_final_dataset.py \
        --paths src/data/processed/train.json src/data/processed/val.json src/data/processed/test.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.entity_schema import canonicalize_entity_dict, canonicalize_entity_label  # noqa: E402

# --------------------------- cấu hình yêu cầu ---------------------------

COMMAND_CONSTRAINTS = {
    "add-contacts": {
        "all": {"CONTACT_NAME", "PHONE"},
    },
    "call": {
        "any": [{"CONTACT_NAME", "RECEIVER", "PHONE"}],
    },
    "make-video-call": {
        "any": [{"CONTACT_NAME", "RECEIVER", "PHONE"}],
    },
    "send-mess": {
        "all": {"MESSAGE"},
        "any": [{"CONTACT_NAME", "RECEIVER", "PHONE"}],
    },
    "set-alarm": {
        "any": [{"TIME", "DATE"}],
        "optional": {"REMINDER_CONTENT", "FREQUENCY"},
    },
    "search-internet": {
        "all": {"QUERY"},
    },
    "search-youtube": {
        "all": {"QUERY"},
        "enforce_values": {"PLATFORM": {"youtube", "yt", "you tube"}},
    },
    "get-info": {
        "all": {"QUERY"},
    },
    "control-device": {
        "all": {"DEVICE", "ACTION"},
    },
    "open-cam": {
        "any": [{"ACTION", "CAMERA_TYPE"}],
    },
}

ENTITY_PRIMARY = {"RECEIVER", "CONTACT_NAME", "PHONE", "TIME", "DATE", "QUERY", "MESSAGE", "DEVICE"}


# --------------------------- hàm tiện ích ---------------------------

def load_dataset(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_entities(entities: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    normalized = defaultdict(list)
    for entity in entities:
        canonical = canonicalize_entity_dict(entity)
        label = canonical.get("label")
        text = canonical.get("text", "")
        if label:
            normalized[label].append(text)
    return normalized


def tokens_count(text: str) -> int:
    return len(text.strip().split())


def check_constraint(command: str, entities: Dict[str, List[str]]) -> List[str]:
    issues: List[str] = []
    spec = COMMAND_CONSTRAINTS.get(command)
    if not spec:
        return issues

    if "all" in spec:
        for required in spec["all"]:
            if not entities.get(required):
                issues.append(f"Thiếu entity bắt buộc '{required}'")

    if "any" in spec:
        for group in spec["any"]:
            if not any(entities.get(name) for name in group):
                issues.append(f"Thiếu ít nhất 1 entity trong nhóm {group}")

    if "enforce_values" in spec:
        for name, valid_values in spec["enforce_values"].items():
            values = entities.get(name, [])
            if values and all(v.lower() not in valid_values for v in values):
                issues.append(f"Entity '{name}' không nằm trong tập {valid_values}")

    return issues


def aggregate_stats(samples: Iterable[Dict[str, object]]) -> Dict[str, object]:
    intent_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    token_lengths: List[int] = []
    violations: Dict[str, List[str]] = defaultdict(list)

    for sample in samples:
        command = sample.get("command") or sample.get("intent")
        text = sample.get("input", "")
        entities_raw = sample.get("entities", [])
        entities = canonicalize_entities(entities_raw)

        intent_counter[command] += 1
        token_lengths.append(tokens_count(text))
        for label, values in entities.items():
            entity_counter[label] += len(values)

        issues = check_constraint(command, entities)
        if issues:
            violations[command].append(
                f"{sample.get('input')[:80]}... -> {issues}"
            )

    return {
        "intent_distribution": intent_counter,
        "entity_distribution": entity_counter,
        "token_lengths": token_lengths,
        "violations": violations,
    }


def print_summary(stats: Dict[str, object], total_samples: int) -> None:
    intents: Counter[str] = stats["intent_distribution"]
    entities: Counter[str] = stats["entity_distribution"]
    lengths: List[int] = stats["token_lengths"]
    violations: Dict[str, List[str]] = stats["violations"]

    print("=== Intent distribution ===")
    for intent, count in intents.most_common():
        print(f"- {intent:18s}: {count}")

    min_count = min(intents.values()) if intents else 0
    max_count = max(intents.values()) if intents else 0
    if max_count > 2 * min_count:
        print(f"[WARN] Lệch class: min={min_count}, max={max_count}")

    print("\n=== Entity occurrences ===")
    for entity, count in entities.most_common():
        print(f"- {entity:20s}: {count}")

    print("\n=== Token length stats ===")
    if lengths:
        over_25 = sum(1 for l in lengths if l > 25)
        print(f"Mean length: {sum(lengths)/len(lengths):.2f}")
        print(f"Min length : {min(lengths)}")
        print(f"Max length : {max(lengths)}")
        print(f">25 tokens : {over_25} samples ({over_25/len(lengths)*100:.2f}%)")

    print("\n=== Constraint violations ===")
    if not violations:
        print("Không phát hiện vi phạm.")
    else:
        for intent, samples in violations.items():
            print(f"- {intent}: {len(samples)} lỗi")
            for sample in samples[:5]:
                print(f"    * {sample}")

    missing_primary = [entity for entity in ENTITY_PRIMARY if entities.get(entity, 0) < 1500]
    if missing_primary:
        print("\n[WARN] Các entity chủ lực chưa đạt 1.5k lượt gán:", ", ".join(missing_primary))

    if total_samples < 10000:
        print(f"[WARN] Tổng mẫu hiện tại {total_samples}, mục tiêu 10k–12k.")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset theo schema final.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Danh sách đường dẫn JSON (có thể là train/val/test).",
    )
    args = parser.parse_args()

    all_samples: List[Dict[str, object]] = []
    for path in args.paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        samples = load_dataset(file_path)
        print(f"Loaded {len(samples)} samples từ {file_path}")
        all_samples.extend(samples)

    print(f"\nTổng số mẫu: {len(all_samples)}")
    stats = aggregate_stats(all_samples)
    print_summary(stats, len(all_samples))


if __name__ == "__main__":
    main()

