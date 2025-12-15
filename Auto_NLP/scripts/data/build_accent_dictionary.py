#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh bảng chuyển đổi không dấu -> có dấu dựa trên corpus hiện tại.

Logic:
- Duyệt qua các file JSON (train/val/test).
- Với mỗi token chứa dấu tiếng Việt, loại dấu để lấy "base token".
- Đếm tần suất mapping base -> token có dấu.
- Chọn biến thể xuất hiện nhiều nhất làm ánh xạ mặc định.
- Gộp với bảng đã có (nếu cung cấp) để giữ các chỉnh sửa thủ công.

Ví dụ:
    python scripts/data/build_accent_dictionary.py \
        --paths src/data/processed/train.json src/data/processed/val.json src/data/processed/test.json \
        --output resources/generated_accent_map.json \
        --existing resources/vietnamese_accent_map.json
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def strip_accents(token: str) -> str:
    normalized = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def has_accent(token: str) -> bool:
    return any(
        unicodedata.category(ch) == "Mn" or ord(ch) > 127
        for ch in unicodedata.normalize("NFD", token)
    )


def tokenize(text: str) -> Iterable[str]:
    return text.split()


def build_mapping_from_file(path: Path, fields: List[str]) -> Dict[str, Counter]:
    mapping: Dict[str, Counter] = defaultdict(Counter)
    data = json.loads(path.read_text(encoding="utf-8"))
    for sample in data:
        for field in fields:
            value = sample.get(field)
            if not isinstance(value, str):
                continue
            for token in tokenize(value):
                clean = token.strip(".,!?;:\"'()[]{}“”‘’…")
                if not clean:
                    continue
                if has_accent(clean):
                    base = strip_accents(clean).lower()
                    mapping[base][clean] += 1
    return mapping


def merge_mappings(sources: List[Dict[str, Counter]]) -> Dict[str, Counter]:
    merged: Dict[str, Counter] = defaultdict(Counter)
    for source in sources:
        for base, counter in source.items():
            merged[base].update(counter)
    return merged


def select_best_mapping(mapping: Dict[str, Counter]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for base, counter in mapping.items():
        if not counter:
            continue
        best = counter.most_common(1)[0][0]
        selected[base] = best
    return selected


def load_existing(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser(description="Sinh bảng chuyển đổi dấu từ dataset.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Danh sách file JSON (train/val/test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="File JSON lưu mapping kết quả.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["input"],
        help="Các trường văn bản cần phân tích (mặc định: input).",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        help="Bảng mapping đã có để merge (ưu tiên existing).",
    )
    args = parser.parse_args()

    all_mappings: List[Dict[str, Counter]] = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: File không tồn tại: {path}")
            continue
        mapping = build_mapping_from_file(path, args.fields)
        all_mappings.append(mapping)
        print(f"Processed {path}: {len(mapping)} base tokens collected.")

    merged = merge_mappings(all_mappings)
    selected = select_best_mapping(merged)

    existing = load_existing(args.existing)
    if existing:
        selected.update(existing)
        print(f"Merged with existing dictionary ({len(existing)} entries).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} mappings to {args.output}")


if __name__ == "__main__":
    main()


