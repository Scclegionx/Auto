#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_master_entities.py
-------------------------

Dùng sau khi anh CHỈNH SỬA THỦ CÔNG trường "input" trong
`src/data/raw/elderly_commands_master.json`.

Mục tiêu:
- Đồng bộ lại `text`, `start`, `end` trong `entities`/`spans` với câu mới.
- Chuẩn hoá nhãn entity (canonicalize).
- Sinh lại trường `bio_labels` bằng DataProcessor.align_labels.

Giả định:
- Danh sách entity (label + text) vẫn đúng về mặt ngữ nghĩa, anh chỉ thay đổi
  câu input (thêm bớt từ, sửa chính tả, ...).

Usage (an toàn, ghi ra file mới):

    python scripts/data/repair_master_entities.py \
        --input src/data/raw/elderly_commands_master.json \
        --output artifacts/elderly_commands_master.repaired.json

Nếu đã chắc chắn, có thể dùng --in-place để ghi đè:

    python scripts/data/repair_master_entities.py \
        --input src/data/raw/elderly_commands_master.json \
        --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.entity_schema import canonicalize_entity_dict  # type: ignore  # noqa: E402
from data.processed.data_processor import DataProcessor  # type: ignore  # noqa: E402


@dataclass
class RepairStats:
    total: int = 0
    fixed_from_offsets: int = 0
    fixed_by_search: int = 0
    not_found: int = 0


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_span(
    text: str,
    entity_text: str,
    old_start: Optional[int],
    old_end: Optional[int],
) -> Optional[Tuple[int, int]]:
    """
    Tìm lại vị trí start/end của entity_text trong câu text mới.

    Chiến lược:
    1. Nếu old_start/old_end vẫn cắt ra đúng substring -> giữ nguyên.
    2. Nếu không, dùng search theo substring:
       - Ưu tiên lần xuất hiện gần old_start nhất (nếu có).
       - Nếu không có old_start, lấy lần xuất hiện đầu tiên.
    """
    entity_text = entity_text or ""
    if not entity_text:
        return None

    n = len(text)

    # 1) Thử dùng lại offset cũ nếu vẫn hợp lệ
    if isinstance(old_start, int) and isinstance(old_end, int):
        if 0 <= old_start < old_end <= n and text[old_start:old_end] == entity_text:
            return old_start, old_end

    # 2) Tìm tất cả vị trí khớp substring (case-sensitive)
    positions: List[int] = []
    start = 0
    while True:
        idx = text.find(entity_text, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        # Thử lại với tìm kiếm không phân biệt hoa thường
        lower_text = text.lower()
        lower_ent = entity_text.lower()
        start = 0
        while True:
            idx = lower_text.find(lower_ent, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

    if not positions:
        return None

    if isinstance(old_start, int) and old_start >= 0:
        # Chọn vị trí gần offset cũ nhất
        best = min(positions, key=lambda p: abs(p - old_start))
    else:
        best = positions[0]

    return best, best + len(entity_text)


def repair_entities_for_sample(
    sample: Dict[str, object], stats: RepairStats
) -> Dict[str, object]:
    text = str(sample.get("input", ""))
    entities_in = sample.get("entities", [])
    if not isinstance(entities_in, list):
        entities_in = []

    entities_out: List[Dict[str, object]] = []

    for raw_ent in entities_in:
        if not isinstance(raw_ent, dict):
            continue

        canonical = canonicalize_entity_dict(raw_ent)
        label = canonical.get("label") or canonical.get("type")
        ent_text = str(canonical.get("text", "") or "").strip()
        old_start = canonical.get("start")
        old_end = canonical.get("end")

        if not label or not ent_text:
            continue

        stats.total += 1
        span = _find_span(text, ent_text, old_start, old_end)
        if span is None:
            stats.not_found += 1
            # Giữ lại entity nhưng đánh dấu start/end = -1 để dễ debug sau
            canonical["start"] = -1
            canonical["end"] = -1
            entities_out.append(canonical)
            continue

        start, end = span
        canonical["start"] = start
        canonical["end"] = end
        # Cập nhật text theo substring thực tế (đảm bảo khớp chính tả mới)
        canonical["text"] = text[start:end]

        if isinstance(old_start, int) and isinstance(old_end, int):
            if 0 <= old_start < old_end <= len(text) and (old_start, old_end) == span:
                stats.fixed_from_offsets += 1
            else:
                stats.fixed_by_search += 1
        else:
            stats.fixed_by_search += 1

        entities_out.append(canonical)

    sample["entities"] = entities_out
    sample["spans"] = entities_out
    return sample


def repair_master(
    samples: List[Dict], processor: DataProcessor
) -> Tuple[List[Dict], RepairStats]:
    stats = RepairStats()
    repaired: List[Dict] = []

    for sample in samples:
        fixed = repair_entities_for_sample(sample, stats)
        # Sinh lại BIO labels từ spans chuẩn hoá
        fixed["bio_labels"] = processor.align_labels(
            fixed.get("input", ""), fixed.get("spans", [])
        )
        repaired.append(fixed)

    return repaired, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đồng bộ lại entities (label/text/start/end) + BIO cho elderly_commands_master.json sau khi sửa input."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Đường dẫn tới elderly_commands_master.json (hoặc file master tương đương).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output",
        type=Path,
        help="Đường dẫn file JSON sau khi sửa (khuyến nghị).",
    )
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Ghi đè trực tiếp lên file input (cẩn thận).",
    )
    args = parser.parse_args()

    samples = load_dataset(args.input)
    processor = DataProcessor()

    repaired, stats = repair_master(samples, processor)

    out_path = args.input if args.in_place else args.output
    assert out_path is not None
    save_dataset(out_path, repaired)

    print(f"Đã sửa {len(repaired)} mẫu và lưu tại: {out_path}")
    print(
        f"- Tổng số entity xử lý: {stats.total}\n"
        f"- Giữ nguyên offset cũ hợp lệ: {stats.fixed_from_offsets}\n"
        f"- Tìm lại bằng search substring: {stats.fixed_by_search}\n"
        f"- KHÔNG tìm thấy trong câu mới (start/end=-1): {stats.not_found}"
    )


if __name__ == "__main__":
    main()







