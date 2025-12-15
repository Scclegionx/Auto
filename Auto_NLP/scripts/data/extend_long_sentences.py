#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kéo dài câu (thêm ngữ cảnh) để tăng tỷ lệ câu > 25 token.

Mỗi intent sẽ được thêm hậu tố dài, không tạo entity mới.

Usage:
    python scripts/data/extend_long_sentences.py \
        --input artifacts/balanced/train_balanced.json \
        --output artifacts/balanced/train_balanced_long.json \
        --ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.processed.data_processor import DataProcessor  # noqa: E402
from data.entity_schema import canonicalize_entity_dict  # noqa: E402

LONG_SUFFIX = {
    "call": ", nhớ giúp tôi gọi cho họ để bàn chuyện gia đình tối nay nhé, đừng bỏ sót giùm tôi nha.",
    "send-mess": ", nhớ ghi rõ như vậy để họ khỏi hiểu nhầm, tôi nhờ trợ lý giúp kỹ một chút nha.",
    "search-internet": ", tôi muốn tìm càng chi tiết càng tốt để chuẩn bị trước, nhớ kiểm tra kỹ cho tôi nhé.",
    "search-youtube": ", nhớ tìm video hướng dẫn thật dễ hiểu vì tôi mới tập làm thôi, giúp tôi nha.",
    "get-info": ", nếu có thông tin mới nhất thì nhắc tôi liền để tôi còn chuẩn bị kịp thời nha trợ lý.",
    "set-alarm": ", để sáng mai tôi còn dậy sớm uống thuốc và tập thể dục cho đúng lời bác sĩ dặn nha.",
    "control-device": ", bật nhẹ nhàng thôi vì tôi đang nghỉ ngơi, đừng làm ồn nha cho tôi dễ chịu.",
    "open-cam": ", giúp tôi chuyển sang chế độ quay rõ nét để tôi chụp hình gửi cho con cháu xem nhé.",
    "add-contacts": ", nhớ lưu kỹ để mai mốt tôi gọi khỏi phải tìm lại mất công nữa đó nghe trợ lý.",
    "make-video-call": ", vì tôi muốn nhìn mặt con cháu cho đỡ nhớ nên làm giùm tôi cẩn thận nha.",
}


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def extend_samples(samples: List[Dict], ratio: float, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    processor = DataProcessor()
    by_command: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        command = sample.get("command", sample.get("intent", "unknown"))
        by_command.setdefault(command, []).append(idx)

    for command, indices in by_command.items():
        suffix = LONG_SUFFIX.get(command)
        if not suffix:
            continue
        num_extend = max(1, int(len(indices) * ratio))
        candidates = [idx for idx in indices if len(samples[idx]["input"].split()) <= 22]
        if len(candidates) < num_extend:
            candidates = indices
        selected = rng.sample(candidates, min(num_extend, len(candidates)))

        for idx in selected:
            sample = samples[idx].copy()
            original_text = sample["input"]
            sample["input"] = f"{original_text.strip()} {suffix}"
            spans = [canonicalize_entity_dict(e) for e in sample.get("entities", [])]
            sample["entities"] = spans
            sample["spans"] = spans
            sample["bio_labels"] = processor.align_labels(sample["input"], spans)
            samples[idx] = sample

    return samples


def main():
    parser = argparse.ArgumentParser(description="Tạo câu dài hơn.")
    parser.add_argument("--input", type=Path, required=True, help="Dataset cân bằng đầu vào.")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn lưu dataset sau khi kéo dài.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Tỷ lệ mẫu mỗi intent cần kéo dài.")
    args = parser.parse_args()

    samples = load_dataset(args.input)
    extended = extend_samples(samples, args.ratio)
    save_dataset(args.output, extended)
    print(f"Saved {len(extended)} samples with longer sentences to {args.output}")


if __name__ == "__main__":
    main()

