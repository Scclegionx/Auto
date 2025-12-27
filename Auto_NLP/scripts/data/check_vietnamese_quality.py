#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiểm tra chất lượng câu tiếng Việt trong dataset (train/val/test).

Các tiêu chí cảnh báo:
  - Câu quá ngắn (ít hơn 3 từ).
  - Không có ký tự có dấu (accent) dù có >= 10 ký tự chữ.
  - Tỷ lệ từ không chứa nguyên âm cao (có thể do nhận dạng sai).
  - Lặp ký tự bất thường (ví dụ "aaaaa", "!!!!!").
  - Xuất hiện ký tự lạ ngoài bảng chữ cái/punctuation phổ biến.

Usage:
    python scripts/data/check_vietnamese_quality.py \
        --paths src/data/processed/train.json src/data/processed/val.json src/data/processed/test.json \
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ACCENTED_CHARS = set(
    "àáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)
BASE_VOWELS = set("aeiouy")
EXTENDED_VOWELS = set("ăâêôơư")
ALLOWED_PUNCT = set(" ,.;:!?-'\"()[]{}…“”‘’/@")
WEIRD_CHAR_PATTERN = re.compile(r"[^\w\sÀ-Ỵà-ỵ.,;:!?\"'()\\[\\]{}…“”‘’/@-]")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{4,}")


def load_dataset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_accent(text: str) -> bool:
    return any(ch in ACCENTED_CHARS for ch in text)


def _is_numeric_token(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _strip_accents(token: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", token) if unicodedata.category(ch) != "Mn"
    )


def word_has_vowel(word: str) -> bool:
    if not word:
        return False
    normalized = _strip_accents(word)
    for ch in normalized:
        if not ch.isalpha():
            continue
        lower = ch.lower()
        if lower in BASE_VOWELS or lower in EXTENDED_VOWELS:
            return True
    return False


def analyze_text(text: str) -> List[str]:
    warnings: List[str] = []
    stripped = text.strip()
    words = stripped.split()

    if len(words) < 3:
        warnings.append("too_short")

    alpha_chars = sum(1 for ch in stripped if ch.isalpha())
    if alpha_chars >= 10 and not has_accent(stripped):
        warnings.append("no_accent")

    if len(words) >= 4:
        words_no_vowel = [
            w
            for w in words
            if len(w) > 2 and not _is_numeric_token(w) and not word_has_vowel(w)
        ]
        if words_no_vowel and len(words_no_vowel) / len(words) >= 0.4:
            warnings.append("many_words_without_vowel")

    if REPEATED_CHAR_PATTERN.search(stripped):
        warnings.append("repeated_characters")

    if WEIRD_CHAR_PATTERN.search(stripped):
        warnings.append("strange_characters")

    return warnings


def analyze_file(path: Path) -> Dict[str, List[Dict[str, str]]]:
    issues: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    data = load_dataset(path)
    for idx, sample in enumerate(data):
        text = sample.get("input", "")
        command = sample.get("command", sample.get("intent", "unknown"))
        warnings = analyze_text(text)
        for warning in warnings:
            issues[warning].append(
                {
                    "index": idx,
                    "command": command,
                    "text": text,
                }
            )
    return issues


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng câu tiếng Việt.")
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
        help="Số ví dụ tối đa in ra cho mỗi loại cảnh báo.",
    )
    args = parser.parse_args()

    total_files = 0
    global_issues: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: File không tồn tại: {path}")
            continue
        total_files += 1
        file_issues = analyze_file(path)
        print(f"\n=== {path} ===")
        if not file_issues:
            print("Không phát hiện cảnh báo nào.")
            continue

        for issue, samples in file_issues.items():
            print(f"- {issue}: {len(samples)} mẫu")
            for sample in samples[: args.limit]:
                preview = sample["text"]
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                print(f"    • [{sample['command']}] #{sample['index']}: {preview}")
            global_issues[issue].extend(
                {"file": str(path), **sample} for sample in samples
            )

    print("\n=== Tổng hợp ===")
    if not global_issues:
        print("Không có cảnh báo.")
        return

    for issue, samples in global_issues.items():
        print(f"- {issue}: {len(samples)} mẫu trên {total_files} file")


if __name__ == "__main__":
    # Cho phép chạy trực tiếp: python scripts/data/check_vietnamese_quality.py --paths ...
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()

