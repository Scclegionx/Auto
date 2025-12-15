"""Sinh câu paraphrase bằng cách thay thế từ/cụm từ đồng nghĩa có dấu."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple


DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "gọi": ["liên lạc", "alo", "thực hiện cuộc gọi"],
    "nhắn tin": ["gửi tin nhắn", "soạn tin", "viết tin nhắn"],
    "mở": ["bật", "khởi động"],
    "tắt": ["đóng", "ngừng"],
    "phát": ["bật", "chạy"],
    "tìm kiếm": ["tra cứu", "tìm", "kiếm"],
    "xem": ["mở xem", "hiển thị", "trình chiếu"],
    "giúp": ["hỗ trợ", "làm ơn", "giùm"],
    "lịch": ["sự kiện", "cuộc hẹn"],
}


def load_dataset(path: Path) -> List[MutableMapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset đầu vào phải là list các dict sample.")
    validated: List[MutableMapping[str, object]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"Sample index {idx} không phải dict.")
        validated.append(item)
    return validated


def load_synonyms(path: Path | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_SYNONYMS
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Synonym map phải là JSON object.")
    cleaned: Dict[str, List[str]] = {}
    for key, values in data.items():
        if isinstance(values, list):
            cleaned[str(key)] = [str(v) for v in values if isinstance(v, str) and v.strip()]
    return cleaned


def paraphrase_text(
    text: str,
    synonyms: Dict[str, List[str]],
    max_replacements: int,
) -> Tuple[str, int]:
    replacements_count = 0
    new_text = text

    # Tránh thay thế lặp vô hạn bằng cách random hóa thứ tự
    synonym_items = list(synonyms.items())
    random.shuffle(synonym_items)

    for base, options in synonym_items:
        if replacements_count >= max_replacements:
            break

        # Thử thay thế theo thứ tự ưu tiên
        if base in new_text:
            replacement = random.choice(options)
            new_text = new_text.replace(base, replacement, 1)
            replacements_count += 1
        elif base.capitalize() in new_text:
            replacement = random.choice(options).capitalize()
            new_text = new_text.replace(base.capitalize(), replacement, 1)
            replacements_count += 1
        elif base.upper() in new_text:
            replacement = random.choice(options).upper()
            new_text = new_text.replace(base.upper(), replacement, 1)
            replacements_count += 1

    return new_text, replacements_count


def generate_paraphrases_for_sample(
    sample: MutableMapping[str, object],
    synonyms: Dict[str, List[str]],
    max_variants: int,
    max_replacements: int,
) -> List[MutableMapping[str, object]]:
    text = str(sample.get("input") or "").strip()
    if not text:
        return []

    paraphrases: List[MutableMapping[str, object]] = []
    created = set()

    attempts = max_variants * 2
    while len(paraphrases) < max_variants and attempts > 0:
        attempts -= 1
        new_text, replaced = paraphrase_text(text, synonyms, max_replacements)
        if replaced == 0:
            continue
        if new_text == text:
            continue
        if new_text in created:
            continue

        variant = deepcopy(sample)
        variant["input"] = new_text
        created.add(new_text)
        paraphrases.append(variant)

    return paraphrases


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh paraphrase tiếng Việt dựa trên từ đồng nghĩa.",
    )
    parser.add_argument("input", type=Path, help="File JSON nguồn.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="File JSON ghi kết quả paraphrase.",
    )
    parser.add_argument(
        "--synonyms",
        type=Path,
        help="JSON từ điển đồng nghĩa. Nếu không truyền sẽ dùng bảng mặc định.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=2,
        help="Số paraphrase tối đa tạo cho mỗi sample (mặc định 2).",
    )
    parser.add_argument(
        "--max-replacements",
        type=int,
        default=2,
        help="Số cụm từ được phép thay thế trong mỗi câu (mặc định 2).",
    )
    parser.add_argument(
        "--drop-fields",
        nargs="*",
        default=["entities", "bio_labels"],
        help="Các trường sẽ loại bỏ trong sample mới để tránh lệch annotation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed cho random nhằm tái lập kết quả.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Giới hạn tổng số câu paraphrase được sinh ra.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    random.seed(args.seed)

    source_data = load_dataset(args.input)
    synonyms = load_synonyms(args.synonyms)

    paraphrased_samples: List[MutableMapping[str, object]] = []

    for sample in source_data:
        variants = generate_paraphrases_for_sample(
            sample,
            synonyms,
            args.max_variants,
            args.max_replacements,
        )
        for variant in variants:
            for field in args.drop_fields:
                variant.pop(field, None)
            paraphrased_samples.append(variant)
            if args.limit and len(paraphrased_samples) >= args.limit:
                break
        if args.limit and len(paraphrased_samples) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(paraphrased_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Tổng sample nguồn: {len(source_data)}")
    print(f"Tổng câu paraphrase: {len(paraphrased_samples)}")
    print(f"Đã ghi kết quả vào: {args.output}")


if __name__ == "__main__":
    main()


