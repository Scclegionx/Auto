"""Chuẩn hóa dữ liệu tiếng Việt: ép Unicode NFC, loại bỏ khoảng trắng dư
và thay thế các từ không dấu bằng bản có dấu theo từ điển tham chiếu.

Ví dụ sử dụng:
    python scripts/augment/normalize_vietnamese.py ^
        data/input.json --output data/output.json --fields input summary
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_FIELDS: Tuple[str, ...] = ("input",)


@dataclass
class NormalizeStats:
    total_samples: int = 0
    normalized_fields: int = 0
    replaced_tokens: int = 0
    samples_missing_accents: int = 0


def load_accent_dictionary(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy bảng chuyển dấu: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Accent dictionary phải là object (mapping).")
    return {str(k).lower(): str(v) for k, v in data.items()}


def normalize_text(
    text: str,
    accent_dict: Mapping[str, str],
) -> Tuple[str, int]:
    """Normalize string và trả về (text_mới, số_token_thay_thế)."""
    original = text
    normalized = unicodedata.normalize("NFC", original)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return normalized, 0

    replacements = 0
    tokens = normalized.split(" ")
    new_tokens: List[str] = []

    for token in tokens:
        key = _strip_punctuation(token).lower()
        replacement = accent_dict.get(key)

        if replacement:
            replacements += 1
            new_token = _apply_case(token, replacement)
        else:
            new_token = token
        new_tokens.append(new_token)

    normalized_text = " ".join(new_tokens)
    return normalized_text, replacements


def _strip_punctuation(token: str) -> str:
    return token.strip(".,!?;:\"'()[]{}")


def _apply_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template.istitle():
        return replacement.capitalize()
    return replacement


def contains_vietnamese_accents(text: str) -> bool:
    for char in text:
        if ord(char) > 127 and unicodedata.category(char).startswith("L"):
            return True
        # Ký tự kết hợp (ví dụ a + dấu sắc)
        if "COMBINING" in unicodedata.name(char, ""):
            return True
    return False


def normalize_sample(
    sample: MutableMapping[str, object],
    fields: Sequence[str],
    accent_dict: Mapping[str, str],
    normalize_entities: bool,
    stats: NormalizeStats,
) -> None:
    stats.total_samples += 1

    for field in fields:
        value = sample.get(field)
        if isinstance(value, str):
            normalized, replacements = normalize_text(value, accent_dict)
            if normalized != value:
                sample[field] = normalized
                stats.normalized_fields += 1
                stats.replaced_tokens += replacements
            if not contains_vietnamese_accents(normalized):
                stats.samples_missing_accents += 1

    if normalize_entities:
        entities = sample.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, MutableMapping):
                    text_value = entity.get("text")
                    if isinstance(text_value, str):
                        normalized, replacements = normalize_text(text_value, accent_dict)
                        if normalized != text_value:
                            entity["text"] = normalized
                            stats.replaced_tokens += replacements


def load_dataset(path: Path) -> List[MutableMapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset phải là mảng JSON chứa các sample dict.")
    validated: List[MutableMapping[str, object]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"Sample index {idx} không phải dict.")
        validated.append(item)
    return validated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa văn bản tiếng Việt (Unicode NFC, dấu đầy đủ)",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Đường dẫn file JSON cần chuẩn hóa.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Đường dẫn ghi kết quả (mặc định ghi đè file gốc).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help="Danh sách trường dạng chuỗi cần chuẩn hóa (mặc định: input).",
    )
    parser.add_argument(
        "--accent-dict",
        type=Path,
        help="JSON mapping từ không dấu -> có dấu để thay thế tự động.",
    )
    parser.add_argument(
        "--normalize-entities",
        action="store_true",
        help="Chuẩn hóa thêm trường entities[].text nếu có.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ thống kê, không ghi file đầu ra.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path: Path = args.input
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {dataset_path}")

    accent_dict = load_accent_dictionary(args.accent_dict)
    data = load_dataset(dataset_path)

    stats = NormalizeStats()
    for sample in data:
        normalize_sample(
            sample,
            args.fields,
            accent_dict,
            args.normalize_entities,
            stats,
        )

    print(f"Tổng sample xử lý: {stats.total_samples}")
    print(f"Số trường đã normalize: {stats.normalized_fields}")
    print(f"Số token thay thế bằng từ điển: {stats.replaced_tokens}")
    if stats.samples_missing_accents:
        print(
            f"Cảnh báo: {stats.samples_missing_accents} sample vẫn thiếu dấu. "
            "Vui lòng kiểm tra thủ công."
        )

    if args.dry_run:
        print("Dry-run hoàn tất, không ghi file mới.")
        return

    output_path: Path = args.output or dataset_path
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã ghi dữ liệu chuẩn hóa tới: {output_path}")


if __name__ == "__main__":
    main()


