"""Sinh mẫu mới bằng cách hoán đổi entity theo từ điển giá trị hợp lệ."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple


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


def load_entity_vocab(path: Path) -> Dict[str, List[str]]:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError("Entity vocab phải là JSON object.")
    cleaned: Dict[str, List[str]] = {}
    for label, values in vocab.items():
        if isinstance(values, list):
            normalized_values = [
                str(value) for value in values if isinstance(value, str) and value.strip()
            ]
            if normalized_values:
                cleaned[str(label)] = normalized_values
    return cleaned


def generate_variants_for_sample(
    sample: MutableMapping[str, object],
    vocab: Dict[str, List[str]],
    max_variants: int,
) -> List[MutableMapping[str, object]]:
    entities = sample.get("entities")
    if not isinstance(entities, list) or not entities:
        return []

    original_entities = [deepcopy(e) for e in entities if isinstance(e, MutableMapping)]
    if not original_entities:
        return []

    choice_matrix: List[List[str]] = []
    original_tuple: List[str] = []
    has_options = False

    for ent in original_entities:
        text = str(ent.get("text") or "")
        original_tuple.append(text)
        label = str(ent.get("label") or "")
        candidates = [value for value in vocab.get(label, []) if value != text]
        choice_row = [text] + candidates
        choice_matrix.append(choice_row)
        if candidates:
            has_options = True

    if not has_options:
        return []

    augmented: List[MutableMapping[str, object]] = []

    counter = 0
    for combo in product(*choice_matrix):
        if list(combo) == original_tuple:
            continue
        new_sample = apply_replacements(sample, original_entities, combo)
        augmented.append(new_sample)
        counter += 1
        if counter >= max_variants:
            break

    return augmented


def apply_replacements(
    base_sample: MutableMapping[str, object],
    entities_template: List[MutableMapping[str, object]],
    replacements: Sequence[str],
) -> MutableMapping[str, object]:
    new_sample = deepcopy(base_sample)
    text = str(new_sample.get("input") or "")
    entities = [deepcopy(ent) for ent in entities_template]

    # Sắp xếp theo start để cập nhật offset chính xác
    order = sorted(
        range(len(entities)),
        key=lambda idx: entities[idx].get("start", idx),
    )

    offset = 0
    for idx in order:
        ent = entities[idx]
        new_text = replacements[idx]
        old_text = str(ent.get("text") or "")

        start = ent.get("start")
        end = ent.get("end")

        if isinstance(start, int) and isinstance(end, int) and start <= end:
            start += offset
            end += offset
            text = text[:start] + new_text + text[end:]
            diff = len(new_text) - (end - start)
            ent["start"] = start
            ent["end"] = start + len(new_text)
            offset += diff
        else:
            text, start_idx = replace_first_occurrence(text, old_text, new_text)
            ent.pop("start", None)
            ent.pop("end", None)
            if start_idx != -1:
                ent["start"] = start_idx
                ent["end"] = start_idx + len(new_text)

        ent["text"] = new_text

    new_sample["input"] = text
    new_sample["entities"] = entities
    return new_sample


def replace_first_occurrence(text: str, old: str, new: str) -> Tuple[str, int]:
    if not old:
        return text, -1
    idx = text.find(old)
    if idx == -1:
        return text, -1
    return text[:idx] + new + text[idx + len(old):], idx


def deduplicate_samples(samples: Iterable[MutableMapping[str, object]]) -> List[MutableMapping[str, object]]:
    seen_inputs = set()
    unique: List[MutableMapping[str, object]] = []
    for sample in samples:
        input_text = str(sample.get("input") or "")
        if input_text not in seen_inputs:
            seen_inputs.add(input_text)
            unique.append(sample)
    return unique


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hoán đổi entity để tạo mẫu mới (giữ nguyên cấu trúc intent/command).",
    )
    parser.add_argument("input", type=Path, help="File JSON nguồn.")
    parser.add_argument(
        "--entity-vocab",
        type=Path,
        required=True,
        help="JSON từ điển entity_label -> danh sách giá trị có dấu.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="File JSON ghi kết quả augmentation.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=3,
        help="Số biến thể tối đa tạo ra cho mỗi sample (mặc định 3).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Giới hạn tổng số sample augmented (mặc định không giới hạn).",
    )
    parser.add_argument(
        "--drop-fields",
        nargs="*",
        default=["bio_labels"],
        help="Các trường sẽ bị loại khỏi sample augmented.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed cho random nhằm tái lập kết quả (mặc định 2025).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    random.seed(args.seed)

    source_data = load_dataset(args.input)
    entity_vocab = load_entity_vocab(args.entity_vocab)

    augmented_samples: List[MutableMapping[str, object]] = []

    for sample in source_data:
        variants = generate_variants_for_sample(sample, entity_vocab, args.max_variants)
        for variant in variants:
            for field in args.drop_fields:
                variant.pop(field, None)
            augmented_samples.append(variant)
            if args.sample_limit and len(augmented_samples) >= args.sample_limit:
                break
        if args.sample_limit and len(augmented_samples) >= args.sample_limit:
            break

    deduped = deduplicate_samples(augmented_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Tổng sample nguồn: {len(source_data)}")
    print(f"Số sample augmented (trước dedup): {len(augmented_samples)}")
    print(f"Số sample augmented (sau dedup): {len(deduped)}")
    print(f"Đã ghi kết quả vào: {args.output}")


if __name__ == "__main__":
    main()


