#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo biến thể dữ liệu phù hợp với đầu vào giọng nói (ASR) cho dataset Auto_NLP.

Các phép biến đổi được hỗ trợ:
  - Hoán đổi định dạng số (digit ↔ chữ) cho PHONE/TIME/DATE/QUERY.
  - Tiêm filler (ờ/à/nè/giúp tôi/...) ở đầu hoặc cuối câu.
  - Bỏ dấu câu.
  - Thay thế từ đồng nghĩa hẹp theo ngữ cảnh.
  - Tiêm lỗi chính tả nhẹ (1 ký tự) ở vùng không phải entity.

Script sẽ giữ nguyên cấu trúc entity theo schema mới (REMINDER_CONTENT, FREQUENCY...),
đồng thời tái tạo spans & bio labels thông qua DataProcessor.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.entity_schema import canonicalize_entity_dict, canonicalize_entity_label  # noqa: E402
from data.processed.data_processor import DataProcessor  # noqa: E402

# --------------------------- Hằng số & cấu hình ---------------------------

RANDOM = random.Random(42)

NUMBER_WORDS = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}
WORD_TO_DIGIT = {v: k for k, v in NUMBER_WORDS.items()}

FILLER_PHRASES = ["ờ", "à", "nè", "giúp tôi", "làm ơn", "mình nhờ", "cho tôi", "xin phép", "nhé", "nhá", "ạ"]

VERB_TRIGGERS = [
    "gửi",
    "nhắn",
    "soạn",
    "bật",
    "tắt",
    "mở",
    "đặt",
    "nhắc",
    "gọi",
    "phát",
    "chạy",
]

SYNONYM_MAP = {
    "gửi": "nhắn",
    "nhắn": "gửi",
    "soạn": "viết",
    "bật": "mở",
    "tắt": "đóng",
    "phát": "chạy",
    "chạy": "phát",
    "video": "gọi hình",
    "gọi hình": "video",
}

PUNCTUATION_PATTERN = re.compile(r"[\,\.\!\?\;:]")

ENTITY_NUMBER_LABELS = {"PHONE", "TIME", "DATE", "QUERY", "FREQUENCY"}

COMMAND_CONSTRAINTS = {
    "send-mess": {"MESSAGE"},
    "set-alarm": {"TIME"},
    "control-device": {"DEVICE"},
    "search-internet": {"QUERY"},
    "search-youtube": {"QUERY"},
}


# --------------------------- Hàm tiện ích ---------------------------

def segment_text(text: str, spans: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Chia text thành các segment liền kề theo spans (entity / non-entity)."""
    sorted_spans = sorted(
        [canonicalize_entity_dict(span) for span in spans],
        key=lambda s: s.get("start", 0),
    )
    segments: List[Dict[str, object]] = []
    cursor = 0
    for span in sorted_spans:
        start = int(span["start"])
        end = int(span["end"])
        label = canonicalize_entity_label(span.get("label", ""))
        if cursor < start:
            segments.append({"text": text[cursor:start], "label": None})
        segments.append({"text": text[start:end], "label": label})
        cursor = end
    if cursor < len(text):
        segments.append({"text": text[cursor:], "label": None})
    return segments


def reassemble_segments(segments: Sequence[Dict[str, object]]) -> Tuple[str, List[Dict[str, object]]]:
    """Gộp segment thành text mới và cập nhật spans."""
    new_text_parts: List[str] = []
    new_spans: List[Dict[str, object]] = []
    cursor = 0
    for segment in segments:
        segment_text = segment["text"]
        new_text_parts.append(segment_text)
        label = segment.get("label")
        if label:
            start = cursor
            end = cursor + len(segment_text)
            new_spans.append(
                {
                    "label": label,
                    "text": segment_text,
                    "start": start,
                    "end": end,
                }
            )
        cursor += len(segment_text)
    return "".join(new_text_parts), new_spans


def convert_digits_to_words(raw: str) -> str:
    return " ".join(NUMBER_WORDS.get(ch, ch) for ch in raw if ch.strip())


def convert_words_to_digits(raw: str) -> Optional[str]:
    tokens = raw.split()
    if not tokens:
        return None
    digits = []
    for token in tokens:
        token_norm = unicodedata.normalize("NFC", token.lower())
        if token_norm not in WORD_TO_DIGIT:
            return None
        digits.append(WORD_TO_DIGIT[token_norm])
    return "".join(digits)


def swap_number_format(segment_text: str) -> str:
    """Nếu là chuỗi digits → chuyển sang chữ; nếu toàn chữ số → chuyển về digits."""
    leading_ws = len(segment_text) - len(segment_text.lstrip())
    trailing_ws = len(segment_text) - len(segment_text.rstrip())
    core = segment_text.strip()
    if not core:
        return segment_text
    new_core: Optional[str] = None
    if core.isdigit():
        new_core = convert_digits_to_words(core)
    else:
        converted = convert_words_to_digits(core)
        if converted:
            new_core = converted
    if not new_core:
        return segment_text
    return (" " * leading_ws) + new_core + (" " * trailing_ws)


def inject_filler(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not segments:
        return segments
    filler = RANDOM.choice(FILLER_PHRASES)
    position = RANDOM.choice(["prefix", "suffix", "verb"])
    new_segments = deepcopy(segments)
    filler_token = f"{filler} "
    if position == "prefix":
        new_segments.insert(0, {"text": filler_token, "label": None})
    elif position == "suffix":
        last = new_segments[-1]
        if last.get("label"):
            new_segments.append({"text": " " + filler, "label": None})
        else:
            last_text = last["text"]
            new_segments[-1] = {"text": f"{last_text.rstrip()} {filler}", "label": None}
    else:  # near verb
        for idx, segment in enumerate(new_segments):
            if segment.get("label"):
                continue
            if any(re.search(rf"\b{verb}\b", segment["text"], flags=re.IGNORECASE) for verb in VERB_TRIGGERS):
                new_segments.insert(idx, {"text": filler_token, "label": None})
                break
        else:
            new_segments.insert(0, {"text": filler_token, "label": None})
    return new_segments


def remove_punctuation(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for segment in segments:
        if segment.get("label"):
            cleaned.append(segment)
        else:
            cleaned.append(
                {
                    "text": PUNCTUATION_PATTERN.sub("", segment["text"]),
                    "label": None,
                }
            )
    return cleaned


def replace_synonyms(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    replaced: List[Dict[str, object]] = []
    for segment in segments:
        if segment.get("label"):
            replaced.append(segment)
            continue
        text = segment["text"]
        new_text = text
        for src, tgt in SYNONYM_MAP.items():
            pattern = r"\b" + re.escape(src) + r"\b"
            if re.search(pattern, new_text, flags=re.IGNORECASE):
                new_text = re.sub(pattern, tgt, new_text, flags=re.IGNORECASE)
        replaced.append({"text": new_text, "label": None})
    return replaced


def inject_typo(segments: List[Dict[str, object]], probability: float = 0.2) -> List[Dict[str, object]]:
    mutated = []
    for segment in segments:
        if segment.get("label") or RANDOM.random() > probability:
            mutated.append(segment)
            continue
        text = list(segment["text"])
        if not text:
            mutated.append(segment)
            continue
        idx = RANDOM.randrange(len(text))
        char = text[idx]
        if char.isalpha():
            text[idx] = RANDOM.choice(["f", "j", "w", "z"]) if char.islower() else RANDOM.choice(["F", "J", "W", "Z"])
        mutated.append({"text": "".join(text), "label": None})
    return mutated


def drop_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    filtered = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", filtered)


def diacritic_dropout(segments: List[Dict[str, object]], drop_ratio: float = 0.25) -> List[Dict[str, object]]:
    new_segments: List[Dict[str, object]] = []
    for segment in segments:
        if segment.get("label") or RANDOM.random() > drop_ratio:
            new_segments.append(segment)
        else:
            new_segments.append({"text": drop_diacritics(segment["text"]), "label": None})
    return new_segments


def augment_segments(
    segments: List[Dict[str, object]],
    operations: Sequence[str],
) -> List[List[Dict[str, object]]]:
    """Áp dụng chuỗi phép biến đổi, trả về danh sách segment variants."""
    variants: List[List[Dict[str, object]]] = []
    base = deepcopy(segments)

    for op in operations:
        current = deepcopy(base)
        if op == "number_swap":
            for segment in current:
                label = segment.get("label")
                if label and label.split("-")[-1] in ENTITY_NUMBER_LABELS:
                    segment["text"] = swap_number_format(segment["text"])
        elif op == "filler":
            current = inject_filler(current)
        elif op == "punctuation":
            current = remove_punctuation(current)
        elif op == "synonym":
            current = replace_synonyms(current)
        elif op == "noise":
            current = inject_typo(current)
        elif op in {"diacritics", "diacritic_dropout"}:
            current = diacritic_dropout(current)
        else:
            continue
        variants.append(current)
    return variants


def regenerate_entities(spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "label": canonicalize_entity_label(span["label"]),
            "text": span["text"],
            "start": int(span["start"]),
            "end": int(span["end"]),
        }
        for span in spans
    ]


def get_spans_from_sample(sample: Dict[str, object]) -> List[Dict[str, object]]:
    if sample.get("spans"):
        return [canonicalize_entity_dict(span) for span in sample["spans"]]
    raw_entities = sample.get("entities") or []
    spans: List[Dict[str, object]] = []
    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue
        if not entity.get("text"):
            continue
        spans.append(
            {
                "label": entity.get("label"),
                "text": entity.get("text"),
                "start": entity.get("start", -1),
                "end": entity.get("end", -1),
            }
        )
    return spans


def augment_sample(sample: Dict[str, object], processor: DataProcessor, operations: Sequence[str]) -> List[Dict[str, object]]:
    text = sample["input"]
    spans = get_spans_from_sample(sample)
    if not spans:
        return []
    segments = segment_text(text, spans)

    augmented_samples: List[Dict[str, object]] = []
    variants = augment_segments(segments, operations)

    for variant in variants:
        new_text, new_spans = reassemble_segments(variant)

        if not new_text.strip():
            continue

        bio_labels = processor.align_labels(new_text, new_spans)
        if not validate_required_entities(sample.get("command"), new_spans):
            continue
        augmented_samples.append(
            {
                "input": new_text,
                "command": sample.get("command"),
                "intent": sample.get("intent", sample.get("command")),
                "entities": regenerate_entities(new_spans),
                "spans": regenerate_entities(new_spans),
                "bio_labels": bio_labels,
                "meta": {
                    "source": "augment_vi_asr",
                    "base_id": sample.get("id"),
                    "ops": operations,
                },
            }
        )
    return augmented_samples


def load_dataset(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(path: Path, samples: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_required_entities(command: Optional[str], spans: Sequence[Dict[str, object]]) -> bool:
    if not command:
        return True
    required = COMMAND_CONSTRAINTS.get(command)
    if not required:
        return True
    labels = {canonicalize_entity_label(span.get("label", "")) for span in spans}
    return all(req in labels for req in required)


def main():
    parser = argparse.ArgumentParser(description="Augment dataset với biến thể phù hợp voice/ASR.")
    parser.add_argument("--input", type=Path, required=True, help="Đường dẫn dataset gốc (processed JSON).")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn lưu dataset augment.")
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["number_swap", "filler", "punctuation", "synonym", "noise"],
        help="Danh sách phép biến đổi cần áp dụng.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu gốc để augment (debug).",
    )
    args = parser.parse_args()

    samples = load_dataset(args.input)
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor = DataProcessor()
    augmented: List[Dict[str, object]] = []

    for sample in samples:
        augmented.extend(augment_sample(sample, processor, args.operations))

    print(f"Generated {len(augmented)} augmented samples from {len(samples)} originals.")
    save_dataset(args.output, augmented)


if __name__ == "__main__":
    main()

