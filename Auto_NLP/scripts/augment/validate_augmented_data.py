"""Kiểm tra chất lượng dataset augmented: cấu trúc, dấu tiếng Việt, trùng lặp."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping


REQUIRED_FIELDS = ("input", "intent", "command")


@dataclass
class ValidationIssue:
    sample_index: int
    message: str
    sample_preview: str = ""


@dataclass
class ValidationReport:
    total_samples: int = 0
    missing_required: List[ValidationIssue] = field(default_factory=list)
    no_accents: List[ValidationIssue] = field(default_factory=list)
    duplicate_inputs: List[ValidationIssue] = field(default_factory=list)
    invalid_entities: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "missing_required": [issue.__dict__ for issue in self.missing_required],
            "no_accents": [issue.__dict__ for issue in self.no_accents],
            "duplicate_inputs": [issue.__dict__ for issue in self.duplicate_inputs],
            "invalid_entities": [issue.__dict__ for issue in self.invalid_entities],
        }

    def has_errors(self) -> bool:
        return any(
            [
                self.missing_required,
                self.no_accents,
                self.duplicate_inputs,
                self.invalid_entities,
            ]
        )


def contains_vietnamese_accents(text: str) -> bool:
    for char in text:
        if ord(char) > 127 and unicodedata.category(char).startswith("L"):
            return True
        if "COMBINING" in unicodedata.name(char, ""):
            return True
    return False


def load_dataset(path: Path) -> List[MutableMapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset phải là list.")
    validated: List[MutableMapping[str, object]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"Sample index {idx} không phải dict.")
        validated.append(item)
    return validated


def validate_dataset(data: List[MutableMapping[str, object]]) -> ValidationReport:
    report = ValidationReport(total_samples=len(data))
    input_counter: Counter[str] = Counter()

    for idx, sample in enumerate(data):
        preview = str(sample.get("input") or "")[:120]

        # Kiểm tra trường bắt buộc
        for field in REQUIRED_FIELDS:
            if field not in sample or not str(sample[field]).strip():
                report.missing_required.append(
                    ValidationIssue(idx, f"Thiếu trường {field}", preview)
                )

        input_text = str(sample.get("input") or "")
        if input_text:
            input_counter[input_text] += 1
            if not contains_vietnamese_accents(input_text):
                report.no_accents.append(
                    ValidationIssue(idx, "Câu không chứa dấu tiếng Việt", preview)
                )

        # Kiểm tra entity nếu có
        entities = sample.get("entities")
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, MutableMapping):
                    report.invalid_entities.append(
                        ValidationIssue(idx, "Entity không phải dict", preview)
                    )
                    continue
                if not ent.get("label"):
                    report.invalid_entities.append(
                        ValidationIssue(idx, "Entity thiếu label", preview)
                    )
                if not ent.get("text"):
                    report.invalid_entities.append(
                        ValidationIssue(idx, "Entity thiếu text", preview)
                    )

    for idx, sample in enumerate(data):
        input_text = str(sample.get("input") or "")
        if input_counter[input_text] > 1:
            report.duplicate_inputs.append(
                ValidationIssue(idx, "Câu trùng với sample khác", input_text[:120])
            )

    return report


def save_report(report: ValidationReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá chất lượng dataset augmented (dấu, trùng lặp, entity).",
    )
    parser.add_argument("input", type=Path, help="File JSON cần kiểm tra.")
    parser.add_argument(
        "--save-report",
        type=Path,
        help="Ghi báo cáo chi tiết ra file JSON.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Trả về mã lỗi khác 0 nếu phát hiện vấn đề.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = load_dataset(args.input)
    report = validate_dataset(data)

    print(f"Tổng số sample: {report.total_samples}")
    print(f"Số sample thiếu trường bắt buộc: {len(report.missing_required)}")
    print(f"Số sample không có dấu: {len(report.no_accents)}")
    print(f"Số sample trùng câu: {len(report.duplicate_inputs)}")
    print(f"Số entity lỗi: {len(report.invalid_entities)}")

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report(report, args.save_report)
        print(f"Đã lưu báo cáo tại: {args.save_report}")

    if args.fail_on_warning and report.has_errors():
        raise SystemExit(1)


if __name__ == "__main__":
    main()


