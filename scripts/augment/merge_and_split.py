"""Gộp các file dataset, khử trùng lặp và chia train/val/test theo stratified command."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple


@dataclass
class SplitRatios:
    train: float
    val: float
    test: float

    def normalized(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Tổng tỉ lệ split phải > 0.")
        return SplitRatios(self.train / total, self.val / total, self.test / total)


def load_dataset(path: Path) -> List[MutableMapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} không phải list.")
    validated: List[MutableMapping[str, object]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"{path} - sample {idx} không phải dict.")
        validated.append(item)
    return validated


def merge_datasets(paths: Sequence[Path]) -> List[MutableMapping[str, object]]:
    combined: List[MutableMapping[str, object]] = []
    for path in paths:
        combined.extend(load_dataset(path))
    return combined


def deduplicate_by_input(data: List[MutableMapping[str, object]]) -> List[MutableMapping[str, object]]:
    seen = set()
    unique: List[MutableMapping[str, object]] = []
    for sample in data:
        key = str(sample.get("input") or "")
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    return unique


def stratified_split(
    data: List[MutableMapping[str, object]],
    ratios: SplitRatios,
    seed: int,
) -> Tuple[List[MutableMapping[str, object]], List[MutableMapping[str, object]], List[MutableMapping[str, object]]]:
    buckets: Dict[str, List[MutableMapping[str, object]]] = defaultdict(list)
    for sample in data:
        command = str(sample.get("command") or "unknown")
        buckets[command].append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    ratios = ratios.normalized()

    for command, items in buckets.items():
        rng.shuffle(items)
        total = len(items)
        n_train = int(total * ratios.train)
        n_val = int(total * ratios.val)
        n_test = total - n_train - n_val
        if n_test < 0:
            n_test = 0
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:n_train + n_val + n_test])
        # Nếu còn dư (do làm tròn), đẩy vào train
        remainder = items[n_train + n_val + n_test:]
        train.extend(remainder)
        if remainder:
            print(f"Cảnh báo: command '{command}' dư {len(remainder)} mẫu, đưa vào train.")

    return train, val, test


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge + dedup dataset, optional stratified split.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Danh sách file JSON sẽ gộp (theo thứ tự ưu tiên: file đứng trước giữ bản gốc khi trùng).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="File JSON lưu dataset đã gộp (nếu không split).",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        help="File JSON lưu train split (khi bật --split).",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        help="File JSON lưu val split (khi bật --split).",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        help="File JSON lưu test split (khi bật --split).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Bật stratified split theo command (yêu cầu cung cấp train/val/test output).",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Tỉ lệ split (mặc định 0.8 0.1 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed cho random khi shuffle và split.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dữ liệu sau khi gộp (mặc định bật khi không split).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    combined = merge_datasets(args.inputs)
    deduped = deduplicate_by_input(combined)

    print(f"Tổng sample sau gộp: {len(combined)}")
    print(f"Sau khử trùng lặp: {len(deduped)}")

    random.seed(args.seed)
    if args.split:
        if not (args.train_output and args.val_output and args.test_output):
            raise ValueError("Cần cung cấp train/val/test output khi bật --split.")
        ratios = SplitRatios(*args.ratios)
        train, val, test = stratified_split(deduped, ratios, args.seed)
        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        _write_json(args.train_output, train)
        _write_json(args.val_output, val)
        _write_json(args.test_output, test)
    else:
        should_shuffle = (not args.split) or args.shuffle
        if should_shuffle:
            random.shuffle(deduped)
        _write_json(args.output, deduped)
        print(f"Đã ghi dataset gộp vào: {args.output}")


def _write_json(path: Path, data: List[MutableMapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


