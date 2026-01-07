from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_stats(path: Path, label_keys: Tuple[str, ...] = ("intent", "command")) -> Dict[str, Counter]:
    """
    Đọc file JSON dạng mảng [ {..}, {..}, ... ] và thống kê:
      - tổng số mẫu
      - phân bố theo các trường trong label_keys (vd: intent, command)
    """
    counts: Dict[str, Counter] = {key: Counter() for key in label_keys}

    with path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total = len(data)
    for sample in data:
        # Phòng trường hợp phần tử không phải dict (rất hiếm)
        if not isinstance(sample, dict):
            continue
        for key in label_keys:
            value = sample.get(key, "UNKNOWN")
            counts[key][value] += 1

    counts["__total__"] = Counter({"total": total})
    return counts


def print_stats(name: str, stats: Dict[str, Counter]) -> None:
    total = stats.get("__total__", Counter()).get("total", 0)
    print(f"\n=== {name} ===")
    print(f"Tổng số mẫu: {total}")
    for key, counter in stats.items():
        if key == "__total__":
            continue
        print(f"\nTop {key}:")
        for label, cnt in counter.most_common(10):
            print(f"  {label}: {cnt}")


def main() -> None:
    data_dir = PROJECT_ROOT / "src" / "data" / "processed"
    train_path = data_dir / "train.json"
    train_clean_path = data_dir / "train.clean.json"

    label_keys = ("intent", "command")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Đang đọc:\n  - {train_path}\n  - {train_clean_path}")

    stats_train = load_stats(train_path, label_keys=label_keys)
    stats_clean = load_stats(train_clean_path, label_keys=label_keys)

    print_stats("train.json", stats_train)
    print_stats("train.clean.json", stats_clean)

    total_train = stats_train["__total__"]["total"]
    total_clean = stats_clean["__total__"]["total"]
    diff = total_train - total_clean

    print("\n=== So sánh tổng số mẫu ===")
    print(f"train.json      : {total_train}")
    print(f"train.clean.json: {total_clean}")
    print(f"Chênh lệch      : {diff} mẫu (dương = train.json nhiều hơn)")


if __name__ == "__main__":
    main()


