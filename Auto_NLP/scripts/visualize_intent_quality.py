from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


DEFAULT_HISTORY_PATH = Path("models/phobert_multitask/training_history.json")
DEFAULT_BENCHMARK_PATH = Path("reports/benchmark_metrics.json")
DEFAULT_OUTPUT_DIR = Path("reports")


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy training_history tại: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"training_history.json phải là list, nhận được: {type(data)}")
    return data


def _load_benchmarks(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy benchmark_metrics tại: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"benchmark_metrics.json phải là list, nhận được: {type(data)}")
    return data


def _plot_intent_over_epochs(history: List[Dict[str, Any]], out_dir: Path) -> None:
    if not history:
        return

    epochs: List[int] = [int(item.get("epoch", i + 1)) for i, item in enumerate(history)]
    # Ép kiểu về float, thay None bằng 0.0 để làm hài lòng type checker và matplotlib
    macro_f1: List[float] = [
        float(item.get("val_intent_macro_f1", 0.0) or 0.0) for item in history
    ]
    weighted_f1: List[float] = [
        float(item.get("val_intent_weighted_f1", 0.0) or 0.0) for item in history
    ]
    accuracy: List[float] = [
        float(item.get("val_intent_accuracy", 0.0) or 0.0) for item in history
    ]

    # Gom tất cả giá trị để tự động chọn thang Y hợp lý
    all_vals: List[float] = macro_f1 + weighted_f1 + accuracy
    if not all_vals:
        return

    min_val = min(all_vals)
    max_val = max(all_vals)

    # Nếu model gần như đã saturate (mọi giá trị ~1.0) thì zoom trục Y lên vùng [min-ε, 1.0]
    if max_val - min_val < 0.05:
        y_min = max(0.0, min_val - 0.01)
        y_max = 1.01
    else:
        y_min = 0.0
        y_max = 1.05

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, macro_f1, marker="o", label="Intent macro F1 (val)")
    plt.plot(epochs, weighted_f1, marker="o", label="Intent weighted F1 (val)")
    plt.plot(epochs, accuracy, marker="o", label="Intent accuracy (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Giá trị metric")
    plt.ylim(y_min, y_max)
    plt.title("Diễn tiến chất lượng Intent theo epoch (validation)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "intent_metrics_over_epochs.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_intent_error_over_epochs(history: List[Dict[str, Any]], out_dir: Path) -> None:
    """Vẽ tỉ lệ lỗi (1 - metric) cho intent để phóng to vùng sai số rất nhỏ."""
    if not history:
        return

    epochs: List[int] = [int(item.get("epoch", i + 1)) for i, item in enumerate(history)]
    acc: List[float] = [
        float(item.get("val_intent_accuracy", 0.0) or 0.0) for item in history
    ]
    macro_f1: List[float] = [
        float(item.get("val_intent_macro_f1", 0.0) or 0.0) for item in history
    ]
    weighted_f1: List[float] = [
        float(item.get("val_intent_weighted_f1", 0.0) or 0.0) for item in history
    ]

    acc_err: List[float] = [max(0.0, 1.0 - v) for v in acc]
    macro_f1_err: List[float] = [max(0.0, 1.0 - v) for v in macro_f1]
    weighted_f1_err: List[float] = [max(0.0, 1.0 - v) for v in weighted_f1]

    all_err: List[float] = acc_err + macro_f1_err + weighted_f1_err
    if not all_err:
        return

    max_err = max(all_err)
    # Phóng to trục Y quanh vùng lỗi nhỏ (ví dụ tới 1.2 * lỗi lớn nhất)
    y_max = max(0.001, max_err * 1.2)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, macro_f1_err, marker="o", label="1 - Intent macro F1 (val)")
    plt.plot(epochs, weighted_f1_err, marker="o", label="1 - Intent weighted F1 (val)")
    plt.plot(epochs, acc_err, marker="o", label="1 - Intent accuracy (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Tỉ lệ lỗi")
    plt.ylim(0.0, y_max)
    plt.title("Tỉ lệ lỗi Intent theo epoch (validation)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "intent_error_over_epochs.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_intent_by_model(benchmarks: List[Dict[str, Any]], out_dir: Path) -> None:
    if not benchmarks:
        return

    models: List[str] = [str(item.get("model", "unknown")) for item in benchmarks]
    intent_f1: List[float] = [
        float(item.get("intent_weighted_f1", 0.0) or 0.0) for item in benchmarks
    ]

    plt.figure(figsize=(7, 4))
    positions = range(len(models))
    bars = plt.bar(positions, intent_f1, color="#4C72B0")
    plt.xticks(positions, models, rotation=20, ha="right")
    plt.ylabel("Intent weighted F1 (val)")
    plt.ylim(0.0, 1.05)
    plt.title("So sánh chất lượng Intent giữa các mô hình")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Ghi giá trị lên đầu cột
    for bar, value in zip(bars, intent_f1):
        if value is None:
            continue
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out_path = out_dir / "intent_f1_by_model.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trực quan hoá chất lượng tác vụ nhận diện ý định (Intent)."
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(DEFAULT_HISTORY_PATH),
        help="Đường dẫn tới training_history.json (mặc định: models/phobert_multitask/training_history.json).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Đường dẫn tới benchmark_metrics.json (mặc định: reports/benchmark_metrics.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục lưu các biểu đồ (mặc định: reports).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_path)
    benchmark_path = Path(args.benchmark_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)
    benchmarks = _load_benchmarks(benchmark_path)

    _plot_intent_over_epochs(history, output_dir)
    _plot_intent_error_over_epochs(history, output_dir)
    _plot_intent_by_model(benchmarks, output_dir)

    # Tránh lỗi UnicodeEncodeError trên một số terminal Windows
    print(f"Da sinh bieu do intent tai thu muc: {output_dir}")


if __name__ == "__main__":
    main()


