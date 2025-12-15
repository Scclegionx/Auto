#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_model_comparison.py

Script trực quan hoá so sánh các backbone (PhoBERT-base/large, XLM-R, viT5)
theo các chỉ số đã có sẵn:
- Chất lượng: intent_weighted_f1, ner_micro_f1.
- Hiệu năng: latency_ms, vram_gb, avg_epoch_time_sec.

Nguồn dữ liệu:
- reports/benchmark_metrics.json
- reports/training_dynamics.json

Chạy ví dụ:
    python scripts/visualize_model_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


BENCHMARK_PATH = Path("reports/benchmark_metrics.json")
TRAIN_DYNAMICS_PATH = Path("reports/training_dynamics.json")
DEFAULT_OUTPUT_DIR = Path("reports")


def _load_json_list(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"File {path} phải chứa một list các object JSON.")
    return data


def _merge_metrics() -> List[Dict]:
    bench = _load_json_list(BENCHMARK_PATH)
    dyn = _load_json_list(TRAIN_DYNAMICS_PATH)

    by_model: Dict[str, Dict] = {}
    for item in bench:
        model = item.get("model")
        if not model:
            continue
        by_model.setdefault(model, {}).update(item)

    for item in dyn:
        model = item.get("model")
        if not model:
            continue
        by_model.setdefault(model, {}).update(item)

    merged = []
    for model, metrics in by_model.items():
        m = {"model": model}
        m.update(metrics)
        merged.append(m)
    return merged


def _plot_quality_and_latency(models: List[Dict], out_dir: Path) -> None:
    names = [m["model"] for m in models]
    intent_f1 = [m.get("intent_weighted_f1") for m in models]
    ner_f1 = [m.get("ner_micro_f1") for m in models]
    latency = [m.get("latency_ms") for m in models]

    x = range(len(models))

    plt.figure(figsize=(8, 5))

    # Subplot 1: F1
    ax1 = plt.subplot(2, 1, 1)
    width = 0.35
    ax1.bar([i - width / 2 for i in x], intent_f1, width=width, label="Intent weighted F1")
    ax1.bar([i + width / 2 for i in x], ner_f1, width=width, label="NER micro F1")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_title("So sánh chất lượng giữa các backbone")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # Subplot 2: Latency
    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(x, latency, color="#ffb74d")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_title("Độ trễ suy luận trung bình")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "model_quality_vs_latency.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_resources_and_speed(models: List[Dict], out_dir: Path) -> None:
    names = [m["model"] for m in models]
    vram = [m.get("vram_gb") for m in models]
    epoch_time = [m.get("avg_epoch_time_sec") for m in models]

    x = range(len(models))

    plt.figure(figsize=(8, 5))

    # Subplot 1: VRAM
    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(x, vram, color="#64b5f6")
    ax1.set_ylabel("VRAM (GB)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_title("Bộ nhớ GPU cần thiết")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Subplot 2: Avg epoch time
    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(x, epoch_time, color="#81c784")
    ax2.set_ylabel("Thời gian/epoch (giây)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_title("Tốc độ huấn luyện trung bình")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "model_resources_and_speed.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trực quan hoá so sánh backbone (PhoBERT, XLM-R, viT5) về chất lượng và tài nguyên."
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_metrics()
    if not merged:
        raise RuntimeError("Không tìm thấy bất kỳ model nào trong benchmark_metrics/training_dynamics.")

    _plot_quality_and_latency(merged, output_dir)
    _plot_resources_and_speed(merged, output_dir)

    print(f"✅ Đã sinh biểu đồ so sánh backbone tại thư mục: {output_dir}")


if __name__ == "__main__":
    main()






