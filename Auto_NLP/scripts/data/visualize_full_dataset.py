#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trực quan hóa đầy đủ master dataset cho báo cáo.
Usage:
    python scripts/data/visualize_full_dataset.py --input src/data/raw/master_dataset_35609.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset từ file JSON."""
    print(f"Đang đọc: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset phải là list, nhận được: {type(data)}")
    return data


def plot_intent_distribution_bar(samples: List[Dict], output_dir: Path) -> None:
    """Bar chart phân bố intent."""
    intent_counter = Counter()
    for sample in samples:
        intent = sample.get("intent", sample.get("command", "unknown"))
        intent_counter[intent] += 1
    
    intents = [intent for intent, _ in intent_counter.most_common()]
    counts = [count for _, count in intent_counter.most_common()]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(intents)), counts, color='#4C72B0')
    plt.xticks(range(len(intents)), intents, rotation=45, ha='right')
    plt.ylabel('Số mẫu')
    plt.title('Phân bố Intent trong dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_intent_bar.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_intent_bar.png")


def plot_intent_distribution_pie(samples: List[Dict], output_dir: Path) -> None:
    """Pie chart phân bố intent."""
    intent_counter = Counter()
    for sample in samples:
        intent = sample.get("intent", sample.get("command", "unknown"))
        intent_counter[intent] += 1
    
    labels = [intent for intent, _ in intent_counter.most_common()]
    sizes = [count for _, count in intent_counter.most_common()]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Tỷ lệ % Intent trong dataset')
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_intent_pie.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_intent_pie.png")


def plot_entity_distribution(samples: List[Dict], output_dir: Path, top_n: int = 16) -> None:
    """Bar chart top entity labels."""
    entity_counter = Counter()
    for sample in samples:
        for ent in sample.get("entities", []):
            if isinstance(ent, dict):
                label = ent.get("label", "UNKNOWN")
                entity_counter[label] += 1
    
    top_entities = entity_counter.most_common(top_n)
    labels = [label for label, _ in top_entities]
    counts = [count for _, count in top_entities]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), counts, color='#55A868')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Số lượng entity')
    plt.title(f'Top {top_n} Entity Labels trong dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_entity_distribution.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_entity_distribution.png")


def plot_sentence_length_histogram(samples: List[Dict], output_dir: Path) -> None:
    """Histogram độ dài câu (số từ)."""
    lengths = []
    for sample in samples:
        text = sample.get("input", "")
        lengths.append(len(text.split()))
    
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='#C44E52')
    plt.xlabel('Độ dài câu (số từ)')
    plt.ylabel('Số mẫu')
    plt.title('Phân bố độ dài câu trong dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Thêm thống kê
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_sentence_length.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_sentence_length.png")


def plot_entities_per_sample(samples: List[Dict], output_dir: Path) -> None:
    """Histogram số entity/mẫu."""
    entity_counts = []
    for sample in samples:
        entities = sample.get("entities", [])
        entity_counts.append(len(entities))
    
    plt.figure(figsize=(8, 5))
    bins = range(0, max(entity_counts) + 2)
    plt.hist(entity_counts, bins=bins, edgecolor='black', alpha=0.7, color='#8172B3')
    plt.xlabel('Số entity trong một mẫu')
    plt.ylabel('Số mẫu')
    plt.title('Phân bố số entity/mẫu')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    mean_ent = np.mean(entity_counts)
    plt.axvline(mean_ent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ent:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_entities_per_sample.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_entities_per_sample.png")


def plot_entity_by_intent_heatmap(samples: List[Dict], output_dir: Path) -> None:
    """Heatmap entity xuất hiện trong từng intent."""
    # Collect entity-intent co-occurrence
    entity_intent_matrix: Dict[str, Dict[str, int]] = {}
    
    for sample in samples:
        intent = sample.get("intent", sample.get("command", "unknown"))
        for ent in sample.get("entities", []):
            if isinstance(ent, dict):
                label = ent.get("label", "UNKNOWN")
                if label not in entity_intent_matrix:
                    entity_intent_matrix[label] = {}
                entity_intent_matrix[label][intent] = entity_intent_matrix[label].get(intent, 0) + 1
    
    # Lấy top entity labels
    entity_totals = {label: sum(counts.values()) for label, counts in entity_intent_matrix.items()}
    top_entities = sorted(entity_totals.items(), key=lambda x: x[1], reverse=True)[:16]
    top_entity_labels = [label for label, _ in top_entities]
    
    # Lấy tất cả intent
    all_intents = sorted(set(sample.get("intent", sample.get("command", "unknown")) for sample in samples))
    # Loại unknown nếu có
    if "unknown" in all_intents:
        all_intents.remove("unknown")
    
    # Tạo ma trận
    matrix = []
    for entity_label in top_entity_labels:
        row = []
        for intent in all_intents:
            count = entity_intent_matrix.get(entity_label, {}).get(intent, 0)
            row.append(count)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=all_intents,
        yticklabels=top_entity_labels,
        annot=False,  # Không ghi số để hình sạch
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Số lần xuất hiện'}
    )
    plt.xlabel('Intent')
    plt.ylabel('Entity Label')
    plt.title('Heatmap: Entity xuất hiện theo Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_entity_by_intent_heatmap.png", dpi=200)
    plt.close()
    print("✅ Đã tạo: dataset_entity_by_intent_heatmap.png")


def save_summary_table(samples: List[Dict], output_dir: Path) -> None:
    """Lưu bảng tóm tắt dataset."""
    intent_counter = Counter()
    entity_counter = Counter()
    total_entities = 0
    
    for sample in samples:
        intent = sample.get("intent", sample.get("command", "unknown"))
        intent_counter[intent] += 1
        
        entities = sample.get("entities", [])
        total_entities += len(entities)
        for ent in entities:
            if isinstance(ent, dict):
                label = ent.get("label", "UNKNOWN")
                entity_counter[label] += 1
    
    lengths = [len(sample.get("input", "").split()) for sample in samples]
    
    summary = {
        "tong_so_mau": len(samples),
        "so_intent": len(intent_counter),
        "so_entity_labels": len(entity_counter),
        "trung_binh_entity_per_mau": total_entities / len(samples) if samples else 0,
        "do_dai_trung_binh": np.mean(lengths) if lengths else 0,
        "do_dai_min": int(np.min(lengths)) if lengths else 0,
        "do_dai_max": int(np.max(lengths)) if lengths else 0,
        "phan_bo_intent": dict(intent_counter.most_common()),
        "top_10_entity": dict(entity_counter.most_common(10)),
    }
    
    # Lưu JSON
    json_path = output_dir / "dataset_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã tạo: dataset_summary.json")
    
    # Tạo markdown table
    md_path = output_dir / "dataset_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Tóm tắt Dataset\n\n")
        f.write("## Thông tin tổng quan\n\n")
        f.write(f"- **Tổng số mẫu**: {summary['tong_so_mau']:,}\n")
        f.write(f"- **Số Intent**: {summary['so_intent']}\n")
        f.write(f"- **Số Entity labels**: {summary['so_entity_labels']}\n")
        f.write(f"- **Trung bình entity/mẫu**: {summary['trung_binh_entity_per_mau']:.2f}\n")
        f.write(f"- **Độ dài câu trung bình**: {summary['do_dai_trung_binh']:.1f} từ\n")
        f.write(f"- **Độ dài min/max**: {summary['do_dai_min']} / {summary['do_dai_max']} từ\n\n")
        
        f.write("## Phân bố Intent\n\n")
        f.write("| Intent | Số mẫu | Tỷ lệ % |\n")
        f.write("|--------|-------:|--------:|\n")
        for intent, count in summary['phan_bo_intent'].items():
            pct = count / summary['tong_so_mau'] * 100
            f.write(f"| {intent} | {count:,} | {pct:.2f}% |\n")
        
        f.write("\n## Top 10 Entity\n\n")
        f.write("| Entity Label | Số lần xuất hiện |\n")
        f.write("|--------------|------------------:|\n")
        for label, count in summary['top_10_entity'].items():
            f.write(f"| {label} | {count:,} |\n")
    
    print(f"✅ Đã tạo: dataset_summary.md")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa đầy đủ master dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/data/raw/master_dataset_35609.json"),
        help="Đường dẫn master dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Thư mục lưu biểu đồ.",
    )
    args = parser.parse_args()
    
    samples = load_dataset(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Tổng số mẫu: {len(samples):,}")
    print(f"{'='*60}\n")
    
    # Sinh các biểu đồ
    print("Đang tạo biểu đồ...")
    plot_intent_distribution_bar(samples, output_dir)
    plot_intent_distribution_pie(samples, output_dir)
    plot_entity_distribution(samples, output_dir)
    plot_sentence_length_histogram(samples, output_dir)
    plot_entities_per_sample(samples, output_dir)
    plot_entity_by_intent_heatmap(samples, output_dir)
    save_summary_table(samples, output_dir)
    
    print(f"\n{'='*60}")
    print("✅ Hoàn tất tất cả biểu đồ!")
    print(f"Xem trong thư mục: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()





