from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STATS_PATH = Path("reports/entity_group_f1.json")
DEFAULT_OUTPUT_DIR = Path("reports")


def _load_entity_stats(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy entity_group_f1.json tại: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    groups = data.get("all_groups") or data.get("selected_groups") or []
    if not isinstance(groups, list):
        raise ValueError(f"Trường all_groups/selected_groups phải là list, nhận được: {type(groups)}")
    return groups


def _plot_f1_by_group(groups: List[Dict], out_dir: Path) -> None:
    names = [g.get("entity_group", "UNK") for g in groups]
    f1_scores = [g.get("f1", 0.0) for g in groups]

    plt.figure(figsize=(8, 4))
    plt.bar(names, f1_scores)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title("F1 theo nhóm entity (val/test)")
    plt.tight_layout()
    out_path = out_dir / "entity_group_f1_all.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_f1_sorted(groups: List[Dict], out_dir: Path) -> None:
    sorted_groups = sorted(groups, key=lambda g: g.get("f1", 0.0), reverse=True)
    names = [g.get("entity_group", "UNK") for g in sorted_groups]
    f1_scores = [g.get("f1", 0.0) for g in sorted_groups]

    plt.figure(figsize=(8, 4))
    plt.bar(names, f1_scores)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title("F1 theo nhóm entity (sắp xếp giảm dần)")
    plt.tight_layout()
    out_path = out_dir / "entity_group_f1_sorted.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_metric_by_group(
    groups: List[Dict], out_dir: Path, metric_key: str, title: str, filename: str
) -> None:
    """Vẽ một metric (Precision / Recall / F1) cho từng nhóm entity thành biểu đồ riêng."""
    names = [g.get("entity_group", "UNK") for g in groups]
    values = [g.get(metric_key, 0.0) for g in groups]

    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_key.capitalize())
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    stats_path = DEFAULT_STATS_PATH
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = _load_entity_stats(stats_path)
    _plot_f1_by_group(groups, out_dir)
    _plot_f1_sorted(groups, out_dir)
    # Biểu đồ riêng cho từng metric
    _plot_metric_by_group(
        groups,
        out_dir,
        metric_key="precision",
        title="Precision theo nhóm entity",
        filename="entity_group_precision_all.png",
    )
    _plot_metric_by_group(
        groups,
        out_dir,
        metric_key="recall",
        title="Recall theo nhóm entity",
        filename="entity_group_recall_all.png",
    )
    # F1 theo nhóm đã có ở entity_group_f1_all.png; nếu cần bản nhãn rõ hơn có thể dùng thêm:
    _plot_metric_by_group(
        groups,
        out_dir,
        metric_key="f1",
        title="F1 theo nhóm entity (bản đơn)",
        filename="entity_group_f1_single.png",
    )
    print(f"✅ Đã sinh biểu đồ entity group tại thư mục: {out_dir}")


if __name__ == "__main__":
    main()


