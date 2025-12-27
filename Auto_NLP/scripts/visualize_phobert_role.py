from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _add_box(ax, xy, width, height, text, facecolor="#FFFFFF", edgecolor="#333333"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )


def _add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.0,
            color="#333333",
        )
    )


def _draw_architecture(out_path: Path) -> None:
    """Hình 2.x.a – Vai trò của PhoBERT trong kiến trúc mô hình đa nhiệm."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.5,
        "PhoBERT trong kiến trúc mô hình đa nhiệm",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    box_w, box_h = 2.0, 1.0

    # Input tokens
    _add_box(
        ax,
        (0.5, 3.0),
        box_w,
        box_h,
        "Câu lệnh\n(tokenized\nbằng PhoBERT)",
        facecolor="#E5F2FF",
    )

    # PhoBERT encoder (trung tâm)
    _add_box(
        ax,
        (3.5, 3.0),
        box_w,
        box_h,
        "PhoBERT\nencoder\n(shared)",
        facecolor="#FFE5EC",
        edgecolor="#C2185B",
    )

    # Heads (intent / command / entity)
    _add_box(
        ax,
        (7.0, 4.0),
        box_w,
        box_h * 0.7,
        "Head\nIntent",
        facecolor="#E8F5E9",
    )
    _add_box(
        ax,
        (7.0, 3.0),
        box_w,
        box_h * 0.7,
        "Head\nCommand",
        facecolor="#E8F5E9",
    )
    _add_box(
        ax,
        (7.0, 2.0),
        box_w,
        box_h * 0.7,
        "Head\nEntity\n(token-level)",
        facecolor="#E8F5E9",
    )

    # Arrows
    _add_arrow(ax, (0.5 + box_w, 3.5), (3.5, 3.5))
    _add_arrow(ax, (3.5 + box_w, 4.1), (7.0, 4.3))
    _add_arrow(ax, (3.5 + box_w, 3.5), (7.0, 3.35))
    _add_arrow(ax, (3.5 + box_w, 2.9), (7.0, 2.4))

    ax.text(
        5,
        1.1,
        "PhoBERT tạo biểu diễn câu (cho Intent/Command)\n"
        "và biểu diễn theo token (cho Entity/NER).",
        ha="center",
        va="center",
        fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _draw_system_pipeline(out_path: Path) -> None:
    """Hình 2.x.b – Vị trí tầng suy luận kết hợp trong pipeline SAM."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.5,
        "PhoBERT trong pipeline suy luận & hybrid của SAM",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    box_w, box_h = 2.2, 1.0

    # User / ASR
    _add_box(
        ax,
        (0.5, 3.0),
        box_w,
        box_h,
        "Người dùng /\nASR đầu ra\n(câu tiếng Việt)",
        facecolor="#E5F2FF",
    )

    # PhoBERT-based multi-task model
    _add_box(
        ax,
        (3.4, 3.0),
        box_w,
        box_h,
        "PhoBERT\nmulti-task\n(intent/command/entity)",
        facecolor="#FFE5EC",
        edgecolor="#C2185B",
    )

    # Hybrid reasoning
    _add_box(
        ax,
        (6.3, 3.6),
        box_w,
        box_h * 0.7,
        "Hybrid\nreasoning\n(FAISS + fuzzy)",
        facecolor="#E8F5E9",
    )

    # API / FE
    _add_box(
        ax,
        (6.3, 2.1),
        box_w,
        box_h * 0.7,
        "API FastAPI\n/predict → FE\n(intent, command, entities)",
        facecolor="#E8F5E9",
    )

    # Arrows
    _add_arrow(ax, (0.5 + box_w, 3.5), (3.4, 3.5))
    _add_arrow(ax, (3.4 + box_w, 3.8), (6.3, 3.95))
    _add_arrow(ax, (3.4 + box_w, 3.2), (6.3, 2.45))

    ax.text(
        5,
        1.0,
        "Embeddings PhoBERT được dùng đồng thời cho mô hình đa nhiệm\n"
        "và làm nền cho tầng suy luận kết hợp.",
        ha="center",
        va="center",
        fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    reports_dir = Path("reports")
    _draw_architecture(reports_dir / "phobert_role_architecture.png")
    _draw_system_pipeline(reports_dir / "phobert_role_system_pipeline.png")
    print("Da luu cac so do PhoBERT tai:", reports_dir)


if __name__ == "__main__":
    main()


