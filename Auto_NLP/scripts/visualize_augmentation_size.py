from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path("src/data")


def count_samples(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
    except Exception:
        return None
    return None


def main() -> None:
    master_path = BASE / "raw" / "elderly_commands_master.json"
    # Một trong các bản gốc trước refactor/augment nằm trong archive
    original_path = (
        BASE
        / "raw"
        / "archive"
        / "elderly_command_dataset_MERGED_13C_VITEXT_no_value.json"
    )

    train_path = BASE / "processed" / "train.json"
    val_path = BASE / "processed" / "val.json"
    test_path = BASE / "processed" / "test.json"

    n_master = count_samples(master_path)
    n_original = count_samples(original_path)
    n_train = count_samples(train_path)
    n_val = count_samples(val_path)
    n_test = count_samples(test_path)

    if n_master is None:
        raise SystemExit(f"Khong doc duoc master dataset tai: {master_path}")
    if n_original is None:
        raise SystemExit(f"Khong doc duoc original dataset tai: {original_path}")

    factor = n_master / n_original if n_original else 0.0

    # Tính tỉ lệ % train/val/test trên tập sau tăng cường
    if None not in (n_train, n_val, n_test) and n_master:
        p_train = n_train / n_master * 100
        p_val = n_val / n_master * 100
        p_test = n_test / n_master * 100
    else:
        p_train = p_val = p_test = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1]})

    # (1) Kích thước trước / sau tăng cường
    labels = ["Truoc tang cuong", "Sau tang cuong"]
    values = [n_original, n_master]

    ax0 = axes[0]
    bars0 = ax0.bar(range(len(labels)), values, color=["#4C72B0", "#C44E52"])
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("So mau")
    ax0.set_title("Kich thuoc du lieu truoc / sau tang cuong")
    ax0.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars0, values):
        height = bar.get_height()
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(100, height * 0.01),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax0.text(
        0.5,
        max(values) * 1.02,
        f"~{factor:.1f}x",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # (2) Tỉ lệ train / val / test trên tập sau tăng cường
    ax1 = axes[1]
    split_labels = ["train", "val", "test"]
    split_values = [p_train, p_val, p_test]
    split_counts = [n_train or 0, n_val or 0, n_test or 0]

    bars1 = ax1.bar(range(len(split_labels)), split_values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax1.set_xticks(range(len(split_labels)))
    ax1.set_xticklabels(split_labels)
    ax1.set_ylabel("% mau trong tap sau tang cuong")
    ax1.set_ylim(0, max(split_values) * 1.2 if any(split_values) else 100)
    ax1.set_title("Phan bo train / val / test tren tap sau tang cuong")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count in zip(bars1, split_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.5, height * 0.02),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_augmentation_size.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Da luu bieu do tai: {out_path}")


if __name__ == "__main__":
    main()


