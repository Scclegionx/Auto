

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_HISTORY_PATH = Path("models/phobert_multitask/training_history.json")
DEFAULT_OUTPUT_DIR = Path("reports")


def _load_history(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy training_history tại: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"training_history.json phải là list, nhận được: {type(data)}")
    return data


def _plot_losses(history: List[Dict], out_dir: Path) -> None:
    epochs = [item.get("epoch", i + 1) for i, item in enumerate(history)]
    train_loss = [item.get("train_loss") for item in history]
    val_loss = [item.get("val_loss") for item in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, val_loss, marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PhoBERT đa tác vụ – Loss theo epoch")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "multitask_losses.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_f1(history: List[Dict], out_dir: Path) -> None:
    epochs = [item.get("epoch", i + 1) for i, item in enumerate(history)]

    intent_f1 = [item.get("val_intent_macro_f1") for item in history]
    command_f1 = [item.get("val_command_macro_f1") for item in history]
    entity_f1 = [item.get("val_entity_f1") for item in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, intent_f1, marker="o", label="Intent macro F1 (val)")
    plt.plot(epochs, command_f1, marker="o", label="Command macro F1 (val)")
    plt.plot(epochs, entity_f1, marker="o", label="Entity F1 (val)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.05)
    plt.title("PhoBERT đa tác vụ – F1 theo epoch")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "multitask_f1.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_intent_command_zoom(history: List[Dict], out_dir: Path) -> None:
    """Biểu đồ zoom riêng cho Intent/Command để tránh bị dính sát trục 1.0."""
    epochs = [item.get("epoch", i + 1) for i, item in enumerate(history)]

    intent_f1 = [item.get("val_intent_macro_f1") for item in history]
    command_f1 = [item.get("val_command_macro_f1") for item in history]

    # Zoom quanh vùng giá trị cao (ví dụ 0.99–1.0)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, intent_f1, marker="o", label="Intent macro F1 (val)")
    plt.plot(epochs, command_f1, marker="o", label="Command macro F1 (val)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0.99, 1.001)
    plt.title("Intent / Command F1 (val) – zoom vùng 0.99–1.00")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "multitask_intent_command_f1_zoom.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Biểu đồ lỗi (1 - F1) để thấy rõ chênh lệch nhỏ
    intent_err = [1.0 - (v or 0.0) for v in intent_f1]
    command_err = [1.0 - (v or 0.0) for v in command_f1]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, intent_err, marker="o", label="1 - Intent macro F1 (val)")
    plt.plot(epochs, command_err, marker="o", label="1 - Command macro F1 (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Error (1 - F1)")
    plt.title("Intent / Command – Sai số so với F1 = 1.0")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "multitask_intent_command_error.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_entity_density(history: List[Dict], out_dir: Path) -> None:
    # Các key này được trainer ghi trong evaluate()
    pred_key = "val_entity_pred_non_o_ratio"
    true_key = "val_entity_true_non_o_ratio"

    if not any(pred_key in item for item in history):
        # Không có thông tin density trong history thì bỏ qua biểu đồ này.
        return

    epochs = [item.get("epoch", i + 1) for i, item in enumerate(history)]
    pred_ratio = [item.get(pred_key) for item in history]
    true_ratio = [item.get(true_key) for item in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, true_ratio, marker="o", label="True non-O ratio (val)")
    plt.plot(epochs, pred_ratio, marker="o", label="Pred non-O ratio (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Tỷ lệ token non-O")
    plt.title("PhoBERT đa tác vụ – Mật độ entity (val)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = out_dir / "multitask_entity_density.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _summarize_best(history: List[Dict], out_dir: Path) -> None:
    """
    Sinh file JSON tóm tắt:
    - Best epoch theo val_entity_f1.
    - Best epoch theo weighted score (giống trainer: 0.7 intent + 0.2 entity + 0.1 command).
    """
    if not history:
        return

    # Best theo entity F1
    best_entity = max(history, key=lambda h: h.get("val_entity_f1", float("-inf")))

    # Best theo weighted score tương tự trainer
    L_INTENT = 0.7
    L_ENTITY = 0.2
    L_COMMAND = 0.1

    def _score(h: Dict) -> float:
        intent = h.get("val_intent_macro_f1", 0.0)
        entity = h.get("val_entity_f1", 0.0)
        command = h.get("val_command_macro_f1", 0.0)
        return L_INTENT * intent + L_ENTITY * entity + L_COMMAND * command

    best_weighted = max(history, key=_score)

    summary = {
        "num_epochs": len(history),
        "best_by_entity_f1": {
            "epoch": best_entity.get("epoch"),
            "val_entity_f1": best_entity.get("val_entity_f1"),
            "val_entity_precision": best_entity.get("val_entity_precision"),
            "val_entity_recall": best_entity.get("val_entity_recall"),
            "val_intent_macro_f1": best_entity.get("val_intent_macro_f1"),
            "val_command_macro_f1": best_entity.get("val_command_macro_f1"),
        },
        "best_by_weighted_score": {
            "epoch": best_weighted.get("epoch"),
            "score": _score(best_weighted),
            "val_intent_macro_f1": best_weighted.get("val_intent_macro_f1"),
            "val_entity_f1": best_weighted.get("val_entity_f1"),
            "val_command_macro_f1": best_weighted.get("val_command_macro_f1"),
        },
    }

    out_path = out_dir / "multitask_training_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trực quan hoá lịch sử huấn luyện PhoBERT đa tác vụ từ training_history.json."
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(DEFAULT_HISTORY_PATH),
        help="Đường dẫn tới training_history.json (mặc định: models/phobert_multitask/training_history.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục lưu các biểu đồ và file tóm tắt (mặc định: reports).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)

    _plot_losses(history, output_dir)
    _plot_f1(history, output_dir)
    _plot_intent_command_zoom(history, output_dir)
    _plot_entity_density(history, output_dir)
    _summarize_best(history, output_dir)

    print(f"✅ Đã sinh biểu đồ training tại thư mục: {output_dir}")


if __name__ == "__main__":
    main()


