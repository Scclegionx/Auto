from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Thêm project root vào sys.path để import được src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.configs.config import IntentConfig, ModelConfig
from core.model_loader import TrainedModelInference


def _load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} phải là list, nhận được: {type(data)}")
    return data


def _collect_intent_predictions(
    dataset: List[Dict], model: TrainedModelInference
) -> Tuple[List[str], List[str]]:
    true_labels: List[str] = []
    pred_labels: List[str] = []

    for item in dataset:
        text = item.get("input", "")
        # Trong processed data, intent/command đều trùng bộ 10 nhãn
        true_intent = item.get("command") or item.get("intent") or "unknown"
        if not text:
            continue

        result = model.predict(text)
        pred_intent = result.get("intent", "unknown")

        true_labels.append(str(true_intent))
        pred_labels.append(str(pred_intent))

    return true_labels, pred_labels


def _compute_per_intent_accuracy(
    y_true: List[str], y_pred: List[str], intents: List[str]
) -> Dict[str, float]:
    """
    Accuracy theo intent = số mẫu đúng của intent đó / tổng số mẫu intent đó trong ground truth.
    Đây chính là recall theo từng lớp, nhưng dễ giải thích với hội đồng hơn dưới tên "per-intent accuracy".
    """
    cm = confusion_matrix(y_true, y_pred, labels=intents)
    per_intent_acc: Dict[str, float] = {}
    for idx, label in enumerate(intents):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        acc = float(correct) / float(total) if total > 0 else 0.0
        per_intent_acc[label] = acc
    return per_intent_acc


def _plot_intent_accuracy(
    per_intent_acc: Dict[str, float], support: Dict[str, int], out_dir: Path
) -> None:
    intents = list(per_intent_acc.keys())
    acc_values = [per_intent_acc[i] for i in intents]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(intents, acc_values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy theo intent")
    plt.xlabel("Intent")
    plt.title("Per-intent accuracy trên tập đánh giá")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    out_path = out_dir / "intent_per_class_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_intent_table(
    per_intent_acc: Dict[str, float], support: Dict[str, int], out_path: Path
) -> None:
    lines = ["| Intent | Accuracy | Support |", "|--------|----------|---------|"]
    for intent, acc in per_intent_acc.items():
        lines.append(f"| {intent} | {acc:.4f} | {support.get(intent, 0)} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    model = TrainedModelInference("models/phobert_multitask/best_model.pt")

    cfg = ModelConfig()
    test_path = Path(cfg.test_data_path)
    dataset = _load_dataset(test_path)

    print(f"Đang tính per-intent accuracy trên: {test_path} (số mẫu = {len(dataset)})")
    y_true, y_pred = _collect_intent_predictions(dataset, model)

    intents = IntentConfig().intent_labels
    per_intent_acc = _compute_per_intent_accuracy(y_true, y_pred, intents)

    support_counter: Counter[str] = Counter(y_true)  # type: ignore[type-arg]

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_intent_accuracy(per_intent_acc, support_counter, out_dir)
    _save_intent_table(per_intent_acc, support_counter, out_dir / "intent_per_class_accuracy.md")

    print("Per-intent accuracy:")
    for intent in intents:
        print(f"- {intent}: acc={per_intent_acc[intent]:.4f}, n={support_counter.get(intent, 0)}")
    print(f"✅ Đã lưu biểu đồ và bảng tại thư mục: {out_dir}")


if __name__ == "__main__":
    main()


