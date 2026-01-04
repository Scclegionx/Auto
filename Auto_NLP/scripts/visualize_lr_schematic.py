#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vẽ biểu đồ MINH HOẠ (schematic) cho ảnh hưởng của các mức learning rate khác nhau
(ví dụ 1e-5, 2e-5, 3e-5) lên tốc độ hội tụ Entity F1.

Lưu ý:
- Đây KHÔNG phải dữ liệu đo đạc trực tiếp, chỉ là đường cong mô phỏng
  để giải thích trực giác trong báo cáo.

Chạy:
    python scripts/visualize_lr_schematic.py

Ảnh sẽ được lưu tại: reports/lr_schematic.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    epochs = np.arange(1, 7)  # 6 epoch cho minh hoạ rõ hơn

    # Các giá trị F1 MINH HOẠ (không phải log thật)
    # lr = 1e-5: tăng chậm và đều
    f1_1e5 = [0.40, 0.44, 0.47, 0.50, 0.52, 0.54]
    # lr = 2e-5: tăng nhanh hơn và ổn định
    f1_2e5 = [0.45, 0.53, 0.58, 0.61, 0.63, 0.65]
    # lr = 3e-5: dao động rõ ở đầu, sau đó mới ổn định hơn
    f1_3e5 = [0.50, 0.46, 0.55, 0.57, 0.58, 0.60]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, f1_1e5, marker="o", label="lr = 1e-5")
    plt.plot(epochs, f1_2e5, marker="o", label="lr = 2e-5")
    plt.plot(epochs, f1_3e5, marker="o", label="lr = 3e-5")

    plt.xlabel("Epoch")
    plt.ylabel("Entity F1")
    plt.ylim(0.3, 0.7)
    plt.title("Minh hoạ định tính ảnh hưởng của learning rate")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lr_schematic.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Đã sinh biểu đồ minh hoạ tại: {out_path}")


if __name__ == "__main__":
    main()


