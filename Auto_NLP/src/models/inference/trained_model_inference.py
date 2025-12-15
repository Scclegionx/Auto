#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple helper để chạy thử mô hình đa tác vụ sau huấn luyện.
"""

from __future__ import annotations

from pathlib import Path

from models.inference.model_loader import load_multi_task_model


def demo() -> None:
    model = load_multi_task_model("phobert_large_intent_model")

    samples = [
        "gọi điện cho mẹ",
        "bật đèn phòng khách",
        "tìm kiếm nhạc trên youtube",
        "đặt báo thức 7 giờ sáng",
        "gửi tin nhắn cho bạn",
    ]

    print("🧪 Demo inference với mô hình đa tác vụ:")
    for idx, text in enumerate(samples, start=1):
        result = model.predict(text)
        print(f"\n{idx}. \"{text}\"")
        print(f"   Intent  : {result['intent']} ({result['intent_confidence']:.2%})")
        print(f"   Command : {result['command']} ({result['command_confidence']:.2%})")
        print(f"   Entities: {result['entities']}")


if __name__ == "__main__":
    if not Path("models/trained/phobert_large_intent_model").exists():
        print("⚠️  Chưa có checkpoint trong models/trained/phobert_large_intent_model")
    else:
        demo()
