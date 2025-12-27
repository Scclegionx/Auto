#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_hybrid_fe_contract.py

Script này dùng để:
- Chạy hybrid model (TrainedModelInference) trên một số câu test đại diện.
- In ra JSON cuối cùng (intent, command, entities) cho từng câu.
- Kiểm tra sơ bộ xem các entity "bắt buộc" cho từng command đã xuất hiện hay chưa.

Mục tiêu: giúp quan sát trực quan mức độ phù hợp giữa output hiện tại và
"hợp đồng entity" mà FE mong đợi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from core.model_loader import load_trained_model
from src.inference.engines.entity_extractor import EntityExtractor


@dataclass
class Contract:
    # Mỗi phần tử trong required_any là 1 nhóm "ít nhất 1 field phải có"
    required_any: List[List[str]]


FE_CONTRACT: Dict[str, Contract] = {
    "send-mess": Contract(
        required_any=[
            ["MESSAGE"],
            ["RECEIVER", "CONTACT_NAME", "PHONE"],
        ]
    ),
    "set-alarm": Contract(
        required_any=[
            ["TIME", "TIMESTAMP"],
        ]
    ),
    "control-device": Contract(
        required_any=[
            ["DEVICE"],
            ["ACTION", "MODE"],
        ]
    ),
    "call": Contract(
        required_any=[
            ["CONTACT_NAME", "RECEIVER", "PHONE"],
        ]
    ),
    "make-video-call": Contract(
        required_any=[
            ["PLATFORM"],
            ["CONTACT_NAME", "RECEIVER", "PHONE"],
        ]
    ),
    "open-cam": Contract(
        required_any=[
            ["ACTION"],
            ["MODE"],
            ["CAMERA_TYPE"],
        ]
    ),
    "search-internet": Contract(
        required_any=[
            ["QUERY"],
            ["PLATFORM"],
        ]
    ),
    "search-youtube": Contract(
        required_any=[
            ["QUERY"],
            ["PLATFORM"],
        ]
    ),
}


TEST_SENTENCES: Sequence[str] = [
    # send-mess
    "gửi tin nhắn cho cháu nội là hôm nay ông bận không sang chơi được",
    "nhắn tin cho con gái bảo tối nay bố không ăn cơm ở nhà",
    "nhắn qua Zalo cho cậu Sáu: mai 7 giờ đi khám",
    # set-alarm
    "Đặt báo thức lúc 6 giờ 30 sáng nhắc uống thuốc",
    "đặt báo thức 6 rưỡi sáng mai nhắc uống thuốc",
    "nhắc ông 9 giờ tối đo đường huyết",
    # control-device
    "bật điều hòa 26 độ ở phòng ngủ",
    "giảm âm lượng xuống 30 phần trăm",
    # call / make-video-call
    "gọi điện cho con trai bằng zalo",
    "gọi video cho cô Hạnh bằng Messenger",
    # open-cam
    "mở camera trước để chụp selfie",
    "mở camera sau xem ngoài cổng có ai không",
    # search
    "mở nhạc trữ tình cho ông nghe",
    "tìm trên web cách đo huyết áp tại nhà",
    "Tìm trên YouTube: bài tập thở cho người cao tuổi",
]


def _check_contract(command: str, entities: Dict[str, object]) -> Dict[str, object]:
    """Trả về dict nhỏ mô tả nhóm nào đã đủ / thiếu."""
    c = FE_CONTRACT.get(command)
    if not c:
        return {"status": "no_contract"}

    missing_groups: List[List[str]] = []
    for group in c.required_any:
        if not any(k in entities and entities.get(k) not in (None, "") for k in group):
            missing_groups.append(group)

    if not missing_groups:
        return {"status": "ok"}
    return {
        "status": "missing",
        "missing_groups": missing_groups,
    }


def main() -> None:
    model = load_trained_model("phobert_multitask")
    extractor = EntityExtractor()

    results: List[Dict[str, object]] = []

    for text in TEST_SENTENCES:
        out = model.predict(text)
        command = str(out.get("command", out.get("intent", "unknown")) or "unknown")
        model_entities = out.get("entities", {}) or {}

        # Dùng specialized extractor để bù entity còn thiếu (receiver, platform, time, v.v.)
        try:
            specialized = extractor.extract_all_entities(text, command)
        except Exception:
            specialized = {}

        # Mô phỏng logic hybrid: model là nguồn chính, specialized dùng để tinh chỉnh/gọt lại
        entities: Dict[str, object] = dict(model_entities)
        entities.update(specialized)

        # Làm sạch nhẹ một số entity phổ biến để log/soi cho dễ (không thay đổi logic train/infer chính)
        if command == "send-mess":
            rec = entities.get("RECEIVER")
            if isinstance(rec, str):
                # Bỏ dấu câu thừa ở cuối receiver (vd: "Cậu Sáu:")
                entities["RECEIVER"] = rec.rstrip(" :;,.").strip()

        if command == "set-alarm":
            # set-alarm không dùng QUERY/YT_QUERY – bỏ bớt để log cho gọn, trùng với hybrid
            entities.pop("QUERY", None)
            entities.pop("YT_QUERY", None)

        # Bắt chước hybrid: search-youtube luôn có PLATFORM=youtube nếu chưa có
        if command == "search-youtube" and not entities.get("PLATFORM"):
            entities["PLATFORM"] = "youtube"
        check = _check_contract(str(command), entities)
        item = {
            "text": text,
            "intent": out.get("intent"),
            "command": command,
            "entities": entities,
            "contract_check": check,
        }
        results.append(item)

    out_path = Path("artifacts/fe_contract_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Đã ghi kết quả kiểm tra hợp đồng FE vào: {out_path}")
    ok = sum(1 for r in results if r["contract_check"].get("status") == "ok")  # type: ignore[union-attr]
    print(f"   {ok}/{len(results)} câu thoả điều kiện entity tối thiểu theo contract (xem file JSON để soi chi tiết).")


if __name__ == "__main__":
    main()


