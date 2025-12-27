from __future__ import annotations

"""
So sánh output raw (MultiTaskInference) và hybrid (TrainedModelInference)
để xem hybrid can thiệp nhiều hay ít, từ đó quyết định triển khai rule hợp lý.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.model_loader import load_trained_model as load_hybrid_model
from models.inference.model_loader import load_multi_task_model


TEST_TEXTS: List[str] = [
    # send-mess / giao tiếp
    "gửi tin nhắn cho cháu nội là hôm nay ông bận không sang chơi được",
    "nhắn tin cho con gái bảo tối nay bố không ăn cơm ở nhà",
    "soạn tin cho bác sĩ là huyết áp hôm nay cao hơn mọi ngày",
    "gửi tin nhắn cho thằng Tý bảo mai 7 giờ sang đón ông",
    "gửi tin cho con trai bằng zalo nói là ông đến muộn một chút",
    "Thêm liên hệ Phan Thanh Chi số 0911 357 591 vào danh bạ",
    "Lưu mẹ Lan 0987 654 321",
    "Gọi cho anh Trường qua Zalo",
    "Bấm số 0911357591 gọi ngay",
    "Gọi video cho cô Hạnh bằng Messenger",
    "Mở cuộc gọi hình cho cháu Bảo qua Zalo",
    "Nhắn tin cho chị Mai: chiều nay con đến đón ạ",
    "Gửi SMS cho số 0987654321: bác khoẻ không",
    "Nhắn qua Zalo cho cậu Sáu: mai 7 giờ đi khám",
    "gọi điện cho con trai bằng zalo",
    "bật cuộc gọi video cho cháu ngoại",
    "gọi cho bác sĩ gia đình giúp tôi",
    "gọi video cho cháu nội để xem mặt nó",

    # set-alarm / reminder
    "Đặt báo thức lúc 6 giờ 30 sáng nhắc uống thuốc",
    "Nhắc tôi 3 giờ chiều mỗi ngày",
    "Hẹn báo thức thứ Hai 7 giờ",
    "đặt báo thức 6 rưỡi sáng mai nhắc uống thuốc",
    "nhắc tôi uống thuốc huyết áp lúc 8 giờ tối",
    "đặt chuông 5 giờ sáng mai để dậy tập thể dục",
    "tạo nhắc nhở chiều nay ba giờ gọi cho con gái",
    "nhắc ông 9 giờ tối đo đường huyết",

    # control-device
    "Bật đèn phòng khách",
    "Giảm âm lượng xuống 30 phần trăm",
    "Tăng độ sáng màn hình lên hai mức",
    "bật đèn phòng khách giúp tôi",
    "tắt đèn phòng ngủ đi con",
    "bật quạt trần trong phòng khách lên",
    "tắt tivi trong phòng khách",
    "bật điều hòa 26 độ ở phòng ngủ",

    # open-cam
    "Mở camera trước để chụp selfie",
    "Chuyển sang quay video camera sau",
    "mở camera trước lên giùm bà",
    "mở camera sau xem ngoài cổng có ai không",
    "bật camera trong phòng khách cho tôi xem",

    # search / media
    "Tìm trên web cách đo huyết áp tại nhà",
    "Tra cứu thời tiết ở Đà Nẵng ngày mai",
    "Hỏi Google: triệu chứng cảm cúm người già",
    "Tìm trên YouTube: bài tập thở cho người cao tuổi",
    "YouTube hướng dẫn dùng Zalo",
    "Cho tôi biết thời tiết Hà Nội tối nay",
    "Tin tức mới nhất hôm nay là gì",
    "mở nhạc trữ tình cho ông nghe",
    "bật bài hát nhạc đỏ mà ông hay nghe",
    "tìm kiếm tin tức thời sự hôm nay",
    "mở youtube cho tôi xem hài Hoài Linh",
]


def _norm_raw_entities(ents: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ent in ents or []:
        label = ent.get("label")
        text = (ent.get("text") or "").strip()
        if not label or not text:
            continue
        out.append((label, text))
    return sorted(set(out))


def _norm_hybrid_entities(ents: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for label, value in (ents or {}).items():
        text = (value or "").strip()
        if not label or not text:
            continue
        out.append((label, text))
    return sorted(set(out))


def main() -> None:
    # Raw multi-task model (entities dạng list)
    raw_model = load_multi_task_model("phobert_multitask")
    # Hybrid wrapper (entities dạng dict + hậu xử lý)
    hybrid_model = load_hybrid_model("phobert_multitask")

    results: List[Dict[str, Any]] = []

    n_samples = 0
    n_intent_diff = 0
    n_command_diff = 0
    n_entity_diff = 0

    for text in TEST_TEXTS:
        n_samples += 1
        raw_out = raw_model.predict(text)
        hyb_out = hybrid_model.predict(text)

        raw_intent = raw_out.get("intent")
        raw_intent_conf = raw_out.get("intent_confidence", raw_out.get("confidence", 0.0))
        raw_command = raw_out.get("command")

        hyb_intent = hyb_out.get("intent")
        hyb_intent_conf = hyb_out.get("intent_confidence", hyb_out.get("confidence", 0.0))
        hyb_command = hyb_out.get("command")

        if raw_intent != hyb_intent:
            n_intent_diff += 1
        if raw_command != hyb_command:
            n_command_diff += 1

        raw_ents_list: List[Dict[str, Any]] = raw_out.get("entities", [])
        hyb_ents_dict: Dict[str, Any] = hyb_out.get("entities", {})

        raw_norm = _norm_raw_entities(raw_ents_list)
        hyb_norm = _norm_hybrid_entities(hyb_ents_dict)

        entity_changed = raw_norm != hyb_norm
        if entity_changed:
            n_entity_diff += 1

        results.append(
            {
                "text": text,
                "raw": {
                    "intent": raw_intent,
                    "intent_confidence": raw_intent_conf,
                    "command": raw_command,
                    "entities": raw_ents_list,
                },
                "hybrid": {
                    "intent": hyb_intent,
                    "intent_confidence": hyb_intent_conf,
                    "command": hyb_command,
                    "entities": hyb_ents_dict,
                },
                "entity_changed": entity_changed,
            }
        )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "tmp_infer_compare_hybrid_raw.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved comparison to {out_path}")
    print(f"  Tổng số câu: {n_samples}")
    print(f"  Số câu intent khác giữa raw vs hybrid: {n_intent_diff}")
    print(f"  Số câu command khác giữa raw vs hybrid: {n_command_diff}")
    print(f"  Số câu entities khác giữa raw vs hybrid: {n_entity_diff}")


if __name__ == "__main__":
    main()








