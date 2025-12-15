from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from models.inference.model_loader import MultiTaskInference, _select_checkpoint


class TrainedModelInference:
    """Compatibility wrapper sử dụng `MultiTaskInference` cho core hybrid system."""

    def __init__(self, model_path: str, _device: Optional[str] = None):
        model_dir = Path(model_path)
        if model_dir.is_file():
            checkpoint_path = model_dir
            model_dir = model_dir.parent
        else:
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory không tồn tại: {model_dir}")
            checkpoint_path = _select_checkpoint(model_dir)

        tokenizer_path = str(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config.json tại {config_path}")

        self._inference = MultiTaskInference(str(checkpoint_path), tokenizer_path, str(config_path))
        self.model_loaded = True
        self.model_path = model_dir

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Gọi multi-task model và hậu xử lý entity cho hybrid system.
        - Loại bỏ entity rác từ special tokens (<s>, </s>, [PAD], ...)
        - Gom MESSAGE thành một chuỗi duy nhất (ghép các mảnh)
        - Lọc PLATFORM theo whitelist (zalo, messenger, viber, youtube, ...)
        """
        result = self._inference.predict(text)

        raw_entities: List[Dict[str, Any]] = result.get("entities", [])
        cleaned_entities: List[Tuple[str, str]] = []

        special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "[PAD]", "[CLS]", "[SEP]"}
        platform_whitelist = {
            "zalo",
            "messenger",
            "facebook",
            "fb",
            "viber",
            "youtube",
            "zalo call",
            "zalo video",
        }

        for ent in raw_entities:
            label = ent.get("label")
            text_value = (ent.get("text") or "").strip()
            if not label or not text_value:
                continue
            if text_value in special_tokens:
                continue

            # Lọc PLATFORM: chỉ giữ nếu khớp whitelist (bỏ các mảnh như "tin", "nhắn"...)
            if label == "PLATFORM":
                low = text_value.lower().strip()
                compact = low.replace(" ", "")
                if compact not in {p.replace(" ", "") for p in platform_whitelist}:
                    continue

            cleaned_entities.append((label, text_value))

        entity_map: Dict[str, Any] = {}
        message_pieces: List[str] = []
        query_pieces: List[str] = []

        # Một số label có thể xuất hiện nhiều lần (MESSAGE, QUERY, ...),
        # các label khác thì ưu tiên:
        # - nếu chỉ có 1 span: dùng span đó
        # - nếu có nhiều span: giữ span "tốt" hơn (dài hơn, không phải trigger như "bật", "mở", ...)
        trigger_verbs = {"bật", "tắt", "mở", "giảm", "tăng", "tìm", "tra", "hỏi"}

        for label, text_value in cleaned_entities:
            if label == "MESSAGE":
                message_pieces.append(text_value)
                continue

            if label == "QUERY":
                query_pieces.append(text_value)
                continue

            # Các label khác: DEVICE, TIME, DATE, PLATFORM, ...
            existing = entity_map.get(label)
            if existing is None:
                entity_map[label] = text_value
                continue

            # Nếu đã có value cho label này, chọn value "tốt" hơn:
            # - Ưu tiên span dài hơn rõ ràng hơn (ví dụ "điều hòa 26 độ" thay cho "bật")
            # - Nếu existing là trigger verb (bật/tắt/mở/...) và text_value dài hơn => thay thế
            existing_len = len(existing)
            new_len = len(text_value)

            if existing in trigger_verbs and new_len > existing_len:
                entity_map[label] = text_value
            elif new_len > existing_len and label in {"DEVICE"}:
                entity_map[label] = text_value
            # Các trường hợp còn lại: giữ nguyên existing để tránh thay đổi quá mạnh

        if message_pieces:
            merged = " ".join(message_pieces)
            merged = " ".join(merged.split())
            entity_map["MESSAGE"] = merged

        if query_pieces:
            # Nếu có nhiều mảnh QUERY, bỏ bớt các trigger verb đứng riêng lẻ như "mở", "tìm", "tra"
            # và ưu tiên phần nội dung chính.
            filtered: List[str] = []
            for q in query_pieces:
                if q in trigger_verbs and len(query_pieces) > 1:
                    continue
                filtered.append(q)
            if not filtered:
                filtered = query_pieces
            merged_query = " ".join(filtered)
            merged_query = " ".join(merged_query.split())
            entity_map["QUERY"] = merged_query

        return {
            "intent": result.get("intent", "unknown"),
            "confidence": result.get("intent_confidence", 0.0),
            "entities": entity_map,
            "command": result.get("command", result.get("intent", "unknown")),
            "model_type": "multi-task",
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path),
        }

def load_trained_model(model_name: str = "phobert_multitask", device: Optional[torch.device] = None) -> TrainedModelInference:
    """
    Load trained model
    
    Args:
        model_name: Name of the model directory
        device: Device to load model on
        
    Returns:
        TrainedModelInference instance
    """
    model_path = Path("models") / model_name
    return TrainedModelInference(str(model_path), device)

# Test function
if __name__ == "__main__":
    print("🚀 Testing TrainedModelInference...")
    
    try:
        # Load model
        model = load_trained_model("phobert_large_intent_model")
        
        # Test cases
        test_cases = [
            "gọi điện cho mẹ",
            "bật đèn phòng khách",
            "tìm kiếm nhạc trên youtube",
            "đặt báo thức 7 giờ sáng",
            "gửi tin nhắn cho bạn"
        ]
        
        print(f"\n🧪 Testing with {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_case}'")
            result = model.predict(test_case)
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Command: {result['command']}")
            print(f"   Entities: {result['entities']}")
            print(f"   Model type: {result['model_type']}")
        
        # Print model info
        print(f"\n📊 Model Info:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ TrainedModelInference test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
