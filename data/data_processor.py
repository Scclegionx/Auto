import json
import re
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from config import model_config, intent_config, entity_config, value_config, command_config

class DataProcessor:
    """Xử lý dữ liệu cho Intent Recognition, Entity Extraction và Command Processing"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.intent_label2id = {label: idx for idx, label in enumerate(intent_config.intent_labels)}
        self.intent_id2label = {idx: label for label, idx in self.intent_label2id.items()}
        self.entity_label2id = {label: idx for idx, label in enumerate(entity_config.entity_labels)}
        self.entity_id2label = {idx: label for label, idx in self.entity_label2id.items()}
        self.value_label2id = {label: idx for idx, label in enumerate(value_config.value_labels)}
        self.value_id2label = {idx: label for label, idx in self.value_label2id.items()}
        self.command_label2id = {label: idx for idx, label in enumerate(command_config.command_labels)}
        self.command_id2label = {idx: label for label, idx in self.command_label2id.items()}
        
        # Data augmentation patterns - Cập nhật theo dataset thực tế
        self.augmentation_patterns = {
            "call": [
                "gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"
            ],
            "make-call": [
                "gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"
            ],
            "set-alarm": [
                "đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "ghi nhớ"
            ],
            "send-mess": [
                "gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"
            ],
            "send-message": [
                "gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"
            ],
            "set-reminder": [
                "đặt nhắc nhở", "ghi nhớ", "lời nhắc", "nhắc tôi", "tạo lời nhắc"
            ],
            "check-weather": [
                "thời tiết", "nhiệt độ", "mưa", "nắng", "dự báo thời tiết"
            ],
            "play-media": [
                "phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"
            ],
            "play-content": [
                "phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"
            ],
            "play-audio": [
                "phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"
            ],
            "read-news": [
                "đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin"
            ],
            "read-content": [
                "đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin"
            ],
            "check-health-status": [
                "kiểm tra sức khỏe", "đo", "theo dõi", "chỉ số", "tình trạng"
            ],
            "general-conversation": [
                "xin chào", "tạm biệt", "cảm ơn", "trò chuyện", "nói chuyện"
            ],
            "open-app": [
                "mở ứng dụng", "khởi động", "chạy", "vào", "mở"
            ],
            "search-content": [
                "tìm kiếm", "tìm", "kiếm", "tra cứu", "tìm hiểu"
            ],
            "make-video-call": [
                "gọi video", "video call", "facetime", "gọi hình ảnh"
            ]
        }
        
        # Intent mapping - Cập nhật theo dataset thực tế
        self.intent_to_command_mapping = {
            "adjust-settings": "adjust_settings",
            "app-tutorial": "app_tutorial",
            "browse-social-media": "browse_social_media",
            "call": "call",
            "check-device-status": "check_device_status",
            "check-health-status": "check_health_status",
            "check-messages": "check_messages",
            "check-weather": "check_weather",
            "control-device": "control_device",
            "general-conversation": "general_conversation",
            "make-call": "make_call",
            "make-video-call": "make_video_call",
            "navigation-help": "navigation_help",
            "open-app": "open_app",
            "open-app-action": "open_app_action",
            "play-audio": "play_audio",
            "play-content": "play_content",
            "play-media": "play_media",
            "provide-instructions": "provide_instructions",
            "read-content": "read_content",
            "read-news": "read_news",
            "search-content": "search_content",
            "search-internet": "search_internet",
            "send-message": "send_message",
            "send-mess": "send_mess",
            "set-alarm": "set_alarm",
            "set-reminder": "set_reminder",
            "view-content": "view_content"
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Tải dataset từ file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def augment_intent_data(self, dataset: List[Dict], augmentation_factor: float = 0.3) -> List[Dict]:
        augmented_data = []
        
        for item in dataset:
            intent = item.get("command", "")
            text = item.get("input", "")

            augmented_data.append(item)

            if intent in self.augmentation_patterns and random.random() < augmentation_factor:
                patterns = self.augmentation_patterns[intent]
                
                for pattern in patterns:
                    augmented_text = self._replace_keywords(text, intent, pattern)
                    if augmented_text != text:
                        augmented_item = item.copy()
                        augmented_item["input"] = augmented_text
                        augmented_data.append(augmented_item)
        
        return augmented_data
    
    def _replace_keywords(self, text: str, intent: str, new_pattern: str) -> str:
        intent_keywords = {
            "call": ["gọi", "điện thoại", "alo"],
            "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text"]
        }
        
        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            for keyword in keywords:
                if keyword in text.lower():
                    text = re.sub(rf'\b{re.escape(keyword)}\b', new_pattern, text, flags=re.IGNORECASE)
                    break
        
        return text
    
    def calculate_intent_confidence(self, text: str, intent: str) -> float:
        confidence = 0.5  # Base confidence

        intent_keywords = {
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "make-call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "ghi nhớ"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],
            "send-message": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],
            "set-reminder": ["đặt nhắc nhở", "ghi nhớ", "lời nhắc", "nhắc tôi", "tạo lời nhắc"],
            "check-weather": ["thời tiết", "nhiệt độ", "mưa", "nắng", "dự báo thời tiết"],
            "play-media": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"],
            "play-content": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"],
            "play-audio": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"],
            "read-news": ["đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin"],
            "read-content": ["đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin"],
            "check-health-status": ["kiểm tra sức khỏe", "đo", "theo dõi", "chỉ số", "tình trạng"],
            "general-conversation": ["xin chào", "tạm biệt", "cảm ơn", "trò chuyện", "nói chuyện"],
            "open-app": ["mở ứng dụng", "khởi động", "chạy", "vào", "mở"],
            "search-content": ["tìm kiếm", "tìm", "kiếm", "tra cứu", "tìm hiểu"],
            "make-video-call": ["gọi video", "video call", "facetime", "gọi hình ảnh"]
        }
        
        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            text_lower = text.lower()
            
            for keyword in keywords:
                if keyword in text_lower:
                    confidence += 0.2
                    break

            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            confidence += min(keyword_count * 0.1, 0.3)

        conflicting_keywords = {
            "call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "make-call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "send-mess": ["gọi", "điện thoại", "alo"],
            "send-message": ["gọi", "điện thoại", "alo"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin"],
            "set-reminder": ["gọi", "nhắn tin", "phát nhạc"],
            "check-weather": ["gọi", "nhắn tin", "phát nhạc"],
            "play-media": ["gọi", "nhắn tin", "kiểm tra"],
            "play-content": ["gọi", "nhắn tin", "kiểm tra"],
            "play-audio": ["gọi", "nhắn tin", "kiểm tra"],
            "read-news": ["gọi", "nhắn tin", "phát nhạc"],
            "read-content": ["gọi", "nhắn tin", "phát nhạc"],
            "check-health-status": ["gọi", "nhắn tin", "phát nhạc"],
            "general-conversation": ["gọi", "nhắn tin", "phát nhạc", "kiểm tra"],
            "make-video-call": ["nhắn tin", "gửi tin nhắn", "soạn tin"]
        }
        
        if intent in conflicting_keywords:
            conflicting = conflicting_keywords[intent]
            text_lower = text.lower()
            
            for keyword in conflicting:
                if keyword in text_lower:
                    confidence -= 0.3
                    break
        
        return min(max(confidence, 0.1), 1.0)
    
    def prepare_intent_data_with_confidence(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu intent với confidence scores"""
        intent_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            
            if intent in self.intent_label2id:
                confidence = self.calculate_intent_confidence(text, intent)

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=model_config.max_length,
                    return_tensors="pt"
                )
                
                intent_data.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "intent_label": self.intent_label2id[intent],
                    "confidence": confidence,
                    "text": text,
                    "original_intent": intent
                })
        
        return intent_data
    
    def validate_intent_prediction(self, predicted_intent: str, confidence: float, 
                                 original_text: str) -> Dict:
        """Validate intent prediction với các rule-based checks"""
        validation_result = {
            "is_valid": True,
            "confidence_adjusted": confidence,
            "warnings": [],
            "suggestions": []
        }

        if confidence < intent_config.confidence_threshold:
            validation_result["is_valid"] = False
            validation_result["warnings"].append(f"Confidence quá thấp: {confidence:.3f}")
            validation_result["suggestions"].append("Cần xác nhận lại từ người dùng")

        text_lower = original_text.lower()

        if predicted_intent == "call" and any(word in text_lower for word in ["nhắn tin", "gửi tin nhắn"]):
            validation_result["confidence_adjusted"] *= 0.7
            validation_result["warnings"].append("Có từ khóa mâu thuẫn với intent")
        
        # Kiểm tra context
        if predicted_intent == "set-alarm" and not any(word in text_lower for word in ["giờ", "phút", "sáng", "chiều", "tối"]):
            validation_result["confidence_adjusted"] *= 0.8
            validation_result["warnings"].append("Thiếu thông tin thời gian")
        
        return validation_result
    
    def map_intent_to_command(self, intent: str) -> str:
        intent_to_command = {
            "call": "make_call",
            "check-health-status": "check_health_status",
            "check-weather": "check_weather",
            "express-emotion": "express_emotion",
            "express-fatigue": "express_fatigue",
            "find-information": "find_information",
            "general-conversation": "general_conversation",
            "general-request": "general_request",
            "play-media": "play_media",
            "read-news": "read_news",
            "report-symptom": "report_symptom",
            "request-comfort": "request_comfort",
            "request-entertainment": "request_entertainment",
            "request-instruction": "request_instruction",
            "send-mess": "send_message",
            "set-alarm": "set_alarm",
            "set-reminder": "set_reminder"
        }
        return intent_to_command.get(intent, "unknown")
    
    def extract_entities_from_dict_list(self, entities: List[Dict]) -> List[str]:
        """Trích xuất text từ list dict entities"""
        return [entity.get("text", "") for entity in entities if isinstance(entity, dict)]
    
    def extract_values_from_dict_list(self, values: List[Dict]) -> List[str]:
        """Trích xuất text từ list dict values"""
        return [value.get("text", "") for value in values if isinstance(value, dict)]
    
    def get_entity_labels_from_dict_list(self, entities: List[Dict]) -> List[str]:
        """Trích xuất label từ list dict entities"""
        return [entity.get("label", "") for entity in entities if isinstance(entity, dict)]
    
    def get_value_labels_from_dict_list(self, values: List[Dict]) -> List[str]:
        """Trích xuất label từ list dict values"""
        return [value.get("label", "") for value in values if isinstance(value, dict)]
    
    def prepare_entity_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Entity Extraction - Cập nhật cho dataset mới"""
        processed_data = []
        
        for item in dataset:
            text = item["input"]
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            # Tạo labels cho từng token
            labels = self.align_labels(text, entities, values)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Encode labels
            label_ids = []
            for label in labels:
                if label in self.entity_label2id:
                    label_ids.append(self.entity_label2id[label])
                else:
                    label_ids.append(self.entity_label2id["O"])
            
            # Pad labels
            while len(label_ids) < model_config.max_length:
                label_ids.append(self.entity_label2id["O"])
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": label_ids,
                "text": text,
                "entities": entities,
                "values": values
            })
        
        return processed_data
    
    def prepare_value_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Value Extraction - Mới thêm"""
        processed_data = []
        
        for item in dataset:
            text = item["input"]
            values = item.get("values", [])
            
            # Tạo labels cho từng token
            value_labels = self.align_value_labels(text, values)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Encode value labels
            label_ids = []
            for label in value_labels:
                if label in self.value_label2id:
                    label_ids.append(self.value_label2id[label])
                else:
                    label_ids.append(self.value_label2id.get("O", 0))
            
            # Pad labels
            while len(label_ids) < model_config.max_length:
                label_ids.append(self.value_label2id.get("O", 0))
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "value_labels": label_ids,
                "text": text,
                "values": values
            })
        
        return processed_data
    
    def align_labels(self, text: str, entities: List[Dict], values: List[Dict]) -> List[str]:
        """Align entity và value labels với tokens - Cập nhật cho dataset mới"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Process entities
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            
            if entity_text and entity_label:
                # Find entity position in tokens
                entity_tokens = self.tokenizer.tokenize(entity_text)
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        # Mark entity tokens
                        for j in range(len(entity_tokens)):
                            if j == 0:
                                labels[i+j] = f"B-{entity_label}"
                            else:
                                labels[i+j] = f"I-{entity_label}"
                        break
        
        return labels
    
    def align_value_labels(self, text: str, values: List[Dict]) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Process values
        for value in values:
            value_text = value.get("text", "")
            value_label = value.get("label", "")
            
            if value_text and value_label:
                # Find value position in tokens
                value_tokens = self.tokenizer.tokenize(value_text)
                for i in range(len(tokens) - len(value_tokens) + 1):
                    if tokens[i:i+len(value_tokens)] == value_tokens:
                        # Mark value tokens
                        for j in range(len(value_tokens)):
                            if j == 0:
                                labels[i+j] = f"B-{value_label}"
                            else:
                                labels[i+j] = f"I-{value_label}"
                        break
        
        return labels
    
    def extract_entities_and_values(self, text: str, entities: List[Dict], values: List[Dict]) -> Dict:
        extracted = {
            "entities": {},
            "values": {}
        }
        
        # Extract entities
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            if entity_text and entity_label:
                extracted["entities"][entity_label] = entity_text
        
        # Extract values
        for value in values:
            value_text = value.get("text", "")
            value_label = value.get("label", "")
            if value_text and value_label:
                extracted["values"][value_label] = value_text
        
        return extracted
    
    def prepare_command_data(self, dataset: List[Dict]) -> List[Dict]:
        processed_data = []
        
        for item in dataset:
            text = item["input"]
            intent = item["command"]
            command = self.map_intent_to_command(intent)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )

            command_id = self.command_label2id[command]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "command_id": command_id,
                "text": text,
                "intent": intent,
                "command": command
            })
        
        return processed_data
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Chia dataset thành train và validation"""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:] 