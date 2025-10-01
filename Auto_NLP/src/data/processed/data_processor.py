import json
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from src.training.configs.config import model_config, intent_config, entity_config, value_config, command_config

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
        
        
        self.intent_to_command_mapping = {
            "call": "call",
            "send-mess": "send-mess", 
            "make-video-call": "make-video-call",
            "check-messages": "check-messages",
            "open-app": "open-app",
            "play-media": "play-media",
            "search-content": "search-content",
            "set-reminder": "set-reminder",
            "set-alarm": "set-alarm",
            "check-weather": "check-weather",
            "general-conversation": "general-conversation",
            "unknown": "unknown"
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Tải dataset từ file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    
    
    def calculate_intent_confidence(self, text: str, intent: str) -> float:
        confidence = 0.5  # Base confidence

        intent_keywords = {
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "ghi nhớ"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],
            "set-reminder": ["đặt nhắc nhở", "ghi nhớ", "lời nhắc", "nhắc tôi", "tạo lời nhắc"],
            "check-weather": ["thời tiết", "nhiệt độ", "mưa", "nắng", "dự báo thời tiết"],
            "play-media": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc"],
            "read-news": ["đọc tin tức", "tin tức", "báo", "thời sự", "cập nhật tin"],
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
            "call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "send-mess": ["gọi", "điện thoại", "alo"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin"],
            "set-reminder": ["gọi", "nhắn tin", "phát nhạc"],
            "check-weather": ["gọi", "nhắn tin", "phát nhạc"],
            "play-media": ["gọi", "nhắn tin", "kiểm tra"],
            "read-news": ["gọi", "nhắn tin", "phát nhạc"],
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
    
    def prepare_multi_task_data(self, dataset: List[Dict], use_rule_based: bool = False) -> List[Dict]:
        """Chuẩn bị dữ liệu cho multi-task learning với lựa chọn rule-based hoặc dataset-based"""
        multi_task_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Prepare intent label
            intent_label = self.intent_label2id.get(intent, 0)
            
            # Prepare entity labels
            if use_rule_based:
                entity_labels = self._extract_entities_with_receiver(text)
            else:
                entity_labels = self.align_labels(text, entities, values)
            
            entity_label_ids = []
            for label in entity_labels:
                if label in self.entity_label2id:
                    entity_label_ids.append(self.entity_label2id[label])
                else:
                    entity_label_ids.append(self.entity_label2id["O"])
            
            # Pad entity labels và đảm bảo khớp với seq_len
            seq_len = int(encoding["input_ids"].shape[1])
            entity_label_ids = ["O"] + entity_label_ids[:seq_len-2] + ["O"]  # [CLS] + content + [SEP]
            entity_label_ids = entity_label_ids[:seq_len] + ["O"] * max(0, seq_len - len(entity_label_ids))
            
            # Prepare value labels
            if use_rule_based:
                value_labels = self._extract_values_with_message(text)
            else:
                value_labels = self.align_value_labels(text, values)
            
            value_label_ids = []
            for label in value_labels:
                if label in self.value_label2id:
                    value_label_ids.append(self.value_label2id[label])
                else:
                    value_label_ids.append(self.value_label2id["O"])
            
            # Pad value labels và đảm bảo khớp với seq_len
            value_label_ids = ["O"] + value_label_ids[:seq_len-2] + ["O"]  # [CLS] + content + [SEP]
            value_label_ids = value_label_ids[:seq_len] + ["O"] * max(0, seq_len - len(value_label_ids))
            
            # Prepare command label - use consistent mapping
            command = self.intent_to_command_mapping.get(intent, "unknown")
            command_label = self.command_label2id.get(command, 0)
            
            multi_task_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_label": intent_label,
                "entity_labels": entity_label_ids,
                "value_labels": value_label_ids,
                "command_label": command_label,
                "text": text,
                "original_intent": intent,
                "entities": entities,
                "values": values
            })
        
        return multi_task_data
    
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
        
        if predicted_intent == "set-alarm" and not any(word in text_lower for word in ["giờ", "phút", "sáng", "chiều", "tối"]):
            validation_result["confidence_adjusted"] *= 0.8
            validation_result["warnings"].append("Thiếu thông tin thời gian")
        
        return validation_result
    
    def map_intent_to_command(self, intent: str) -> str:
        """Map intent to command using kebab-case convention"""
        return self.intent_to_command_mapping.get(intent, "unknown")
    
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
            
            labels = self.align_labels(text, entities, values)
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            label_ids = []
            for label in labels:
                if label in self.entity_label2id:
                    label_ids.append(self.entity_label2id[label])
                else:
                    label_ids.append(self.entity_label2id["O"])
            
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
            
            value_labels = self.align_value_labels(text, values)
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            label_ids = []
            for label in value_labels:
                if label in self.value_label2id:
                    label_ids.append(self.value_label2id[label])
                else:
                    label_ids.append(self.value_label2id.get("O", 0))
            
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
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        for entity in entities:
            entity_text = entity.get("text", "")
            # Chuẩn hóa key: ưu tiên "label", fallback "type"
            entity_label = entity.get("label", entity.get("type", ""))
            
            if entity_text and entity_label:
                entity_tokens = self.tokenizer.tokenize(entity_text)
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
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
        
        for value in values:
            value_text = value.get("text", "")
            # Chuẩn hóa key: ưu tiên "label", fallback "type"
            value_label = value.get("label", value.get("type", ""))
            
            if value_text and value_label:
                value_tokens = self.tokenizer.tokenize(value_text)
                for i in range(len(tokens) - len(value_tokens) + 1):
                    if tokens[i:i+len(value_tokens)] == value_tokens:
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
        
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            if entity_text and entity_label:
                extracted["entities"][entity_label] = entity_text
        
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
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )

            # Use get() with fallback to avoid KeyError
            command_id = self.command_label2id.get(command, self.command_label2id.get("unknown", 0))
            
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
    
    def prepare_multi_task_data_rule_based(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho multi-task learning với rule-based extraction cho RECEIVER và MESSAGE"""
        multi_task_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            
            # Tokenize text with offset mapping for better alignment
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            # Intent label
            intent_label = self.intent_label2id.get(intent, 0)
            
            # Entity labels (IOB2 format) - rule-based with improved alignment
            offsets = encoding["offset_mapping"].squeeze(0).tolist()
            entity_labels = self._extract_entities_with_receiver_improved(text, offsets)
            
            # Value labels (IOB2 format) - rule-based with improved alignment
            value_labels = self._extract_values_with_message_improved(text, offsets)
            
            # Command label - use consistent mapping
            command = self.intent_to_command_mapping.get(intent, "unknown")
            command_label = self.command_label2id.get(command, 0)
            
            # Đảm bảo độ dài nhãn khớp với seq_len (bao gồm special tokens)
            seq_len = int(encoding["input_ids"].shape[1])
            
            # Chỉnh sửa entity_labels để khớp với seq_len
            entity_labels = ["O"] + entity_labels[:seq_len-2] + ["O"]  # [CLS] + content + [SEP]
            entity_labels = entity_labels[:seq_len] + ["O"] * max(0, seq_len - len(entity_labels))
            
            # Chỉnh sửa value_labels để khớp với seq_len
            value_labels = ["O"] + value_labels[:seq_len-2] + ["O"]  # [CLS] + content + [SEP]
            value_labels = value_labels[:seq_len] + ["O"] * max(0, seq_len - len(value_labels))
            
            multi_task_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_label": intent_label,
                "entity_labels": entity_labels,
                "value_labels": value_labels,
                "command_label": command_label,
                "text": text,
                "original_intent": intent
            })
        
        return multi_task_data
    
    def _extract_entities_with_receiver(self, text: str) -> List[str]:
        """Trích xuất entities với xử lý đặc biệt cho RECEIVER - regex cải tiến"""
        tokens = self.tokenizer.tokenize(text)
        entity_labels = ['O'] * len(tokens)
        
        # Xử lý RECEIVER - tìm tên người nhận (cải tiến regex)
        receiver_patterns = [
            r'(?:nhắn|gửi)\s+cho\s+(.+?)(?=\s+rằng\b|$)',
            r'\bcho\s+(.+?)(?=\s+rằng\b|$)',
            r'([A-Za-zÀ-ỹ\s]+)\s+rằng'
        ]
        
        for pattern in receiver_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                receiver_name = match.group(1).strip()
                receiver_tokens = self.tokenizer.tokenize(receiver_name)
                for i in range(len(tokens) - len(receiver_tokens) + 1):
                    if tokens[i:i+len(receiver_tokens)] == receiver_tokens:
                        entity_labels[i] = "B-RECEIVER"
                        for j in range(1, len(receiver_tokens)):
                            entity_labels[i+j] = "I-RECEIVER"
                        break
                break
        
        # Xử lý PLATFORM - tìm nền tảng gửi (cải tiến)
        platform_patterns = [
            r'\bvào\s+(?!lúc\b)(\w+)(?=\s|$)',  # Negative lookahead để tránh "vào lúc"
            r'\btrên\s+(\w+)(?=\s|$)',
            r'\bqua\s+(\w+)(?=\s|$)'
        ]
        for pattern in platform_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                platform = match.group(1).strip()
                platform_tokens = self.tokenizer.tokenize(platform)
                for i in range(len(tokens) - len(platform_tokens) + 1):
                    if tokens[i:i+len(platform_tokens)] == platform_tokens:
                        entity_labels[i] = "B-PLATFORM"
                        for j in range(1, len(platform_tokens)):
                            entity_labels[i+j] = "I-PLATFORM"
                        break
                break
        
        # Xử lý TIME - tìm thời gian (cải tiến, ưu tiên TIME trước NUMBER)
        time_patterns = [
            r'\blúc\s+(\d{1,2}(?::\d{2})?h?)\b',
            r'\b(\d{1,2}h(?:\d{2})?)\b',
            r'\b(\d{1,2}:\d{2})\b'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_text = match.group(1)  # Lấy group(1) thay vì group(0)
                time_tokens = self.tokenizer.tokenize(time_text)
                for i in range(len(tokens) - len(time_tokens) + 1):
                    if tokens[i:i+len(time_tokens)] == time_tokens:
                        entity_labels[i] = "B-TIME"
                        for j in range(1, len(time_tokens)):
                            entity_labels[i+j] = "I-TIME"
                        break
                break
        
        return entity_labels
    
    def _extract_values_with_message(self, text: str) -> List[str]:
        """Trích xuất values với xử lý đặc biệt cho MESSAGE - regex cải tiến"""
        tokens = self.tokenizer.tokenize(text)
        value_labels = ['O'] * len(tokens)
        
        # Xử lý MESSAGE - tìm nội dung tin nhắn sau "rằng" (cải tiến regex)
        message_patterns = [
            r'rằng\s+(.+?)(?=$|[.!?])',
            r'nhắn\s+(.+?)(?=$|[.!?])',
            r'gửi\s+(.+?)(?=$|[.!?])'
        ]
        
        for pattern in message_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                message_text = match.group(1).strip()
                message_tokens = self.tokenizer.tokenize(message_text)
                for i in range(len(tokens) - len(message_tokens) + 1):
                    if tokens[i:i+len(message_tokens)] == message_tokens:
                        value_labels[i] = "B-MESSAGE"
                        for j in range(1, len(message_tokens)):
                            value_labels[i+j] = "I-MESSAGE"
                        break
                break
        
        # Xử lý NUMBER - tìm số (chỉ khi chưa được gán TIME)
        number_patterns = [r'\b(\d+)\b']
        for pattern in number_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number_text = match.group(1)
                number_tokens = self.tokenizer.tokenize(number_text)
                for i in range(len(tokens) - len(number_tokens) + 1):
                    if tokens[i:i+len(number_tokens)] == number_tokens:
                        # Chỉ gán NUMBER nếu chưa được gán TIME
                        if not any("TIME" in value_labels[j] for j in range(i, i+len(number_tokens))):
                            value_labels[i] = "B-NUMBER"
                            for j in range(1, len(number_tokens)):
                                value_labels[i+j] = "I-NUMBER"
                        break
                break
        
        return value_labels
    
    def _extract_entities_with_receiver_improved(self, text: str, offset_mapping: Optional[List[Tuple[int, int]]] = None) -> List[str]:
        """Trích xuất entities với offset mapping để alignment chính xác hơn"""
        if offset_mapping is None:
            # Fallback to original method
            return self._extract_entities_with_receiver(text)
        
        tokens = self.tokenizer.tokenize(text)
        entity_labels = ['O'] * len(tokens)
        
        # Xử lý RECEIVER với offset mapping
        receiver_patterns = [
            r'(?:nhắn|gửi)\s+cho\s+(.+?)(?=\s+rằng\b|$)',
            r'\bcho\s+(.+?)(?=\s+rằng\b|$)',
            r'([A-Za-zÀ-ỹ\s]+)\s+rằng'
        ]
        
        for pattern in receiver_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)  # Group 1 (receiver name)
                # Map character positions to token positions
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-RECEIVER" if token_start <= start_char < token_end else "I-RECEIVER"
                break
        
        # Xử lý PLATFORM với offset mapping
        platform_patterns = [
            r'\bvào\s+(?!lúc\b)(\w+)(?=\s|$)',  # Negative lookahead để tránh "vào lúc"
            r'\btrên\s+(\w+)(?=\s|$)',
            r'\bqua\s+(\w+)(?=\s|$)'
        ]
        
        for pattern in platform_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-PLATFORM" if token_start <= start_char < token_end else "I-PLATFORM"
                break
        
        # Xử lý TIME với offset mapping
        time_patterns = [
            r'\blúc\s+(\d{1,2}(?::\d{2})?h?)\b',
            r'\b(\d{1,2}h(?:\d{2})?)\b',
            r'\b(\d{1,2}:\d{2})\b'
        ]
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-TIME" if token_start <= start_char < token_end else "I-TIME"
                break
        
        return entity_labels
    
    def _extract_values_with_message_improved(self, text: str, offset_mapping: Optional[List[Tuple[int, int]]] = None) -> List[str]:
        """Trích xuất values với offset mapping để alignment chính xác hơn"""
        if offset_mapping is None:
            # Fallback to original method
            return self._extract_values_with_message(text)
        
        tokens = self.tokenizer.tokenize(text)
        value_labels = ['O'] * len(tokens)
        
        # Xử lý MESSAGE với offset mapping
        message_patterns = [
            r'rằng\s+(.+?)(?=$|[.!?])',
            r'nhắn\s+(.+?)(?=$|[.!?])',
            r'gửi\s+(.+?)(?=$|[.!?])'
        ]
        
        for pattern in message_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(value_labels):
                                value_labels[i] = "B-MESSAGE" if token_start <= start_char < token_end else "I-MESSAGE"
                break
        
        # Xử lý NUMBER với offset mapping (chỉ khi chưa được gán TIME)
        number_patterns = [r'\b(\d+)\b']
        for pattern in number_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(value_labels):
                                # Chỉ gán NUMBER nếu chưa được gán TIME
                                if not any("TIME" in value_labels[j] for j in range(max(0, i-1), min(len(value_labels), i+2))):
                                    value_labels[i] = "B-NUMBER" if token_start <= start_char < token_end else "I-NUMBER"
                break
        
        return value_labels 