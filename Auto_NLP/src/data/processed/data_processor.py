import json
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from training.configs.config import ModelConfig, IntentConfig, EntityConfig, CommandConfig
from data.entity_schema import (
    canonicalize_entity_dict,
    canonicalize_entity_label,
    canonicalize_label_sequence,
)

# Instantiate config classes
model_config = ModelConfig()
intent_config = IntentConfig()
entity_config = EntityConfig()
command_config = CommandConfig()
# Removed Enhanced Dynamic BIO Generator - use entities directly

class DataProcessor:
    """Xử lý dữ liệu cho Intent Recognition, Entity Extraction và Command Processing"""
    
    def __init__(self, config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.intent_label2id = {label: idx for idx, label in enumerate(intent_config.intent_labels)}
        self.intent_id2label = {idx: label for label, idx in self.intent_label2id.items()}
        self.entity_label2id = {label: idx for idx, label in enumerate(entity_config.entity_labels)}
        self.entity_id2label = {idx: label for label, idx in self.entity_label2id.items()}
        self.command_label2id = {label: idx for idx, label in enumerate(command_config.command_labels)}
        self.command_id2label = {idx: label for label, idx in self.command_label2id.items()}
        
        
        self.intent_to_command_mapping = {
            "call": "call",
            "make-video-call": "make-video-call",
            "send-mess": "send-mess",
            "add-contacts": "add-contacts",
            "search-internet": "search-internet",
            "search-youtube": "search-youtube",
            "get-info": "get-info",
            "set-alarm": "set-alarm",
            "open-cam": "open-cam",
            "control-device": "control-device",
            "unknown": "unknown"
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Tải dataset từ file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    
    
    def calculate_intent_confidence(self, text: str, intent: str) -> float:
        confidence = 0.5  # Base confidence
        text_lower = text.lower()  # Define once to avoid repetition

        intent_keywords = {
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "make-video-call": ["gọi video", "video call", "facetime", "gọi hình ảnh"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],
            "add-contacts": ["thêm", "lưu", "danh bạ", "liên lạc", "số điện thoại"],
            "search-internet": ["tìm kiếm", "tìm", "kiếm", "tra cứu", "tìm hiểu", "google"],
            "search-youtube": ["youtube", "yt", "video youtube", "tìm video"],
            "get-info": ["thời tiết", "nhiệt độ", "tin tức", "đọc tin", "kiểm tra", "xem tin"],
            "set-alarm": ["đặt báo thức", "báo thức", "hẹn giờ", "đánh thức", "chuông báo", "lúc", "giờ", "phút"],
            "open-cam": ["mở camera", "bật camera", "chụp ảnh", "quay video", "camera"],
            "control-device": ["wifi", "bluetooth", "đèn pin", "âm lượng", "chế độ im lặng", "định vị"]
        }
        
        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            
            for keyword in keywords:
                if keyword in text_lower:
                    confidence += 0.2
                    break

            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            confidence += min(keyword_count * 0.1, 0.3)

        conflicting_keywords = {
            "call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "make-video-call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],
            "send-mess": ["gọi", "điện thoại", "alo"],
            "add-contacts": ["gọi", "nhắn tin"],
            "search-internet": ["gọi", "nhắn tin"],
            "search-youtube": ["gọi", "nhắn tin"],
            "get-info": ["gọi", "nhắn tin"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin"],
            "open-cam": ["gọi", "nhắn tin"],
            "control-device": ["gọi", "nhắn tin"]
        }
        
        if intent in conflicting_keywords:
            conflicting = conflicting_keywords[intent]
            
            for keyword in conflicting:
                if keyword in text_lower:
                    confidence -= 0.3
                    break
        
        # Boost confidence cho set-alarm khi có từ khóa thời gian
        if intent == "set-alarm":
            time_keywords = ["lúc", "giờ", "phút", "sáng", "chiều", "tối", "hôm nay", "ngày mai"]
            if any(keyword in text_lower for keyword in time_keywords):
                confidence += 0.2
        
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
            entities = self._canonicalize_entity_dicts(item.get("entities", []))
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_tokens_mask = encoding.pop("special_tokens_mask").squeeze().tolist()
            
            # Prepare intent label
            intent_label = self.intent_label2id.get(intent, 0)
            
            # Prepare entity labels
            if use_rule_based:
                entity_labels = self._extract_entities_with_receiver(text)
            else:
                # Use spans instead of entities for better alignment
                spans = self._canonicalize_entity_dicts(item.get("spans", []))
                entity_labels = self.align_labels(text, spans)

            entity_labels = canonicalize_label_sequence(entity_labels)
            
            label_id_sequence: List[int] = [
                self.entity_label2id.get(label, self.entity_label2id["O"]) for label in entity_labels
            ]

            # Map labels lên chuỗi có cả special tokens, padding đều đánh -100
            seq_len = int(encoding["input_ids"].shape[1])
            non_special_capacity = sum(1 for token_flag in special_tokens_mask if token_flag == 0)
            trimmed_labels = label_id_sequence[:non_special_capacity]
            label_iter = iter(trimmed_labels)

            final_entity_label_ids: List[int] = []
            for is_special in special_tokens_mask:
                if is_special:
                    final_entity_label_ids.append(-100)
                else:
                    try:
                        final_entity_label_ids.append(next(label_iter))
                    except StopIteration:
                        final_entity_label_ids.append(self.entity_label2id["O"])

            # Đảm bảo độ dài chuẩn (phòng trường hợp tokenizer trả về nhiều hơn max_length)
            if len(final_entity_label_ids) < seq_len:
                final_entity_label_ids.extend([-100] * (seq_len - len(final_entity_label_ids)))
            else:
                final_entity_label_ids = final_entity_label_ids[:seq_len]
            
            # Prepare command label - use consistent mapping
            command = self.intent_to_command_mapping.get(intent, "unknown")
            command_label = self.command_label2id.get(command, 0)
            
            multi_task_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_label": intent_label,
                "entity_labels": final_entity_label_ids,
                "command_label": command_label,
                "text": text,
                "original_intent": intent,
                "entities": entities
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
    
    def get_entity_labels_from_dict_list(self, entities: List[Dict]) -> List[str]:
        """Trích xuất label từ list dict entities"""
        return [entity.get("label", "") for entity in entities if isinstance(entity, dict)]

    def _canonicalize_entity_dicts(self, entities: List[Dict]) -> List[Dict]:
        """Chuẩn hóa danh sách entity dict về schema mới."""
        if not entities:
            return []
        return [canonicalize_entity_dict(entity) for entity in entities if isinstance(entity, dict)]
    
    def prepare_entity_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Entity Extraction - Cập nhật cho dataset mới"""
        processed_data = []
        
        for item in dataset:
            text = item["input"]
            
            # Get entities directly from dataset
            entities = self._canonicalize_entity_dicts(item.get("entities", []))
            spans = self._canonicalize_entity_dicts(item.get("spans", []))
            
            # Create simple BIO tags from spans (which have valid start/end positions)
            bio_tags = self._create_bio_tags_from_entities(text, spans)
            labels = canonicalize_label_sequence(bio_tags)
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_tokens_mask = encoding.pop("special_tokens_mask").squeeze().tolist()
            
            label_ids = []
            for label in labels:
                if label in self.entity_label2id:
                    label_ids.append(self.entity_label2id[label])
                else:
                    label_ids.append(self.entity_label2id["O"])
            
            # Consistent padding với special tokens bị bỏ qua trong loss (đặt -100)
            seq_len = int(encoding["input_ids"].shape[1])
            non_special_capacity = sum(1 for flag in special_tokens_mask if flag == 0)
            trimmed_labels = label_ids[:non_special_capacity]
            label_iter = iter(trimmed_labels)

            final_label_ids: List[int] = []
            for is_special in special_tokens_mask:
                if is_special:
                    final_label_ids.append(-100)
                else:
                    try:
                        final_label_ids.append(next(label_iter))
                    except StopIteration:
                        final_label_ids.append(self.entity_label2id["O"])

            if len(final_label_ids) < seq_len:
                final_label_ids.extend([-100] * (seq_len - len(final_label_ids)))
            else:
                final_label_ids = final_label_ids[:seq_len]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": final_label_ids,
                "text": text,
                "entities": entities
            })
        
        return processed_data
    
    def _create_bio_tags_from_entities(self, text: str, entities: List[Dict]) -> List[str]:
        """Tạo BIO tags đơn giản từ entities - sử dụng span data nếu có"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Initialize BIO tags
        bio_tags = ["O"] * len(tokens)
        
        # Process each entity
        for entity in self._canonicalize_entity_dicts(entities):
            label = entity.get('label', '')
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            
            # Skip if no label or invalid span
            if not label or start == -1 or end == -1 or start >= end:
                continue
            
            # Find tokens that overlap with entity span
            entity_tokens = []
            char_pos = 0
            
            for i, token in enumerate(tokens):
                # Clean token
                clean_token = token.replace('##', '').replace('▁', '')
                
                # Find token position in original text
                token_start = text.find(clean_token, char_pos)
                if token_start != -1:
                    token_end = token_start + len(clean_token)
                    
                    # Check if token overlaps with entity
                    if not (token_end <= start or token_start >= end):
                        entity_tokens.append(i)
                    
                    char_pos = token_end
            
            # Assign BIO tags - check if label already has B- prefix
            for i, token_idx in enumerate(entity_tokens):
                if token_idx < len(bio_tags):
                    if i == 0:
                        if label.startswith('B-'):
                            tag = label
                        else:
                            tag = f"B-{label}"
                    else:
                        if label.startswith('B-'):
                            tag = f"I-{label[2:]}"  # Remove B- and add I-
                        else:
                            tag = f"I-{label}"
                    bio_tags[token_idx] = canonicalize_entity_label(tag)
        
        return bio_tags
    
    def align_labels(self, text: str, entities: List[Dict]) -> List[str]:
        """Align entity labels với tokens sử dụng span positions"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Process entities using span positions
        for entity in self._canonicalize_entity_dicts(entities):
            entity_label = entity.get("label", entity.get("type", ""))
            entity_text = entity.get("text", "")
            
            if entity_label and entity_text:
                # Tokenize entity text to find matching tokens
                entity_tokens = self.tokenizer.tokenize(entity_text)
                
                # Find entity tokens in main tokens
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        # Assign BIO tags - check if label already has B- prefix
                        for j in range(len(entity_tokens)):
                            if i + j < len(labels):
                                if j == 0:
                                    if entity_label.startswith('B-'):
                                        tag = entity_label
                                    else:
                                        tag = f"B-{entity_label}"
                                else:
                                    if entity_label.startswith('B-'):
                                        tag = f"I-{entity_label[2:]}"  # Remove B- and add I-
                                    else:
                                        tag = f"I-{entity_label}"
                                labels[i + j] = canonicalize_entity_label(tag)
                        break
        
        return labels
    
    def extract_entities(self, text: str, entities: List[Dict]) -> Dict[str, str]:
        extracted: Dict[str, str] = {}
        for entity in self._canonicalize_entity_dicts(entities):
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            if entity_text and entity_label:
                extracted[canonicalize_entity_label(entity_label)] = entity_text
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
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8, stratify: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Chia dataset thành train và validation với shuffle và stratify"""
        import random
        from collections import defaultdict
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Shuffle data
        random.shuffle(data)
        
        if not stratify:
            # Simple split without stratification
            split_idx = int(len(data) * train_ratio)
            return data[:split_idx], data[split_idx:]
        
        # Stratified split to maintain class distribution
        # Group by intent/command
        groups = defaultdict(list)
        for item in data:
            intent = item.get("command", "unknown")
            groups[intent].append(item)
        
        train_data = []
        val_data = []
        
        for intent, items in groups.items():
            # Shuffle within each group
            random.shuffle(items)
            split_idx = int(len(items) * train_ratio)
            train_data.extend(items[:split_idx])
            val_data.extend(items[split_idx:])
        
        # Shuffle final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        return train_data, val_data
    
    def prepare_multi_task_data_rule_based(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu multi-task với rule-based cho các entity trọng yếu"""
        multi_task_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            
            # Tokenize text for multi-task learning
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_tokens_mask = encoding.pop("special_tokens_mask").squeeze().tolist()
            
            # Intent label
            intent_label = self.intent_label2id.get(intent, 0)
            
            # Entity labels (IOB2 format) - rule-based
            entity_labels = canonicalize_label_sequence(self._extract_entities_with_receiver(text))
            
            # Command label - use consistent mapping
            command = self.intent_to_command_mapping.get(intent, "unknown")
            command_label = self.command_label2id.get(command, 0)
            
            # Convert entity labels to IDs
            entity_label_ids = [
                self.entity_label2id.get(lab, self.entity_label2id["O"]) for lab in entity_labels
            ]

            seq_len = int(encoding["input_ids"].shape[1])
            non_special_capacity = sum(1 for flag in special_tokens_mask if flag == 0)
            trimmed_labels = entity_label_ids[:non_special_capacity]
            label_iter = iter(trimmed_labels)
            final_entity_ids: List[int] = []
            for is_special in special_tokens_mask:
                if is_special:
                    final_entity_ids.append(-100)
                else:
                    try:
                        final_entity_ids.append(next(label_iter))
                    except StopIteration:
                        final_entity_ids.append(self.entity_label2id["O"])

            if len(final_entity_ids) < seq_len:
                final_entity_ids.extend([-100] * (seq_len - len(final_entity_ids)))
            else:
                final_entity_ids = final_entity_ids[:seq_len]
            
            multi_task_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_label": intent_label,
                "entity_labels": final_entity_ids,
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
    
    def _extract_entities_with_receiver_improved(self, text: str, offset_mapping: Optional[List[Tuple[int, int]]] = None) -> List[str]:
        """Trích xuất entities với offset mapping để alignment chính xác hơn"""
        if offset_mapping is None:
            # Fallback to original method
            return self._extract_entities_with_receiver(text)
        
        tokens = self.tokenizer.tokenize(text)
        entity_labels = ['O'] * len(tokens)
        
        # Xử lý RECEIVER với offset mapping - avoid overwriting
        receiver_patterns = [
            r'(?:nhắn|gửi)\s+cho\s+(.+?)(?=\s+rằng\b|$)',
            r'\bcho\s+(.+?)(?=\s+rằng\b|$)',
            r'([A-Za-zÀ-ỹ\s]+)\s+rằng'
        ]
        
        matched = False
        for pattern in receiver_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)  # Group 1 (receiver name)
                # Map character positions to token positions
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-RECEIVER" if token_start <= start_char < token_end else "I-RECEIVER"
                matched = True
                break
            if matched:
                break
        
        # Xử lý PLATFORM với offset mapping - avoid overwriting
        platform_patterns = [
            r'\bvào\s+(?!lúc\b)(\w+)(?=\s|$)',  # Negative lookahead để tránh "vào lúc"
            r'\btrên\s+(\w+)(?=\s|$)',
            r'\bqua\s+(\w+)(?=\s|$)'
        ]
        
        matched = False
        for pattern in platform_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-PLATFORM" if token_start <= start_char < token_end else "I-PLATFORM"
                matched = True
                break
            if matched:
                break
        
        # Xử lý TIME với offset mapping - avoid overwriting
        time_patterns = [
            r'\blúc\s+(\d{1,2}(?::\d{2})?h?)\b',
            r'\b(\d{1,2}h(?:\d{2})?)\b',
            r'\b(\d{1,2}:\d{2})\b'
        ]
        
        matched = False
        for pattern in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(entity_labels):
                                entity_labels[i] = "B-TIME" if token_start <= start_char < token_end else "I-TIME"
                matched = True
                break
            if matched:
                break
        
        return entity_labels
    
    