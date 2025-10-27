import json
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from training.configs.config import model_config, intent_config, entity_config, value_config, command_config
# Removed Enhanced Dynamic BIO Generator - use entities directly

class DataProcessor:
    """Xử lý dữ liệu cho Intent Recognition, Entity Extraction và Command Processing"""
    
    def __init__(self, config=None):
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
            # Unified 13 commands
            "call": "call",
            "make-video-call": "make-video-call",
            "send-mess": "send-mess",
            "add-contacts": "add-contacts",
            "play-media": "play-media",
            "view-content": "view-content",
            "search-internet": "search-internet",
            "search-youtube": "search-youtube",
            "get-info": "get-info",
            "set-alarm": "set-alarm",
            "set-event-calendar": "set-event-calendar",
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
            # Unified 13 commands
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],
            "make-video-call": ["gọi video", "video call", "facetime", "gọi hình ảnh"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],
            "add-contacts": ["thêm", "lưu", "danh bạ", "liên lạc", "số điện thoại"],
            "play-media": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "mở nhạc", "chơi nhạc", "phát"],
            "view-content": ["xem", "mở", "hiển thị", "bài báo", "link", "ảnh"],
            "search-internet": ["tìm kiếm", "tìm", "kiếm", "tra cứu", "tìm hiểu", "google"],
            "search-youtube": ["youtube", "yt", "video youtube", "tìm video"],
            "get-info": ["thời tiết", "nhiệt độ", "tin tức", "đọc tin", "kiểm tra", "xem tin"],
            "set-alarm": ["đặt báo thức", "báo thức", "hẹn giờ", "đánh thức", "chuông báo", "lúc", "giờ", "phút"],
            "set-event-calendar": ["nhắc nhở", "ghi nhớ", "lời nhắc", "nhắc tôi", "tạo lời nhắc", "sự kiện", "lịch"],
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
            "add-contacts": ["gọi", "nhắn tin", "phát nhạc"],
            "play-media": ["gọi", "nhắn tin", "kiểm tra", "tìm kiếm"],
            "view-content": ["gọi", "nhắn tin", "phát nhạc"],
            "search-internet": ["gọi", "nhắn tin", "phát nhạc"],
            "search-youtube": ["gọi", "nhắn tin", "phát nhạc"],
            "get-info": ["gọi", "nhắn tin", "phát nhạc"],
            "set-alarm": ["gửi tin nhắn", "gọi", "nhắn tin", "sự kiện"],
            "set-event-calendar": ["gọi", "nhắn tin", "phát nhạc", "báo thức"],
            "open-cam": ["gọi", "nhắn tin", "phát nhạc"],
            "control-device": ["gọi", "nhắn tin", "phát nhạc"]
        }
        
        if intent in conflicting_keywords:
            conflicting = conflicting_keywords[intent]
            
            for keyword in conflicting:
                if keyword in text_lower:
                    confidence -= 0.3
                    break
        
        # Special logic for set-alarm vs set-event-calendar distinction
        if intent == "set-alarm":
            # Boost confidence if time-related keywords are present
            time_keywords = ["lúc", "giờ", "phút", "sáng", "chiều", "tối", "hôm nay", "ngày mai"]
            if any(keyword in text_lower for keyword in time_keywords):
                confidence += 0.2
        elif intent == "set-event-calendar":
            # Penalize if time keywords are present (should be set-alarm instead)
            time_keywords = ["lúc", "giờ", "phút"]
            if any(keyword in text_lower for keyword in time_keywords):
                confidence -= 0.2
        
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
                # Use spans instead of entities for better alignment
                spans = item.get("spans", [])
                entity_labels = self.align_labels(text, spans, values)
            
            entity_label_ids = []
            for label in entity_labels:
                if label in self.entity_label2id:
                    entity_label_ids.append(self.entity_label2id[label])
                else:
                    entity_label_ids.append(self.entity_label2id["O"])
            
            # Pad entity labels và đảm bảo khớp với seq_len
            seq_len = int(encoding["input_ids"].shape[1])
            o_id = self.entity_label2id.get("O", 0)
            entity_label_ids = [o_id] + entity_label_ids[:seq_len-2] + [o_id]  # [CLS] + content + [SEP]
            entity_label_ids = entity_label_ids[:seq_len] + [o_id] * max(0, seq_len - len(entity_label_ids))
            
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
            o_id = self.value_label2id.get("O", 0)
            value_label_ids = [o_id] + value_label_ids[:seq_len-2] + [o_id]  # [CLS] + content + [SEP]
            value_label_ids = value_label_ids[:seq_len] + [o_id] * max(0, seq_len - len(value_label_ids))
            
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
            
            # Get entities directly from dataset
            entities = item.get("entities", [])
            spans = item.get("spans", [])
            values = item.get("values", [])
            
            # Create simple BIO tags from spans (which have valid start/end positions)
            bio_tags = self._create_bio_tags_from_entities(text, spans)
            labels = bio_tags
            
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
            
            # Consistent padding with [CLS] + content + [SEP] format
            seq_len = int(encoding["input_ids"].shape[1])
            o_id = self.entity_label2id["O"]
            label_ids = [o_id] + label_ids[:seq_len-2] + [o_id]  # [CLS] + content + [SEP]
            label_ids = label_ids[:seq_len] + [o_id] * max(0, seq_len - len(label_ids))
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": label_ids,
                "text": text,
                "entities": entities,
                "values": values
            })
        
        return processed_data
    
    def _create_bio_tags_from_entities(self, text: str, entities: List[Dict]) -> List[str]:
        """Tạo BIO tags đơn giản từ entities - sử dụng span data nếu có"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Initialize BIO tags
        bio_tags = ["O"] * len(tokens)
        
        # Process each entity
        for entity in entities:
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
                        # Check if label already has B- prefix
                        if label.startswith('B-'):
                            bio_tags[token_idx] = label
                        else:
                            bio_tags[token_idx] = f"B-{label}"
                    else:
                        # Check if label already has B- prefix
                        if label.startswith('B-'):
                            bio_tags[token_idx] = f"I-{label[2:]}"  # Remove B- and add I-
                        else:
                            bio_tags[token_idx] = f"I-{label}"
        
        return bio_tags
    
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
            
            # Consistent padding with [CLS] + content + [SEP] format
            seq_len = int(encoding["input_ids"].shape[1])
            o_id = self.value_label2id.get("O", 0)
            label_ids = [o_id] + label_ids[:seq_len-2] + [o_id]  # [CLS] + content + [SEP]
            label_ids = label_ids[:seq_len] + [o_id] * max(0, seq_len - len(label_ids))
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "value_labels": label_ids,
                "text": text,
                "values": values
            })
        
        return processed_data
    
    def align_labels(self, text: str, entities: List[Dict], values: List[Dict]) -> List[str]:
        """Align entity và value labels với tokens sử dụng span positions"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Process entities using span positions
        for entity in entities:
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
                            if i+j < len(labels):
                                if j == 0:
                                    # Check if label already has B- prefix
                                    if entity_label.startswith('B-'):
                                        labels[i+j] = entity_label
                                    else:
                                        labels[i+j] = f"B-{entity_label}"
                                else:
                                    # Check if label already has I- prefix
                                    if entity_label.startswith('B-'):
                                        labels[i+j] = f"I-{entity_label[2:]}"  # Remove B- and add I-
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
            
            # Ensure value_text is string
            if isinstance(value_text, (int, float)):
                value_text = str(value_text)
            elif not isinstance(value_text, str):
                value_text = ""
            
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
        """Chuẩn bị dữ liệu cho multi-task learning với rule-based extraction cho RECEIVER và MESSAGE"""
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
                return_tensors="pt"
            )
            
            # Intent label
            intent_label = self.intent_label2id.get(intent, 0)
            
            # Entity labels (IOB2 format) - rule-based with improved alignment
            # Entity labels (IOB2 format) - rule-based
            entity_labels = self._extract_entities_with_receiver(text)
            
            # Value labels (IOB2 format) - rule-based
            value_labels = self._extract_values_with_message(text)
            
            # Command label - use consistent mapping
            command = self.intent_to_command_mapping.get(intent, "unknown")
            command_label = self.command_label2id.get(command, 0)
            
            # FIX: Convert string labels to IDs to avoid mixing str and int
            # Entity labels conversion
            entity_label_ids = []
            for lab in entity_labels:
                entity_label_ids.append(self.entity_label2id.get(lab, self.entity_label2id["O"]))
            
            # Value labels conversion
            value_label_ids = []
            for lab in value_labels:
                value_label_ids.append(self.value_label2id.get(lab, self.value_label2id["O"]))
            
            # Đảm bảo độ dài nhãn khớp với seq_len (bao gồm special tokens)
            seq_len = int(encoding["input_ids"].shape[1])
            
            # Pad entity labels
            o_id_ent = self.entity_label2id["O"]
            entity_label_ids = [o_id_ent] + entity_label_ids[:seq_len-2] + [o_id_ent]  # [CLS] + content + [SEP]
            entity_label_ids = entity_label_ids[:seq_len] + [o_id_ent] * max(0, seq_len - len(entity_label_ids))
            
            # Pad value labels
            o_id_val = self.value_label2id["O"]
            value_label_ids = [o_id_val] + value_label_ids[:seq_len-2] + [o_id_val]  # [CLS] + content + [SEP]
            value_label_ids = value_label_ids[:seq_len] + [o_id_val] * max(0, seq_len - len(value_label_ids))
            
            multi_task_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_label": intent_label,
                "entity_labels": entity_label_ids,  # Use IDs, not strings
                "value_labels": value_label_ids,    # Use IDs, not strings
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
    
    def _extract_values_with_message_improved(self, text: str, offset_mapping: Optional[List[Tuple[int, int]]] = None) -> List[str]:
        """Trích xuất values với offset mapping để alignment chính xác hơn"""
        if offset_mapping is None:
            # Fallback to original method
            return self._extract_values_with_message(text)
        
        tokens = self.tokenizer.tokenize(text)
        value_labels = ['O'] * len(tokens)
        
        # Xử lý MESSAGE với offset mapping - avoid overwriting
        message_patterns = [
            r'rằng\s+(.+?)(?=$|[.!?])',
            r'nhắn\s+(.+?)(?=$|[.!?])',
            r'gửi\s+(.+?)(?=$|[.!?])'
        ]
        
        matched = False
        for pattern in message_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char, end_char = match.span(1)
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start is not None and token_end is not None:
                        if token_start <= start_char < token_end or token_start < end_char <= token_end:
                            if i < len(value_labels):
                                value_labels[i] = "B-MESSAGE" if token_start <= start_char < token_end else "I-MESSAGE"
                matched = True
                break
            if matched:
                break
        
        # Xử lý NUMBER với offset mapping (chỉ khi chưa được gán TIME) - avoid overwriting
        number_patterns = [r'\b(\d+)\b']
        matched = False
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
                matched = True
                break
            if matched:
                break
        
        return value_labels 