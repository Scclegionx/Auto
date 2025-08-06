import json
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from config import model_config, intent_config, entity_config, command_config

class DataProcessor:
    """Xử lý dữ liệu cho Intent Recognition, Entity Extraction và Command Processing"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.intent_label2id = {label: idx for idx, label in enumerate(intent_config.intent_labels)}
        self.intent_id2label = {idx: label for label, idx in self.intent_label2id.items()}
        self.entity_label2id = {label: idx for idx, label in enumerate(entity_config.entity_labels)}
        self.entity_id2label = {idx: label for label, idx in self.entity_label2id.items()}
        self.command_label2id = {label: idx for idx, label in enumerate(command_config.command_labels)}
        self.command_id2label = {idx: label for label, idx in self.command_label2id.items()}
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Tải dataset từ file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def map_intent_to_command(self, intent: str) -> str:
        """Map intent sang command tương ứng"""
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
    
    def align_labels(self, text: str, entities: List[Dict], values: List[Dict]) -> List[str]:
        """
        Chuyển đổi entities và values thành nhãn IOB2 cho từng token
        Sử dụng cấu trúc List Dict với text và label
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Tạo mapping từ text gốc sang tokens
        text_lower = text.lower()
        
        # Xử lý entities (RECEIVER)
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity.get("text", "")
                entity_label = entity.get("label", "")
            else:
                entity_text = str(entity)
                entity_label = "FAMILY_RELATIONSHIP"  # Default label
            
            entity_lower = entity_text.lower()
            start_pos = text_lower.find(entity_lower)
            if start_pos != -1:
                # Tìm tokens tương ứng với entity
                entity_tokens = self.tokenizer.tokenize(entity_text)
                if len(entity_tokens) > 0:
                    # Tìm vị trí bắt đầu của entity trong tokens
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            # Map entity label to IOB2 format
                            if entity_label in ["FAMILY_RELATIONSHIP", "CONTACT_PERSON"]:
                                labels[i] = "B-RECEIVER"
                                for j in range(i+1, i+len(entity_tokens)):
                                    if j < len(labels):
                                        labels[j] = "I-RECEIVER"
                            break
        
        # Xử lý values
        for value in values:
            if isinstance(value, dict):
                value_text = value.get("text", "")
                value_label = value.get("label", "")
            else:
                value_text = str(value)
                value_label = "MESSAGE_CONTENT"  # Default label
            
            value_lower = value_text.lower()
            start_pos = text_lower.find(value_lower)
            if start_pos != -1:
                # Phân loại value type dựa trên label
                if value_label in ["TIME_EXPRESSION", "DATE_EXPRESSION"]:
                    label_prefix = "TIME"
                elif value_label in ["MESSAGE_CONTENT", "REMINDER_CONTENT"]:
                    label_prefix = "MESSAGE"
                else:
                    label_prefix = "MESSAGE"  # Default
                
                value_tokens = self.tokenizer.tokenize(value_text)
                if len(value_tokens) > 0:
                    # Tìm vị trí bắt đầu của value trong tokens
                    for i in range(len(tokens) - len(value_tokens) + 1):
                        if tokens[i:i+len(value_tokens)] == value_tokens:
                            labels[i] = f"B-{label_prefix}"
                            for j in range(i+1, i+len(value_tokens)):
                                if j < len(labels):
                                    labels[j] = f"I-{label_prefix}"
                            break
        
        return labels
    
    def _is_time_value(self, value: str) -> bool:
        """Kiểm tra xem value có phải là thời gian không"""
        time_patterns = [
            r'\d+[h:]\d*',  # 10h, 6h30
            r'\d+\s+giờ\s+(sáng|chiều|tối)',  # 5 giờ chiều
            r'\d+\s+giờ\s+kém\s+\d+',  # 8 giờ kém 15
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, value.lower()):
                return True
        return False
    
    def prepare_intent_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Intent Recognition"""
        processed_data = []
        
        for item in dataset:
            text = item["input"]
            intent = item["command"]
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Encode intent
            intent_id = self.intent_label2id[intent]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_id": intent_id,
                "text": text,
                "intent": intent
            })
        
        return processed_data
    
    def prepare_entity_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Entity Extraction"""
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
    
    def prepare_command_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Command Processing"""
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
            
            # Encode command
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