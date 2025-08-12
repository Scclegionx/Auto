import json
import re
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, BatchEncoding
from config import model_config, intent_config, entity_config, command_config

class DataProcessor:
    """Xử lý dữ liệu cho Intent Recognition, Entity Extraction và Command Processing"""
    
    def __init__(self, max_seq_length: int = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.intent_label2id = {label: idx for idx, label in enumerate(intent_config.intent_labels)}
        self.intent_id2label = {idx: label for label, idx in self.intent_label2id.items()}
        self.entity_label2id = {label: idx for idx, label in enumerate(entity_config.entity_labels)}
        self.entity_id2label = {idx: label for label, idx in self.entity_label2id.items()}
        self.command_label2id = {label: idx for idx, label in enumerate(command_config.command_labels)}
        self.command_id2label = {idx: label for label, idx in self.command_label2id.items()}
        self.max_seq_length = max_seq_length or model_config.max_length
        
        # Initialize standard mappings for entities and values
        self._init_entity_mappings()
        self._init_value_mappings()
    
    def _init_entity_mappings(self):
        """Initialize mappings for entity labels to standard IOB2 tags"""
        self.entity_mapping = {
            "FAMILY_RELATIONSHIP": "RECEIVER",
            "CONTACT_PERSON": "RECEIVER",
            "LOCATION": "LOCATION",
            "TIME_EXPRESSION": "TIME",
            "DATE_EXPRESSION": "TIME",
            "ARTIST_NAME": "ARTIST",
            # Add any additional entity mappings here
        }
    
    def _init_value_mappings(self):
        """Initialize mappings for value labels to standard IOB2 tags"""
        self.value_mapping = {
            # Thời gian
            "TIME_EXPRESSION": "TIME",
            "DATE_EXPRESSION": "TIME",
            # Nội dung
            "MESSAGE_CONTENT": "MESSAGE",
            "REMINDER_CONTENT": "MESSAGE",
            # Thời tiết
            "WEATHER_CONDITION": "WEATHER",
            # Sức khỏe
            "HEALTH_METRIC": "HEALTH",
            "SYMPTOM": "HEALTH",
            # Media
            "MEDIA_CONTENT": "MEDIA",
            "MEDIA_TYPE": "MEDIA_TYPE",
            # Tin tức
            "NEWS_CATEGORY": "NEWS",
            "NEWS_SOURCE": "NEWS",
            "TOPIC_NEWS": "TOPIC",
            # Tìm kiếm
            "SEARCH_QUERY": "QUERY",
            # Cảm xúc
            "EMOTION": "EMOTION",
            # Giải trí
            "ENTERTAINMENT_TYPE": "ENTERTAINMENT",
            # Gọi điện
            "CALL_TYPE": "CALL_TYPE",
            # Tần suất và thời lượng
            "FREQUENCY": "FREQUENCY",
            "DURATION": "DURATION",
            # Hành động
            "ACTION_VERB": "ACTION",
            # Biểu đạt
            "EXPRESSION": "EXPRESSION",
            "FATIGUE_EXPRESSION": "FATIGUE",
            # Hướng dẫn
            "INSTRUCTION_TOPIC": "INSTRUCTION",
            # Kích hoạt
            "TRIGGER": "TRIGGER",
            # Add any additional value mappings here
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Tải dataset từ file JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading dataset from {file_path}: {e}")
            return []
    
    def save_dataset(self, data: List[Dict], file_path: str) -> bool:
        """Lưu dataset vào file JSON"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving dataset to {file_path}: {e}")
            return False
    
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
        return intent_to_command.get(intent, "unknown_command")
    
    def extract_entities_from_dict_list(self, entities: List[Dict]) -> List[str]:
        """Trích xuất text từ list dict entities"""
        if not entities:
            return []
        return [entity.get("text", "") for entity in entities if isinstance(entity, dict) and "text" in entity]
    
    def extract_values_from_dict_list(self, values: List[Dict]) -> List[str]:
        """Trích xuất text từ list dict values"""
        if not values:
            return []
        return [value.get("text", "") for value in values if isinstance(value, dict) and "text" in value]
    
    def get_entity_labels_from_dict_list(self, entities: List[Dict]) -> List[str]:
        """Trích xuất label từ list dict entities"""
        if not entities:
            return []
        return [entity.get("label", "") for entity in entities if isinstance(entity, dict) and "label" in entity]
    
    def get_value_labels_from_dict_list(self, values: List[Dict]) -> List[str]:
        """Trích xuất label từ list dict values"""
        if not values:
            return []
        return [value.get("label", "") for value in values if isinstance(value, dict) and "label" in value]
    
    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa text trước khi xử lý"""
        # Thay thế nhiều khoảng trắng bằng một khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        # Loại bỏ khoảng trắng ở đầu và cuối
        text = text.strip()
        return text
    
    def _find_token_span(self, tokens: List[str], entity_tokens: List[str]) -> List[Tuple[int, int]]:
        """Tìm tất cả vị trí của entity_tokens trong tokens"""
        spans = []
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                spans.append((i, i + len(entity_tokens)))
        return spans
    
    def align_labels(self, text: str, entities: List[Dict], values: List[Dict]) -> List[str]:
        """
        Chuyển đổi entities và values thành nhãn IOB2 cho từng token
        Sử dụng cấu trúc List Dict với text và label
        """
        text = self._normalize_text(text)
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)
        
        # Xử lý entities
        for entity in entities:
            if not isinstance(entity, dict) or "text" not in entity or "label" not in entity:
                continue
                
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            
            # Bỏ qua entities trống
            if not entity_text.strip():
                continue
                
            # Tokenize entity
            entity_tokens = self.tokenizer.tokenize(entity_text)
            
            # Map entity label to standard format
            standard_label = self.entity_mapping.get(entity_label, "ENTITY")
            
            # Tìm tất cả vị trí của entity trong tokens
            spans = self._find_token_span(tokens, entity_tokens)
            
            # Gán nhãn IOB2 cho mỗi vị trí tìm thấy
            for start, end in spans:
                labels[start] = f"B-{standard_label}"
                for j in range(start+1, end):
                    labels[j] = f"I-{standard_label}"
        
        # Xử lý values
        for value in values:
            if not isinstance(value, dict) or "text" not in value or "label" not in value:
                continue
                
            value_text = value.get("text", "")
            value_label = value.get("label", "")
            
            # Bỏ qua values trống
            if not value_text.strip():
                continue
                
            # Tokenize value
            value_tokens = self.tokenizer.tokenize(value_text)
            
            # Map value label to standard format
            standard_label = self.value_mapping.get(value_label, "VALUE")
            
            # Tìm tất cả vị trí của value trong tokens
            spans = self._find_token_span(tokens, value_tokens)
            
            # Gán nhãn IOB2 cho mỗi vị trí tìm thấy - chỉ ghi đè lên nhãn O
            for start, end in spans:
                # Chỉ ghi đè nhãn O hoặc ghi đè theo ưu tiên nếu cần
                if labels[start] == "O":
                    labels[start] = f"B-{standard_label}"
                    for j in range(start+1, end):
                        if j < len(labels) and labels[j] == "O":
                            labels[j] = f"I-{standard_label}"
        
        return labels
    
    def _is_time_expression(self, text: str) -> bool:
        """Kiểm tra xem text có phải là biểu thức thời gian không"""
        time_patterns = [
            r'\d+[h:]\d*',  # 10h, 6h30, 10:30
            r'\d+\s+giờ(\s+(sáng|chiều|tối|trưa))?',  # 5 giờ, 5 giờ chiều
            r'\d+\s+giờ\s+kém\s+\d+',  # 8 giờ kém 15
            r'(sáng|chiều|tối|trưa)(\s+này)?',  # sáng này, chiều
            r'(hôm\s+nay|ngày\s+mai|hôm\s+qua)',  # hôm nay, ngày mai
            r'(thứ\s+\w+|chủ\s+nhật)(\s+này|\s+tới)?',  # thứ hai, chủ nhật này
            r'ngày\s+\d+(\s+tháng\s+\d+)?(\s+năm\s+\d+)?',  # ngày 15, ngày 15 tháng 6
            r'tháng\s+\d+(\s+năm\s+\d+)?',  # tháng 6, tháng 6 năm 2025
            r'\d+\/\d+(\/\d+)?',  # 15/6, 15/6/2025
        ]
        
        text_lower = text.lower()
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _is_frequency_expression(self, text: str) -> bool:
        """Kiểm tra xem text có phải là biểu thức tần suất không"""
        frequency_patterns = [
            r'hàng\s+(ngày|tuần|tháng|năm)',  # hàng ngày, hàng tuần
            r'mỗi\s+(ngày|tuần|tháng|năm)',  # mỗi ngày, mỗi tuần
            r'(hằng|hàng)\s+(sáng|chiều|tối)',  # hằng sáng, hàng chiều
            r'\d+\s+(lần|giờ|phút)\s+một',  # 2 lần một ngày, 3 giờ một lần
            r'(thường\s+xuyên|định\s+kỳ)',  # thường xuyên, định kỳ
        ]
        
        text_lower = text.lower()
        for pattern in frequency_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _is_duration_expression(self, text: str) -> bool:
        """Kiểm tra xem text có phải là biểu thức thời lượng không"""
        duration_patterns = [
            r'\d+\s+(giây|phút|tiếng|giờ|ngày|tuần|tháng|năm)',  # 5 phút, 2 giờ
            r'trong\s+\d+\s+(giây|phút|tiếng|giờ|ngày|tuần|tháng|năm)',  # trong 5 phút
            r'kéo\s+dài\s+\d+\s+(giây|phút|tiếng|giờ|ngày|tuần|tháng|năm)',  # kéo dài 5 phút
        ]
        
        text_lower = text.lower()
        for pattern in duration_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _infer_value_label(self, text: str, command: str) -> str:
        """Suy luận nhãn cho value dựa trên nội dung và command"""
        text_lower = text.lower()
        
        # Kiểm tra biểu thức thời gian
        if self._is_time_expression(text):
            return "TIME_EXPRESSION"
            
        # Kiểm tra biểu thức tần suất
        if self._is_frequency_expression(text):
            return "FREQUENCY"
            
        # Kiểm tra biểu thức thời lượng
        if self._is_duration_expression(text):
            return "DURATION"
            
        # Suy luận dựa trên command
        if command == "send-mess":
            return "MESSAGE_CONTENT"
        elif command == "set-reminder":
            return "REMINDER_CONTENT"
        elif command == "call":
            if "zoom" in text_lower or "facetime" in text_lower or "video" in text_lower:
                return "CALL_TYPE"
        elif command == "play-media":
            if "nhạc" in text_lower or "bài" in text_lower or "album" in text_lower:
                return "MEDIA_CONTENT"
            elif "phim" in text_lower or "video" in text_lower:
                return "MEDIA_TYPE"
        elif command == "check-weather":
            return "WEATHER_CONDITION"
        elif command == "read-news":
            if "thể thao" in text_lower or "kinh tế" in text_lower or "chính trị" in text_lower:
                return "NEWS_CATEGORY"
        elif command == "find-information":
            return "SEARCH_QUERY"
        
        # Default fallback
        return "UNKNOWN_VALUE"
    
    def _infer_entity_label(self, text: str, command: str) -> str:
        """Suy luận nhãn cho entity dựa trên nội dung và command"""
        text_lower = text.lower()
        
        # Các mối quan hệ gia đình phổ biến
        family_terms = [
            "mẹ", "ba", "bố", "cha", "bà", "ông", "chị", "anh", "em", "cậu", "mợ", 
            "dì", "chú", "bác", "cô", "con", "cháu", "vợ", "chồng", "gái", "trai"
        ]
        
        for term in family_terms:
            if term in text_lower or f"{term} " in text_lower or f" {term}" in text_lower:
                return "FAMILY_RELATIONSHIP"
        
        # Kiểm tra địa điểm
        location_indicators = ["ở", "tại", "quận", "huyện", "thành phố", "tỉnh", "đường", "phố"]
        for indicator in location_indicators:
            if indicator in text_lower:
                return "LOCATION"
        
        # Default là CONTACT_PERSON cho các command liên quan đến giao tiếp
        if command in ["call", "send-mess"]:
            return "CONTACT_PERSON"
        
        # Default fallback
        return "UNKNOWN_ENTITY"
    
    def _augment_entities_and_values(self, item: Dict) -> Dict:
        """Bổ sung entities và values bằng cách suy luận từ nội dung"""
        text = item.get("input", "")
        command = item.get("command", "")
        entities = item.get("entities", [])
        values = item.get("values", [])
        
        # Nếu đã có đầy đủ entities và values, không cần bổ sung
        if entities and values:
            return item
            
        # TODO: Triển khai các heuristic phức tạp hơn để xác định entities và values
        # Đây chỉ là một ví dụ đơn giản
            
        # Copy để không làm thay đổi input
        item_copy = item.copy()
        
        # Bổ sung entities và values nếu cần
        if command == "send-mess" and not any(e.get("label") == "FAMILY_RELATIONSHIP" for e in entities):
            # Tìm người nhận tin nhắn
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ["cho", "tới", "đến", "với"] and i < len(words) - 1:
                    potential_receiver = words[i+1]
                    if any(term in potential_receiver.lower() for term in ["mẹ", "ba", "anh", "chị", "em"]):
                        if "entities" not in item_copy:
                            item_copy["entities"] = []
                        item_copy["entities"].append({
                            "text": potential_receiver,
                            "label": "FAMILY_RELATIONSHIP"
                        })
                        break
        
        # Bổ sung nội dung tin nhắn nếu là send-mess và không có value
        if command == "send-mess" and not any(v.get("label") == "MESSAGE_CONTENT" for v in values):
            if ":" in text:
                message_content = text.split(":", 1)[1].strip()
                if message_content:
                    if "values" not in item_copy:
                        item_copy["values"] = []
                    item_copy["values"].append({
                        "text": message_content,
                        "label": "MESSAGE_CONTENT"
                    })
        
        return item_copy
    
    def enrich_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Làm giàu dataset bằng cách suy luận và bổ sung entities và values"""
        enriched_dataset = []
        
        for item in dataset:
            enriched_item = self._augment_entities_and_values(item)
            enriched_dataset.append(enriched_item)
            
        return enriched_dataset
    
    def prepare_intent_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Intent Recognition"""
        processed_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            
            if not text or not intent:
                continue
                
            # Normalize text
            text = self._normalize_text(text)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Encode intent
            intent_id = self.intent_label2id.get(intent, 0)  # Default to 0 if not found
            
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
            text = item.get("input", "")
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            if not text:
                continue
                
            # Normalize text
            text = self._normalize_text(text)
            
            # Tạo labels cho từng token
            labels = self.align_labels(text, entities, values)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
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
            while len(label_ids) < self.max_seq_length:
                label_ids.append(self.entity_label2id["O"])
            
            # Truncate if needed
            if len(label_ids) > self.max_seq_length:
                label_ids = label_ids[:self.max_seq_length]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label_ids),
                "text": text,
                "entities": entities,
                "values": values
            })
        
        return processed_data
    
    def prepare_command_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho Command Processing"""
        processed_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            
            if not text or not intent:
                continue
                
            # Normalize text
            text = self._normalize_text(text)
            
            # Map intent to command
            command = self.map_intent_to_command(intent)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Encode command
            command_id = self.command_label2id.get(command, 0)  # Default to 0 if not found
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "command_id": command_id,
                "text": text,
                "intent": intent,
                "command": command
            })
        
        return processed_data
    
    def prepare_joint_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho joint Intent và Entity model"""
        processed_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            if not text or not intent:
                continue
                
            # Normalize text
            text = self._normalize_text(text)
            
            # Tạo labels cho từng token
            entity_labels = self.align_labels(text, entities, values)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Encode intent
            intent_id = self.intent_label2id.get(intent, 0)
            
            # Encode entity labels
            entity_label_ids = []
            for label in entity_labels:
                if label in self.entity_label2id:
                    entity_label_ids.append(self.entity_label2id[label])
                else:
                    entity_label_ids.append(self.entity_label2id["O"])
            
            # Pad entity labels
            while len(entity_label_ids) < self.max_seq_length:
                entity_label_ids.append(self.entity_label2id["O"])
            
            # Truncate if needed
            if len(entity_label_ids) > self.max_seq_length:
                entity_label_ids = entity_label_ids[:self.max_seq_length]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_id": intent_id,
                "entity_labels": torch.tensor(entity_label_ids),
                "text": text,
                "intent": intent,
                "entities": entities,
                "values": values
            })
        
        return processed_data
    
    def prepare_unified_data(self, dataset: List[Dict]) -> List[Dict]:
        """Chuẩn bị dữ liệu cho mô hình thống nhất (Intent, Entity, Command)"""
        processed_data = []
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            if not text or not intent:
                continue
                
            # Normalize text
            text = self._normalize_text(text)
            
            # Map intent to command
            command = self.map_intent_to_command(intent)
            
            # Tạo labels cho từng token
            entity_labels = self.align_labels(text, entities, values)
            
            # Encode text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Encode intent
            intent_id = self.intent_label2id.get(intent, 0)
            
            # Encode command
            command_id = self.command_label2id.get(command, 0)
            
            # Encode entity labels
            entity_label_ids = []
            for label in entity_labels:
                if label in self.entity_label2id:
                    entity_label_ids.append(self.entity_label2id[label])
                else:
                    entity_label_ids.append(self.entity_label2id["O"])
            
            # Pad entity labels
            while len(entity_label_ids) < self.max_seq_length:
                entity_label_ids.append(self.entity_label2id["O"])
            
            # Truncate if needed
            if len(entity_label_ids) > self.max_seq_length:
                entity_label_ids = entity_label_ids[:self.max_seq_length]
            
            processed_data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "intent_id": intent_id,
                "entity_labels": torch.tensor(entity_label_ids),
                "command_id": command_id,
                "text": text,
                "intent": intent,
                "command": command,
                "entities": entities,
                "values": values
            })
        
        return processed_data
    
    def create_data_loaders(self, data: List[Dict], batch_size: int = 16, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Tạo DataLoader từ dữ liệu đã xử lý"""
        dataset = torch.utils.data.Dataset()
        
        # Implement dataset class based on data format
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = CustomDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Chia dataset thành train, validation và test"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Tổng tỷ lệ phải bằng 1"
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        train_idx = int(len(data) * train_ratio)
        val_idx = train_idx + int(len(data) * val_ratio)
        
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]
        
        return train_data, val_data, test_data
    
    def analyze_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Phân tích dataset để hiểu phân bố của dữ liệu"""
        analysis = {
            "total_examples": len(dataset),
            "intent_distribution": {},
            "entity_types": {},
            "value_types": {},
            "avg_text_length": 0,
            "max_text_length": 0,
            "min_text_length": float('inf') if dataset else 0,
            "examples_with_entities": 0,
            "examples_with_values": 0
        }
        
        total_length = 0
        
        for item in dataset:
            text = item.get("input", "")
            intent = item.get("command", "")
            entities = item.get("entities", [])
            values = item.get("values", [])
            
            # Intent distribution
            if intent:
                analysis["intent_distribution"][intent] = analysis["intent_distribution"].get(intent, 0) + 1
            
            # Text length statistics
            if text:
                length = len(text)
                total_length += length
                analysis["max_text_length"] = max(analysis["max_text_length"], length)
                analysis["min_text_length"] = min(analysis["min_text_length"], length)
            
            # Entity statistics
            if entities:
                analysis["examples_with_entities"] += 1
                for entity in entities:
                    if isinstance(entity, dict) and "label" in entity:
                        label = entity["label"]
                        analysis["entity_types"][label] = analysis["entity_types"].get(label, 0) + 1
            
            # Value statistics
            if values:
                analysis["examples_with_values"] += 1
                for value in values:
                    if isinstance(value, dict) and "label" in value:
                        label = value["label"]
                        analysis["value_types"][label] = analysis["value_types"].get(label, 0) + 1
        
        # Calculate average text length
        if dataset:
            analysis["avg_text_length"] = total_length / len(dataset)
        
        return analysis