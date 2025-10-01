import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import re
from datetime import datetime
import json
import os
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import NLP processor with absolute import
try:
    from src.inference.engines.nlp_processor import NLPProcessor
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.inference.engines.nlp_processor import NLPProcessor

try:
    from src.training.configs.config import model_config
    from src.inference.engines.reasoning_engine import ReasoningEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("🔧 Trying alternative import...")
    
    # Alternative import path
    sys.path.insert(0, str(project_root / "src"))
    from training.configs.config import model_config
    from inference.engines.reasoning_engine import ReasoningEngine

class IntentRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.3

class IntentResponse(BaseModel):
    input_text: str
    intent: str
    confidence: float
    command: str
    entities: Dict[str, str]
    value: str
    processing_time: float
    timestamp: str

class SimpleIntentModel(nn.Module):
    """Model tối ưu cho Intent Recognition với Large model và GPU"""
    
    def __init__(self, model_name, num_intents, config):
        super().__init__()
        self.config = config
        
        self.phobert = AutoModel.from_pretrained(
            model_name,
            gradient_checkpointing=config.gradient_checkpointing
        )
        
        hidden_size = self.phobert.config.hidden_size
        
        if config.model_size == "large":
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 4),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size // 4, num_intents)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, num_intents)
            )
    
    def forward(self, input_ids, attention_mask):
        if self.config.gradient_checkpointing and self.training:
            outputs = self.phobert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False  # Tắt cache để tiết kiệm memory
            )
        else:
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_output = (sequence_output * attention_mask_expanded).sum(dim=1) / attention_mask_expanded.sum(dim=1)
        
        logits = self.classifier(pooled_output)
        return logits

class PhoBERT_SAM_API:
    """API class cho PhoBERT_SAM"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize NLP processor
        self.nlp_processor = NLPProcessor(self.device)
        
        # Legacy attributes for backward compatibility
        self.model = None
        self.tokenizer = None
        self.id_to_intent = None
        self.intent_to_command = self.nlp_processor.get_intent_to_command_mapping()
        
        # Legacy entity patterns đã được chuyển vào entity_extractor.py
        # và được cải thiện trong communication_optimizer.py
        
        self.reasoning_engine = ReasoningEngine()
        print("Reasoning Engine initialized")
    
    def _extract_entities_simple(self, text: str) -> dict:
        """Extract entities đơn giản và hiệu quả"""
        entities = {}
        text_lower = text.lower()
        
        # RECEIVER - Lấy người nhận
        receiver_match = re.search(r"cho\s+((?:ba|bố|mẹ|anh|chị|em|cô|chú|bác|ông|bà)\s*[\w\s]*?)(?:\s+(?:rằng|là|nói|nhắn|gửi|lúc|tại|ở|vào|ngày|giờ|$))", text_lower)
        if receiver_match:
            entities["RECEIVER"] = receiver_match.group(1).strip()
        
        # TIME - Lấy thời gian
        time_match = re.search(r"(sáng|trưa|chiều|tối|đêm|\d{1,2}h)", text_lower)
        if time_match:
            entities["TIME"] = time_match.group(1).strip()
        
        # LOCATION - Lấy địa điểm
        location_match = re.search(r"tại\s+((?:bệnh viện|trường|công viên|nhà|công ty|văn phòng|phòng)\s*[\w\s]*?)(?:\s+(?:lúc|giờ|vào|ngày|$))", text_lower)
        if location_match:
            entities["LOCATION"] = location_match.group(1).strip()
        
        # PLATFORM - Mặc định SMS
        platform_match = re.search(r"qua\s+(zalo|facebook|messenger|telegram|instagram|tiktok)", text_lower)
        if platform_match:
            entities["PLATFORM"] = platform_match.group(1).lower()
        else:
            entities["PLATFORM"] = "sms"
        
        # MESSAGE - Lấy nội dung tin nhắn đầy đủ (bao gồm cả phần sau dấu phẩy)
        message_patterns = [
            r"là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$)",  # "là" + toàn bộ nội dung đến cuối câu
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$)",  # "rằng" + toàn bộ nội dung đến cuối câu
            r"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$)"   # "nói" + toàn bộ nội dung đến cuối câu
        ]
        
        for pattern in message_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                message = match.group(1).strip()
                if message and len(message) > 5:
                    entities["MESSAGE"] = message
                    break
        
        return entities
    
    def _predict_intent_simple(self, text: str) -> tuple:
        """Predict intent đơn giản và nhanh"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["nhắn", "tin", "gửi", "message", "sms"]):
            return "send-mess", 0.9
        elif any(word in text_lower for word in ["gọi", "điện", "phone", "call"]):
            return "call", 0.9
        elif any(word in text_lower for word in ["báo thức", "nhắc", "alarm", "reminder"]):
            return "set-alarm", 0.9
        else:
            return "unknown", 0.0
    
    def load_model(self):
        """Load trained model với hỗ trợ large model"""
        try:
                        
            # Load config
            from src.training.configs.config import ModelConfig
            model_config = ModelConfig()
            
            model_dir = f"models/phobert_{model_config.model_size}_intent_model"
            
            best_model_path = None
            if os.path.exists(model_dir):
                best_models = []
                for filename in os.listdir(model_dir):
                    if filename.endswith('.pth') and 'best' in filename:
                        file_path = f"{model_dir}/{filename}"
                        file_time = os.path.getmtime(file_path)
                        best_models.append((file_path, file_time))
                
                if best_models:
                    best_models.sort(key=lambda x: x[1], reverse=True)
                    best_model_path = best_models[0][0]
                    print(f"🎯 Found latest best model: {os.path.basename(best_model_path)}")
            
            if not best_model_path:
                model_path = f"{model_dir}/model.pth"
                if not os.path.exists(model_path):
                    model_path = "models/best_simple_intent_model.pth"
                    if not os.path.exists(model_path):
                        print(f"No model found, using reasoning engine only")
                        self.model = None
                        self.tokenizer = None
                        self.id_to_intent = None
                        return True
            else:
                model_path = best_model_path
            
            print(f"Loading model from: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'intent_to_id' not in checkpoint or 'id_to_intent' not in checkpoint:
                print("Model không có metadata, sử dụng reasoning engine only")
                self.model = None
                self.tokenizer = None
                self.id_to_intent = None
                return True
            
            # Tạo model với config
            self.model = SimpleIntentModel(model_config.model_name, len(checkpoint['intent_to_id']), model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            # Enable mixed precision nếu có GPU và config cho phép
            if self.device.type == "cuda" and model_config.use_fp16:
                self.model = self.model.half()  # Convert to FP16
                print("🔧 Enabled FP16 for GPU inference")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            
            self.id_to_intent = checkpoint['id_to_intent']
            
            model_info = {
                'model_size': checkpoint.get('model_size', 'unknown'),
                'total_parameters': checkpoint.get('total_parameters', 0),
                'trainable_parameters': checkpoint.get('trainable_parameters', 0),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'is_best': checkpoint.get('is_best', False),
                'validation_accuracy': checkpoint.get('validation_accuracy', 'unknown')
            }
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Number of intents: {len(self.id_to_intent)}")
            print(f"Model info: {model_info}")
            print(f"Available intents: {list(self.id_to_intent.values())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
            self.id_to_intent = None
            return True
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities using NLPProcessor"""
        return self.nlp_processor.entity_extractor.extract_all_entities(text)
    
    def generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        """Generate value using NLPProcessor"""
        return self.nlp_processor.value_generator.generate_value(intent, entities, original_text)
    
    def predict_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Predict intent using NLPProcessor"""
        return self.nlp_processor.intent_predictor.predict_intent(text, confidence_threshold)
    
    async def process_text(self, text: str, confidence_threshold: float = 0.3) -> IntentResponse:
        """Process text using NLPProcessor"""
        result = self.nlp_processor.process_text(text, confidence_threshold)
        
        return IntentResponse(
            input_text=result["input_text"],
            intent=result["intent"],
            confidence=result["confidence"],
            command=result["command"],
            entities=result["entities"],
            value=result["value"],
            processing_time=result["processing_time"],
            timestamp=result["timestamp"]
        )
    
    async def predict_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Predict with reasoning using NLPProcessor"""
        return await self.nlp_processor.process_with_reasoning(text)
    
    def load_model(self) -> bool:
        """Load model using NLPProcessor"""
        # Try to find model file - Updated paths to match actual model location
        model_paths = [
            "models/phobert_large_intent_model/model_epoch_10_best.pth",
            "models/phobert_large_intent_model/model_epoch_3_best.pth", 
            "models/phobert_large_intent_model/model_epoch_1_best.pth",
            "models/phobert_large_intent_model/model_best.pth"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                success = self.nlp_processor.load_model(model_path)
                if success:
                    # Update legacy attributes
                    self.model = self.nlp_processor.intent_predictor.model
                    self.tokenizer = self.nlp_processor.intent_predictor.tokenizer
                    self.id_to_intent = self.nlp_processor.intent_predictor.id_to_intent
                    print(f"Successfully loaded trained model from {model_path}")
                    return True
                else:
                    print(f"Failed to load model from {model_path}")
        
        print("No trained model found, using fallback methods")
        return True  # Return True to allow fallback methods
    
    # Legacy methods for backward compatibility
    def _extract_entities_simple(self, text: str) -> dict:
        """Legacy method - redirect to new extractor"""
        return self.extract_entities(text)
    
    def _predict_intent_simple(self, text: str) -> tuple:
        """Legacy method - redirect to new predictor"""
        result = self.predict_intent(text)
        return result["intent"], result["confidence"]
    
    
    
    # Old generate_value method removed - now using NLPProcessor
    def _old_generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        if intent == "unknown" or intent == "error":
            return "Không thể xác định"
        
        if intent in ["call", "call", "make-video-call"]:
            receiver = entities.get("RECEIVER", "")
            if not receiver:
                potential_receivers = re.findall(r"(?:gọi|gọi cho|gọi điện cho|nhắn tin cho|gửi cho)\s+(\w+(?:\s+\w+){0,2})", original_text, re.IGNORECASE)
                if potential_receivers:
                    receiver = potential_receivers[0]
                else:
                    receiver = "người nhận"
            
            if intent == "make-video-call":
                return f"Gọi video cho {receiver}"
            else:
                return f"Gọi điện cho {receiver}"
        
        elif intent in ["send-mess", "send-mess"]:
            message = entities.get("MESSAGE", "")
            receiver = entities.get("RECEIVER", "")
            
            if "kiểm tra" in original_text.lower() and "từ" in original_text.lower():
                match = re.search(r"từ\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if match:
                    from_person = match.group(1).strip()
                    return f"từ {from_person}"
            
            if message:
                if "rằng là" in message:
                    content = message.split("rằng là", 1)[-1].strip()
                    if content.startswith("là "):
                        content = content[3:].strip()
                    return content
                elif message.startswith("là "):
                    content = message[3:].strip()
                    return content
                elif "rằng" in message:
                    content = message.split("rằng", 1)[-1].strip()
                    if content.startswith("là "):
                        content = content[3:].strip()
                    return content
                elif " là " in message:
                    content = message.split(" là ", 1)[-1].strip()
                    return content
                else:
                    return message
            
            else:
                patterns = [
                    r"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhắn|gửi|nhắn tin|gửi tin nhắn)(?:\s+cho\s+\w+)?(?:\s+qua\s+\w+)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content.startswith("là "):
                            content = content[3:].strip()
                        return content
                
                if receiver:
                    return f"Tin nhắn cho {receiver}"
                else:
                    return "Nội dung tin nhắn"
        
        elif intent in ["set-alarm", "set-reminder"]:
            time_info = entities.get("TIME", "")
            
            if not time_info:
                time_patterns = [
                    r"(\d{1,2})\s*(?:giờ|h|:)\s*(\d{1,2})?\s*(?:phút)?",
                    r"(\d{1,2})\s*(?:giờ|h)\s*(?:rưỡi|buổi|sáng|trưa|chiều|tối)",
                    r"(\d{1,2})\s*(?:giờ|h)"
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        if match.group(2):  # Hour and minute
                            time_info = f"{match.group(1)}:{match.group(2)}"
                        else:  # Just hour
                            time_info = f"{match.group(1)}:00"
                        break
            
            if time_info:
                period_match = re.search(r"(sáng|trưa|chiều|tối|đêm)", original_text, re.IGNORECASE)
                if period_match and period_match.group(1) not in time_info:
                    time_info = f"{time_info} {period_match.group(1)}"
            
            if intent == "set-alarm":
                description = "Báo thức"
                description_match = re.search(r"(?:báo thức|alarm)(?:\s+để)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if description_match:
                    description = description_match.group(1).strip()
                
                return f"{time_info} - {description}" if time_info else "Thời gian báo thức"
            else:  # set-reminder
                description = "Nhắc nhở"
                
                medicine_patterns = [
                    r"uống\s+(\d+\s+)?(?:viên\s+)?(?:thuốc\s+)?(?:tiểu\s+đường|huyết\s+áp|tim|vitamin|sắt|cảm|đau\s+đầu|kháng\s+sinh)",
                    r"(?:thuốc\s+)?(?:tiểu\s+đường|huyết\s+áp|tim|vitamin|sắt|cảm|đau\s+đầu|kháng\s+sinh)",
                    r"uống\s+(.+?)(?:\s+(?:lúc|vào|sau|trước|nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhắc nhở|nhắc|reminder)(?:\s+về|về|về việc|việc)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in medicine_patterns:
                    description_match = re.search(pattern, original_text, re.IGNORECASE)
                    if description_match:
                        if description_match.groups():
                            description = description_match.group(1).strip()
                        else:
                            description = description_match.group(0).strip()
                        break
                
                return f"{time_info} - {description}" if time_info else description
        
        elif intent == "check-weather":
            location = entities.get("LOCATION", "")
            time = entities.get("TIME", "")
            
            if not location:
                location_match = re.search(r"(?:thời tiết|nhiệt độ|mưa|nắng)(?:\s+ở|tại|của)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if location_match:
                    location = location_match.group(1).strip()
                else:
                    location = "khu vực hiện tại"
            
            if time:
                return f"{location} ({time})"
            else:
                return location
        
        elif intent == "check-device-status":
            device = entities.get("DEVICE", "thiết bị")
            return f"Kiểm tra trạng thái {device}"
        
        elif intent == "check-health-status":
            health_aspect = entities.get("HEALTH", "sức khỏe")
            return f"Kiểm tra {health_aspect}"
        
        elif intent == "check-messages":
            platform = entities.get("PLATFORM", "")
            receiver = entities.get("RECEIVER", "")
            
            if "từ" in original_text.lower():
                match = re.search(r"từ\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if match:
                    from_person = match.group(1).strip()
                    return f"từ {from_person}"
            
            if receiver and receiver != "trên":  # "trên" không phải người gửi
                return f"từ {receiver}"
            elif platform:
                return f"Kiểm tra {platform}"
            else:
                return "Kiểm tra tin nhắn"
        
        elif intent in ["play-media", "play-audio", "play-content"]:
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            platform = entities.get("PLATFORM", "")
            
            if not content and not query:
                content_patterns = [
                    r"(?:phát|mở|bật|nghe|xem)(?:\s+bài|nhạc|phim|video|clip)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:bài hát|bài|ca khúc|nhạc|phim|video|clip)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm kiếm|tìm|search)\s+(?:trên|qua|bằng|dùng)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:danh sách|list|playlist)\s+(?:nhạc|music|video|clip|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|music|video|clip|phim)\s+(?:mới nhất|hot|trending|phổ biến)\s+(?:của|do|bởi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in content_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        break
            
            final_content = content if content else query
            
            if final_content:
                if platform:
                    return f"{final_content} trên {platform}"
                else:
                    return final_content
            else:
                return "Nội dung phát"
        
        elif intent in ["read-news", "read-content"]:
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            
            if not content and not query:
                content_match = re.search(r"(?:đọc|đọc tin|đọc báo|đọc tin tức)(?:\s+về|về)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if content_match:
                    content = content_match.group(1).strip()
            
            if intent == "read-news":
                if content:
                    return f"Tin tức về {content}"
                elif query:
                    return f"Tin tức về {query}"
                else:
                    return "Tin tức"
            else:  # read-content
                if content:
                    return f"Đọc: {content}"
                elif query:
                    return f"Đọc về: {query}"
                else:
                    return "Nội dung đọc"
        
        elif intent == "view-content":
            content = entities.get("CONTENT", "")
            query = entities.get("QUERY", "")
            
            if content:
                return f"Xem: {content}"
            elif query:
                return f"Xem về: {query}"
            else:
                return "Nội dung xem"
        
        elif intent == "open-app":
            app = entities.get("APP", "")
            
            if not app:
                app_match = re.search(r"(?:mở|vào|khởi động|chạy|sử dụng|dùng)(?:\s+ứng dụng|app|phần mềm)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if app_match:
                    app = app_match.group(1).strip()
                else:
                    app = "ứng dụng"
            
            return app
        
        elif intent == "open-app-action":
            app = entities.get("APP", "ứng dụng")
            action = entities.get("ACTION", "hành động")
            return f"{action} trong {app}"
        
        elif intent in ["search-content", "search-internet"]:
            query = entities.get("QUERY", "")
            platform = entities.get("PLATFORM", "")
            
            if not query:
                query_patterns = [
                    r"(?:tìm|tìm kiếm|search|tra cứu|tra|kiếm|tìm hiểu)(?:\s+về|về|thông tin về)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm|tìm kiếm|search|tra cứu|tra|kiếm|tìm hiểu)(?:\s+(?:cho tôi|cho mình|cho bác|cho cô|cho chú|giúp tôi|giúp mình|giúp bác|giúp cô|giúp chú))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search|phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:tìm kiếm|tìm|search)\s+(?:trên|qua|bằng|dùng)\s+(?:youtube|facebook|zalo|instagram|tiktok)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:danh sách|list|playlist)\s+(?:nhạc|music|video|clip|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|music|video|clip|phim)\s+(?:mới nhất|hot|trending|phổ biến)\s+(?:của|do|bởi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                
                for pattern in query_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        break
            
            if query:
                if platform:
                    return f"{query} trên {platform}"
                else:
                    return query
            else:
                return "Từ khóa tìm kiếm"
        
        elif intent == "browse-social-media":
            platform = entities.get("PLATFORM", "")
            
            if not platform:
                platform_match = re.search(r"(?:lướt|duyệt|xem|vào|mở)(?:\s+(?:facebook|fb|zalo|instagram|tiktok|youtube|twitter))(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if platform_match:
                    platform = platform_match.group(1).strip()
                else:
                    platform = "mạng xã hội"
            
            return f"Duyệt {platform}"
        
        elif intent == "control-device":
            device = entities.get("DEVICE", "thiết bị")
            action = entities.get("ACTION", "")
            
            if not action:
                action_match = re.search(r"(?:bật|tắt|mở|đóng|khóa|mở khóa|điều chỉnh|tăng|giảm|thay đổi)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if action_match:
                    action = action_match.group(0).strip()  # Use full match as the action
                else:
                    action = "điều khiển"
            
            return f"{action} {device}"
        
        elif intent == "adjust-settings":
            setting = entities.get("SETTING", "")
            
            if not setting:
                setting_match = re.search(r"(?:cài đặt|thiết lập|điều chỉnh|thay đổi|chỉnh|sửa)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if setting_match:
                    setting = setting_match.group(1).strip()
                else:
                    setting = "cài đặt"
            
            return f"Điều chỉnh {setting}"
        
        elif intent == "app-tutorial":
            app = entities.get("APP", "")
            
            if not app:
                app_match = re.search(r"(?:hướng dẫn|chỉ dẫn|chỉ|dạy|bày)(?:\s+(?:sử dụng|dùng|cách))?(?:\s+(?:ứng dụng|app|phần mềm))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if app_match:
                    app = app_match.group(1).strip()
                else:
                    app = "ứng dụng"
            
            return f"Hướng dẫn sử dụng {app}"
        
        elif intent == "navigation-help":
            destination = entities.get("LOCATION", "")
            
            if not destination:
                destination_match = re.search(r"(?:đường|đường đi|chỉ đường|chỉ|đi|tới|đến|về)(?:\s+(?:tới|đến|về))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if destination_match:
                    destination = destination_match.group(1).strip()
                else:
                    destination = "đích đến"
            
            return f"Điều hướng đến {destination}"
        
        elif intent == "provide-instructions":
            topic = entities.get("TOPIC", "")
            
            if not topic:
                topic_match = re.search(r"(?:hướng dẫn|chỉ dẫn|chỉ|dạy|bày)(?:\s+(?:về|cách))?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])", original_text, re.IGNORECASE)
                if topic_match:
                    topic = topic_match.group(1).strip()
                else:
                    topic = "chủ đề"
            
            return f"Hướng dẫn về {topic}"
        
        elif intent == "general-conversation":
            if "xin chào" in original_text.lower() or "hello" in original_text.lower() or "hi" in original_text.lower():
                return "Chào hỏi"
            elif "tạm biệt" in original_text.lower() or "bye" in original_text.lower():
                return "Tạm biệt"
            elif "cảm ơn" in original_text.lower() or "thanks" in original_text.lower() or "thank" in original_text.lower():
                return "Cảm ơn"
            elif "xin lỗi" in original_text.lower() or "sorry" in original_text.lower():
                return "Xin lỗi"
            elif "khỏe không" in original_text.lower() or "thế nào" in original_text.lower():
                return "Hỏi thăm"
            else:
                return "Trò chuyện thông thường"
        
        else:
            if "tìm" in original_text.lower() or "tìm kiếm" in original_text.lower() or "search" in original_text.lower():
                search_patterns = [
                    r"(?:tìm|tìm kiếm|search)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:vào|mở)\s+\w+\s+(?:để|để mà|mà)\s+(?:tìm|tìm kiếm|search)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in search_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            elif "phát" in original_text.lower() or "nghe" in original_text.lower() or "xem" in original_text.lower():
                media_patterns = [
                    r"(?:phát|nghe|xem)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:nhạc|video|phim)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in media_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            elif "mở" in original_text.lower() or "vào" in original_text.lower():
                app_patterns = [
                    r"(?:mở|vào)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
                    r"(?:ứng dụng|app)\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])"
                ]
                for pattern in app_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            return f"Thực hiện hành động: {intent}"
    
    # Old predict_intent method removed - now using NLPProcessor
    def _old_predict_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Old predict intent method - replaced by NLPProcessor"""
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            if self.model.dtype == torch.float16:
                input_ids = input_ids.half()
                attention_mask = attention_mask.half()
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                confidence = probabilities.max().item()
                intent = self.id_to_intent[predicted.item()]
            
            if confidence < confidence_threshold:
                intent = "unknown"
                confidence = 0.0
            
            return {
                "intent": intent,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "probabilities": []
            }
    
    # Old predict_with_reasoning method removed - now using NLPProcessor
    async def _old_predict_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Old predict with reasoning method - replaced by NLPProcessor"""
        try:
            print(f"REASONING PREDICTION: '{text}'")
            start_time = datetime.now()
            
            try:
                model_result = None
                model_confidence = 0.0
                model_intent = "unknown"
                
                if self.model and self.tokenizer:
                    model_result = self.predict_intent(text)
                    model_confidence = model_result.get("confidence", 0.0)
                    model_intent = model_result.get("intent", "unknown")
                    print(f"MODEL PREDICTION: {model_intent} (confidence: {model_confidence:.3f})")
            except Exception as e:
                print(f"Model prediction error: {str(e)}")
                model_result = {"intent": "unknown", "confidence": 0.0}
                model_confidence = 0.0
                model_intent = "unknown"
            
            if model_confidence < 0.6 or model_intent == "unknown":
                                
                text_lower = text.lower()
                message_keywords = ["nhắn tin", "gửi tin", "soạn tin", "text", "sms", "message", "gửi", "nhắn"]
                has_message_keyword = any(keyword in text_lower for keyword in message_keywords)
                
                try:
                    reasoning_result = self.reasoning_engine.reasoning_predict(text)
                    reasoning_intent = reasoning_result.get("intent", "unknown")
                    reasoning_confidence = reasoning_result.get("confidence", 0.0)
                    print(f"REASONING PREDICTION: {reasoning_intent} (confidence: {reasoning_confidence:.3f})")
                except Exception as e:
                    print(f"Reasoning engine error: {e}")
                    print("Using simple prediction...")
                    reasoning_intent, reasoning_confidence = self._predict_intent_simple(text)
                    reasoning_result = {"intent": reasoning_intent, "confidence": reasoning_confidence}
                    print(f"SIMPLE PREDICTION: {reasoning_intent} (confidence: {reasoning_confidence:.3f})")
                
                call_keywords = ["cuộc gọi", "gọi thoại", "gọi điện", "thực hiện gọi", "thực hiện cuộc gọi"]
                has_call_keyword = any(keyword in text_lower for keyword in call_keywords)
                
                if has_call_keyword and reasoning_intent != "call":
                    print("🔧 Override intent to call due to call keywords")
                    reasoning_intent = "call"
                    reasoning_confidence = max(reasoning_confidence, 0.8)  # Boost confidence
                
                elif has_message_keyword and reasoning_intent != "send-mess":
                    print("🔧 Override intent to send-mess due to message keywords")
                    reasoning_intent = "send-mess"
                    reasoning_confidence = max(reasoning_confidence, 0.7)  # Boost confidence
                
                final_intent = reasoning_intent
                final_confidence = reasoning_confidence
                method = "reasoning_engine"
                
                if model_intent != "unknown" and model_confidence >= 0.4:
                    if model_intent == reasoning_intent:
                        final_confidence = max(model_confidence, reasoning_confidence) + 0.1
                        final_confidence = min(final_confidence, 0.99)  # Cap at 0.99
                        method = "model_reasoning_agreement"
                    else:
                        if model_confidence > reasoning_confidence + 0.2:  # Model significantly more confident
                            final_intent = model_intent
                            final_confidence = model_confidence
                            method = "model_override"
            else:
                final_intent = model_intent
                final_confidence = model_confidence
                method = "trained_model"
                reasoning_result = None
            
            try:
                # Sử dụng logic extract đơn giản và hiệu quả
                entities = self._extract_entities_simple(text)
                
            except Exception as e:
                print(f"Error extracting entities: {e}")
                entities = {}
            
            command = self.intent_to_command.get(final_intent, "unknown")
            
            try:
                value = self.generate_value(final_intent, entities, text)
            except Exception as e:
                print(f"Error generating value: {e}")
                value = f"Thực hiện hành động: {final_intent}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "text": text,
                "intent": final_intent,
                "confidence": final_confidence,
                "command": command,
                "entities": entities,
                "value": value,
                "method": method,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if reasoning_result:
                result["reasoning_details"] = {
                    "semantic_similarity": reasoning_result.get("semantic_similarity", {})
                }
            
            return result
                    
        except Exception as e:
            print(f"Error in reasoning prediction: {str(e)}")
            import traceback
            return {
                "text": text,
                "intent": "unknown",
                "confidence": 0.0,
                "command": "unknown",
                "entities": {},
                "value": "",
                "method": "error",
                "error": str(e),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def batch_predict_with_reasoning(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch predict với reasoning engine"""
        results = []
        for text in texts:
            result = await self.predict_with_reasoning(text)
            results.append(result)
        return results
    
    async def analyze_text_semantics(self, text: str) -> Dict[str, Any]:
        """Phân tích semantic của text"""
        try:
            embedding = self.reasoning_engine.get_text_embedding(text)
            
            similar_intents = self.reasoning_engine.find_similar_intents(text)
            
            context_features = self.reasoning_engine.extract_context_features(text)
            
            pattern_results = self.reasoning_engine.pattern_matching(text)
            
            keyword_results = self.reasoning_engine.keyword_matching(text)
            
            return {
                "text": text,
                "embedding_shape": embedding.shape,
                "similar_intents": similar_intents,
                "context_features": context_features,
                "pattern_matching": pattern_results,
                "keyword_matching": keyword_results,
                "semantic_analysis": {
                    "has_time_context": context_features["has_time"],
                    "has_person_context": context_features["has_person"],
                    "has_action_context": context_features["has_action"],
                    "has_object_context": context_features["has_object"]
                }
            }
            
        except Exception as e:
            return {
                "text": text,
                "error": str(e)
            }
    
    async def update_knowledge_base(self, new_patterns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Cập nhật knowledge base của reasoning engine"""
        try:
            self.reasoning_engine.update_knowledge_base(new_patterns)
            return {
                "status": "success",
                "message": "Knowledge base updated successfully",
                "updated_patterns": new_patterns
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def save_knowledge_base(self, file_path: str = "knowledge_base.json") -> Dict[str, Any]:
        """Lưu knowledge base"""
        try:
            self.reasoning_engine.save_knowledge_base(file_path)
            return {
                "status": "success",
                "message": f"Knowledge base saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def load_knowledge_base(self, file_path: str = "knowledge_base.json") -> Dict[str, Any]:
        """Load knowledge base"""
        try:
            self.reasoning_engine.load_knowledge_base(file_path)
            return {
                "status": "success",
                "message": f"Knowledge base loaded from {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    # Old process_text method removed - now using NLPProcessor
    async def _old_process_text(self, text: str, confidence_threshold: float = 0.3) -> IntentResponse:
        """Old process text method - replaced by NLPProcessor"""
        start_time = datetime.now()
        
        intent_result = self.predict_intent(text, confidence_threshold)
        
        entities = self.extract_entities(text)
        
        command = self.intent_to_command.get(intent_result["intent"], "unknown")
        
        value = self.generate_value(intent_result["intent"], entities, text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntentResponse(
            input_text=text,
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            command=command,
            entities=entities,
            value=value,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

api = PhoBERT_SAM_API()

app = FastAPI(
    title="PhoBERT_SAM API",
    description="API cho Intent Recognition và Entity Extraction cho người cao tuổi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (có thể thay đổi thành specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả HTTP methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

@app.on_event("startup")
async def startup_event():
    """Khởi tạo model khi server start"""
    if not api.load_model():
        raise Exception("Không thể load model!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PhoBERT_SAM API",
        "version": "1.0.0",
        "status": "running",
        "available_intents": list(api.intent_to_command.keys()) if api.id_to_intent else []
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": api.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_intent(request: IntentRequest):
    """Predict intent và extract entities - Simplified response"""
    try:
        if not api.model:
            reasoning_result = await api.predict_with_reasoning(request.text)
            
            return {
                "text": request.text,
                "intent": reasoning_result["intent"],
                "confidence": reasoning_result["confidence"],
                "command": reasoning_result["command"],
                "entities": reasoning_result["entities"],
                "value": reasoning_result["value"],
                "method": reasoning_result.get("method", "reasoning_engine"),
                "processing_time": reasoning_result["processing_time"],
                "timestamp": reasoning_result["timestamp"],
                "reasoning_details": reasoning_result.get("reasoning_details", {})
            }
        
        result = await api.process_text(request.text, request.confidence_threshold)
        
        return {
            "text": result.input_text,
            "intent": result.intent,
            "confidence": result.confidence,
            "command": result.command,
            "entities": result.entities,
            "value": result.value,
            "method": "trained_model",
            "processing_time": result.processing_time,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        
        # Return simplified error response
        return {
            "text": request.text,
            "intent": "unknown",
            "confidence": 0.0,
            "command": "unknown",
            "entities": {},
            "value": "",
            "method": "error",
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/intents")
async def get_intents():
    """Lấy danh sách intents có sẵn"""
    if not api.id_to_intent:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    return {
        "intents": list(api.id_to_intent.values()),
        "intent_to_command": api.intent_to_command
    }

@app.get("/entities")
async def get_entity_patterns():
    """Lấy patterns cho entity extraction"""
    return {
        "entity_patterns": api.entity_patterns
    }

@app.post("/batch_predict")
async def batch_predict(texts: List[str], confidence_threshold: float = 0.3):
    """Predict nhiều texts cùng lúc"""
    try:
        if not api.model:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        results = []
        for text in texts:
            result = await api.process_text(text, confidence_threshold)
            results.append(result.dict())
        
        return {
            "results": results,
            "total_processed": len(texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý batch: {str(e)}")

@app.post("/predict-with-reasoning")
async def predict_with_reasoning(request: Dict[str, Any]):
    """Predict intent với reasoning engine"""
    text = request.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    result = await api.predict_with_reasoning(text)
    return result

@app.post("/batch-predict-with-reasoning")
async def batch_predict_with_reasoning(request: Dict[str, Any]):
    """Batch predict với reasoning engine"""
    texts = request.get("texts", [])
    if not texts:
        return {"error": "Texts list is required"}
    
    results = await api.batch_predict_with_reasoning(texts)
    return {"results": results}

@app.post("/analyze-semantics")
async def analyze_semantics(request: Dict[str, Any]):
    """Phân tích semantic của text"""
    text = request.get("text", "")
    if not text:
        return {"error": "Text is required"}
    
    result = await api.analyze_text_semantics(text)
    return result

@app.post("/update-knowledge-base")
async def update_knowledge_base(request: Dict[str, Any]):
    """Cập nhật knowledge base"""
    new_patterns = request.get("patterns", {})
    if not new_patterns:
        return {"error": "Patterns are required"}
    
    result = await api.update_knowledge_base(new_patterns)
    return result

@app.post("/save-knowledge-base")
async def save_knowledge_base(request: Dict[str, Any]):
    """Lưu knowledge base"""
    file_path = request.get("file_path", "knowledge_base.json")
    result = await api.save_knowledge_base(file_path)
    return result

@app.post("/load-knowledge-base")
async def load_knowledge_base(request: Dict[str, Any]):
    """Load knowledge base"""
    file_path = request.get("file_path", "knowledge_base.json")
    result = await api.load_knowledge_base(file_path)
    return result

if __name__ == "__main__":
    print("🚀 Starting PhoBERT_SAM API Server...")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎯 Predict Endpoint: POST http://localhost:8000/predict")
    print("=" * 50)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
