#!/usr/bin/env python3
"""
API Server cho PhoBERT_SAM
Xử lý Intent Recognition và Entity Extraction
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import re
from datetime import datetime
import json
from config import model_config

# Pydantic models
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
    """Model đơn giản cho Intent Recognition"""
    
    def __init__(self, model_name, num_intents):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class PhoBERTAPI:
    """API class cho PhoBERT_SAM"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id_to_intent = None
        self.intent_to_command = None
        self.device = torch.device("cpu")
        
        # Intent to Command mapping
        self.intent_to_command = {
            "call": "make_call",
            "send-mess": "send_message", 
            "set-alarm": "set_alarm",
            "check-weather": "check_weather",
            "play-media": "play_media",
            "read-news": "read_news",
            "check-health-status": "check_health",
            "set-reminder": "set_reminder",
            "general-conversation": "chat"
        }
        
        # Entity patterns
        self.entity_patterns = {
            "RECEIVER": [
                r"cho\s+(\w+)",
                r"gọi\s+(\w+)", 
                r"nhắn\s+(\w+)",
                r"(\w+)\s+(?:ơi|à)",
                r"(?:bố|mẹ|ông|bà|anh|chị|em|con|cháu)"
            ],
            "TIME": [
                r"(\d{1,2}:\d{2})",
                r"(\d{1,2})\s*giờ",
                r"(\d{1,2})\s*phút",
                r"(sáng|chiều|tối|đêm)",
                r"(hôm\s+nay|ngày\s+mai|tuần\s+sau)"
            ],
            "MESSAGE": [
                r"rằng\s+(.+)",
                r"nói\s+(.+)",
                r"nhắn\s+(.+)",
                r"gửi\s+(.+)"
            ],
            "LOCATION": [
                r"ở\s+(\w+)",
                r"tại\s+(\w+)",
                r"(\w+)\s+(?:thành\s+phố|tỉnh|quận|huyện)"
            ]
        }
    
    def load_model(self):
        """Load model đã training"""
        try:
            print("🔄 Loading PhoBERT model...")
            
            # Load checkpoint
            checkpoint = torch.load("models/best_simple_intent_model.pth", map_location='cpu')
            
            # Tạo model
            self.model = SimpleIntentModel(model_config.model_name, len(checkpoint['intent_to_id']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            
            # Load mappings
            self.id_to_intent = checkpoint['id_to_intent']
            
            print(f"✅ Model loaded successfully!")
            print(f"   - Validation accuracy: {checkpoint['val_acc']:.2f}%")
            print(f"   - Number of intents: {len(checkpoint['intent_to_id'])}")
            print(f"   - Available intents: {list(checkpoint['intent_to_id'].keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities từ text"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if entity_type == "MESSAGE":
                        # Lấy toàn bộ message
                        entities[entity_type] = matches[0].strip()
                    else:
                        # Lấy entity đầu tiên tìm được
                        entities[entity_type] = matches[0].strip()
                    break
        
        return entities
    
    def generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        """Generate value dựa trên intent và entities"""
        if intent == "unknown" or intent == "error":
            return "Không thể xác định"
        
        # Tạo value dựa trên intent và entities
        if intent == "call":
            receiver = entities.get("RECEIVER", "người nhận")
            return f"Gọi điện cho {receiver}"
        
        elif intent == "send-mess":
            receiver = entities.get("RECEIVER", "người nhận")
            message = entities.get("MESSAGE", "tin nhắn")
            return f"Gửi tin nhắn cho {receiver}: {message}"
        
        elif intent == "set-alarm":
            time_info = entities.get("TIME", "thời gian")
            return f"Đặt báo thức lúc {time_info}"
        
        elif intent == "set-reminder":
            time_info = entities.get("TIME", "thời gian")
            return f"Đặt nhắc nhở lúc {time_info}"
        
        elif intent == "check-weather":
            location = entities.get("LOCATION", "khu vực hiện tại")
            return f"Kiểm tra thời tiết tại {location}"
        
        elif intent == "play-media":
            return "Phát nhạc/phim"
        
        elif intent == "read-news":
            return "Đọc tin tức"
        
        elif intent == "check-health-status":
            return "Kiểm tra tình trạng sức khỏe"
        
        elif intent == "general-conversation":
            return "Trò chuyện thông thường"
        
        else:
            return f"Thực hiện hành động: {intent}"
    
    def predict_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Predict intent và confidence"""
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=model_config.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                confidence = probabilities.max().item()
                intent = self.id_to_intent[predicted.item()]
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                intent = "unknown"
                confidence = 0.0
            
            return {
                "intent": intent,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            print(f"❌ Error predicting intent: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "probabilities": []
            }
    
    def process_text(self, text: str, confidence_threshold: float = 0.3) -> IntentResponse:
        """Xử lý text và trả về kết quả đầy đủ"""
        start_time = datetime.now()
        
        # Predict intent
        intent_result = self.predict_intent(text, confidence_threshold)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Get command
        command = self.intent_to_command.get(intent_result["intent"], "unknown")
        
        # Generate value based on intent and entities
        value = self.generate_value(intent_result["intent"], entities, text)
        
        # Calculate processing time
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

# Initialize API
api = PhoBERTAPI()

# FastAPI app
app = FastAPI(
    title="PhoBERT_SAM API",
    description="API cho Intent Recognition và Entity Extraction cho người cao tuổi",
    version="1.0.0"
)

# Add CORS middleware
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

@app.post("/predict", response_model=IntentResponse)
async def predict_intent(request: IntentRequest):
    """Predict intent và extract entities"""
    try:
        if not api.model:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        result = api.process_text(request.text, request.confidence_threshold)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

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
            result = api.process_text(text, confidence_threshold)
            results.append(result.dict())
        
        return {
            "results": results,
            "total_processed": len(texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý batch: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting PhoBERT_SAM API Server...")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🎯 Predict Endpoint: POST http://localhost:8000/predict")
    print("=" * 50)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
