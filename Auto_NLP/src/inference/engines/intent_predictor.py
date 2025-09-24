"""
Intent Predictor Module
Xử lý dự đoán intent từ text input
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional
import re

class IntentPredictor:
    """Module dự đoán intent từ text"""
    
    def __init__(self, device=None):
        self.model = None
        self.tokenizer = None
        self.id_to_intent = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str, tokenizer_name: str = "vinai/phobert-base"):
        """Load trained model và tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model config
            num_intents = checkpoint.get('num_intents', 10)
            model_name = checkpoint.get('model_name', tokenizer_name)
            
            # Create model
            self.model = SimpleIntentModel(model_name, num_intents, checkpoint.get('config', {}))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load intent mapping
            self.id_to_intent = checkpoint.get('id_to_intent', {})
            
            print(f"✅ Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Dự đoán intent từ text"""
        if not self.model or not self.tokenizer:
            return self._fallback_intent_prediction(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                logits = self.model(inputs["input_ids"], inputs["attention_mask"])
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_id = torch.max(probabilities, dim=1)
                
                confidence = confidence.item()
                predicted_id = predicted_id.item()
                
                intent = self.id_to_intent.get(predicted_id, "unknown")
                
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "method": "trained_model"
                }
                
        except Exception as e:
            print(f"❌ Error in intent prediction: {e}")
            return self._fallback_intent_prediction(text)
    
    def _fallback_intent_prediction(self, text: str) -> Dict:
        """Fallback intent prediction khi model không available - Cải thiện cho CALL"""
        text_lower = text.lower()
        
        # Simple keyword-based intent detection - Cải thiện cho "nói chuyện điện thoại"
        intent_keywords = {
            "CALL": ["gọi", "alo", "gọi điện", "gọi thoại", "nói chuyện điện thoại", "nói chuyện", "trò chuyện", "liên lạc"],
            "MESSAGE": ["nhắn tin", "gửi tin", "soạn tin", "sms", "nhắn", "gửi"],
            "REMINDER": ["nhắc", "nhắc nhở", "reminder", "đừng quên"],
            "ALARM": ["báo thức", "đánh thức", "alarm", "dậy"],
            "SEARCH": ["tìm", "tìm kiếm", "search", "google"],
            "open-app": ["mở", "khởi động", "chạy"],
            "play-media": ["phát", "chơi", "nghe", "xem"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return {
                    "intent": intent,
                    "confidence": 0.7,
                    "method": "keyword_fallback"
                }
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "method": "fallback"
        }

class SimpleIntentModel(nn.Module):
    """Simple intent classification model"""
    
    def __init__(self, model_name, num_intents, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
