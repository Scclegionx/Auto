"""
Hybrid Intent Predictor
Kết hợp trained model với reasoning engine để tạo ra kết quả tốt nhất
"""

import torch
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference.engines.intent_predictor import IntentPredictor

class HybridIntentPredictor:
    """Hybrid predictor kết hợp trained model và reasoning engine"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.trained_predictor = IntentPredictor(device)
        self.reasoning_engine = self._init_reasoning_engine()
        
        # Confidence thresholds
        self.model_threshold = 0.3  # Threshold for trained model
        self.reasoning_threshold = 0.6  # Threshold for reasoning engine
        
        # Intent mapping
        self.intent_mapping = {
            'send-mess': 0, 'call': 1, 'set-alarm': 2, 'set-event-calendar': 3,
            'get-info': 4, 'add-contacts': 5, 'control-device': 6, 'make-video-call': 7,
            'open-cam': 8, 'play-media': 9, 'search-internet': 10, 'view-content': 11,
            'search-youtube': 12
        }
        self.id_to_intent = {v: k for k, v in self.intent_mapping.items()}
    
    def _init_reasoning_engine(self):
        """Initialize reasoning engine với keyword patterns"""
        return {
            'call': [
                r'gọi\s+cho', r'gọi\s+điện', r'gọi\s+điện\s+thoại', r'liên\s+lạc',
                r'gọi\s+video', r'gọi\s+face\s+time', r'gọi\s+skype'
            ],
            'send-mess': [
                r'gửi\s+tin', r'nhắn\s+tin', r'gửi\s+tin\s+nhắn', r'nhắn\s+cho',
                r'gửi\s+message', r'gửi\s+thông\s+báo'
            ],
            'set-alarm': [
                r'báo\s+thức', r'đặt\s+báo\s+thức', r'báo\s+thức\s+lúc', r'nhắc\s+nhở',
                r'đặt\s+giờ', r'báo\s+thức\s+giờ'
            ],
            'set-event-calendar': [
                r'tạo\s+sự\s+kiện', r'thêm\s+lịch', r'đặt\s+lịch', r'tạo\s+lịch',
                r'sự\s+kiện', r'lịch\s+hẹn', r'cuộc\s+họp'
            ],
            'get-info': [
                r'hỏi\s+thông\s+tin', r'tìm\s+hiểu', r'cho\s+biết', r'hỏi\s+về',
                r'thông\s+tin\s+về', r'giá\s+cả', r'thời\s+tiết'
            ],
            'add-contacts': [
                r'thêm\s+liên\s+hệ', r'thêm\s+danh\s+bạ', r'lưu\s+số', r'thêm\s+số',
                r'tạo\s+liên\s+hệ', r'lưu\s+liên\s+lạc'
            ],
            'control-device': [
                r'bật\s+điều\s+hòa', r'tắt\s+điều\s+hòa', r'bật\s+quạt', r'tắt\s+quạt',
                r'điều\s+chỉnh', r'kiểm\s+soát\s+thiết\s+bị', r'bật\s+thiết\s+bị'
            ],
            'make-video-call': [
                r'gọi\s+video', r'gọi\s+face\s+time', r'gọi\s+skype', r'video\s+call',
                r'gọi\s+hình', r'gọi\s+video\s+call'
            ],
            'open-cam': [
                r'mở\s+camera', r'chụp\s+ảnh', r'quay\s+video', r'mở\s+máy\s+ảnh',
                r'camera', r'chụp\s+hình'
            ],
            'play-media': [
                r'phát\s+nhạc', r'bật\s+nhạc', r'nghe\s+nhạc', r'phát\s+video',
                r'bật\s+video', r'phát\s+podcast', r'nghe\s+podcast'
            ],
            'search-internet': [
                r'tìm\s+kiếm', r'tìm\s+trên\s+mạng', r'search', r'tìm\s+thông\s+tin',
                r'tìm\s+hiểu\s+về', r'tra\s+cứu'
            ],
            'view-content': [
                r'xem\s+tin\s+tức', r'đọc\s+báo', r'xem\s+báo', r'xem\s+thông\s+tin',
                r'xem\s+nội\s+dung', r'đọc\s+nội\s+dung'
            ],
            'search-youtube': [
                r'tìm\s+trên\s+youtube', r'youtube', r'tìm\s+video', r'tìm\s+trên\s+yt',
                r'video\s+youtube', r'yt'
            ]
        }
    
    def load_trained_model(self, model_path: str) -> bool:
        """Load trained model"""
        return self.trained_predictor.load_model(model_path)
    
    def predict_intent(self, text: str) -> Dict:
        """Hybrid intent prediction"""
        # Step 1: Try trained model first
        model_result = self._try_trained_model(text)
        
        # Step 2: Try reasoning engine
        reasoning_result = self._try_reasoning_engine(text)
        
        # Step 3: Combine results
        final_result = self._combine_results(model_result, reasoning_result, text)
        
        return final_result
    
    def _try_trained_model(self, text: str) -> Optional[Dict]:
        """Try trained model prediction"""
        try:
            if self.trained_predictor.model is not None:
                result = self.trained_predictor.predict_intent(text)
                if result.get('method') == 'trained_model':
                    return result
        except Exception as e:
            print(f"Trained model error: {e}")
        return None
    
    def _try_reasoning_engine(self, text: str) -> Dict:
        """Try reasoning engine prediction"""
        text_lower = text.lower()
        
        # Calculate scores for each intent
        intent_scores = {}
        for intent, patterns in self.reasoning_engine.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            intent_scores[intent] = score
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
            
            # Calculate confidence based on score
            total_score = sum(intent_scores.values())
            confidence = best_score / max(total_score, 1)
            
            return {
                'intent': best_intent,
                'confidence': confidence,
                'method': 'reasoning_engine',
                'scores': intent_scores
            }
        
        return {
            'intent': 'unknown',
            'confidence': 0.0,
            'method': 'reasoning_engine'
        }
    
    def _combine_results(self, model_result: Optional[Dict], reasoning_result: Dict, text: str) -> Dict:
        """Combine results from both methods"""
        
        # If no model result, use reasoning
        if model_result is None:
            return reasoning_result
        
        model_confidence = model_result.get('confidence', 0.0)
        reasoning_confidence = reasoning_result.get('confidence', 0.0)
        
        # Decision logic
        if model_confidence >= self.model_threshold:
            # Model is confident enough
            if reasoning_confidence >= self.reasoning_threshold:
                # Both are confident - check if they agree
                if model_result['intent'] == reasoning_result['intent']:
                    # They agree - use model with boosted confidence
                    return {
                        'intent': model_result['intent'],
                        'confidence': min(model_confidence + 0.1, 1.0),
                        'method': 'hybrid_agreement',
                        'model_confidence': model_confidence,
                        'reasoning_confidence': reasoning_confidence
                    }
                else:
                    # They disagree - use model (trained model has priority)
                    return {
                        'intent': model_result['intent'],
                        'confidence': model_confidence,
                        'method': 'hybrid_model_priority',
                        'model_confidence': model_confidence,
                        'reasoning_confidence': reasoning_confidence,
                        'reasoning_intent': reasoning_result['intent']
                    }
            else:
                # Only model is confident
                return {
                    'intent': model_result['intent'],
                    'confidence': model_confidence,
                    'method': 'trained_model_only',
                    'model_confidence': model_confidence,
                    'reasoning_confidence': reasoning_confidence
                }
        else:
            # Model is not confident enough
            if reasoning_confidence >= self.reasoning_threshold:
                # Use reasoning engine
                return {
                    'intent': reasoning_result['intent'],
                    'confidence': reasoning_confidence,
                    'method': 'reasoning_engine_only',
                    'model_confidence': model_confidence,
                    'reasoning_confidence': reasoning_confidence
                }
            else:
                # Neither is confident - use reasoning as fallback
                return {
                    'intent': reasoning_result['intent'],
                    'confidence': reasoning_confidence,
                    'method': 'hybrid_fallback',
                    'model_confidence': model_confidence,
                    'reasoning_confidence': reasoning_confidence
                }
    
    def get_prediction_info(self) -> Dict:
        """Get information about the hybrid system"""
        return {
            'model_loaded': self.trained_predictor.model is not None,
            'model_threshold': self.model_threshold,
            'reasoning_threshold': self.reasoning_threshold,
            'available_intents': list(self.intent_mapping.keys()),
            'reasoning_patterns': len(self.reasoning_engine)
        }

def main():
    """Test hybrid system"""
    # Set UTF-8 encoding for Windows
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    print("TESTING HYBRID INTENT PREDICTOR")
    print("=" * 40)
    
    # Create hybrid predictor
    predictor = HybridIntentPredictor()
    
    # Load trained model
    model_path = "src/training/scripts/models/phobert_large_intent_model/model_epoch_3.pth"
    if os.path.exists(model_path):
        print(f"Loading trained model: {model_path}")
        success = predictor.load_trained_model(model_path)
        print(f"Model loaded: {success}")
    else:
        print("Trained model not found, using reasoning engine only")
    
    # Test cases
    test_cases = [
        "Gọi cho Nguyễn Văn A",
        "Gửi tin nhắn cho bạn",
        "Báo thức lúc 7 giờ",
        "Tạo sự kiện họp nhóm",
        "Hỏi thời tiết hôm nay",
        "Thêm liên hệ mới",
        "Bật điều hòa",
        "Gọi video cho bạn",
        "Mở camera",
        "Phát nhạc",
        "Tìm kiếm thông tin",
        "Xem tin tức",
        "Tìm video trên YouTube"
    ]
    
    print(f"\nTesting {len(test_cases)} cases...")
    
    for i, text in enumerate(test_cases, 1):
        result = predictor.predict_intent(text)
        print(f"\nTest {i}: '{text}'")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result['method']}")
        if 'model_confidence' in result:
            print(f"  Model conf: {result['model_confidence']:.3f}")
        if 'reasoning_confidence' in result:
            print(f"  Reasoning conf: {result['reasoning_confidence']:.3f}")
    
    # System info
    info = predictor.get_prediction_info()
    print(f"\nSystem info: {info}")

if __name__ == "__main__":
    main()
