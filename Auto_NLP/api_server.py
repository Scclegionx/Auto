from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import logging
from datetime import datetime
import os
import traceback

# Import các modules của hệ thống
try:
    from inference import PhoBERTSAMInference
    from data import DataProcessor
    from models import create_model
    import config
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODELS_AVAILABLE = False

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('official_api.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)

# Global variables
inference_engine = None
data_processor = None
models_loaded = False

def load_models():
    global inference_engine, data_processor, models_loaded
    
    if not MODELS_AVAILABLE:
        logging.warning("Models not available - using mock mode")
        return False
    
    try:
        model_paths = [
            './models/best_unified_model.pth',
            './models/best_intent_model.pth',
            './models/best_entity_model.pth',
            './models/best_command_model.pth'
        ]
        
        models_exist = any(os.path.exists(path) for path in model_paths)
        
        if not models_exist:
            logging.warning("No trained models found - using mock mode")
            return False
        
        # Khởi tạo inference engine
        inference_engine = PhoBERTSAMInference()
        data_processor = DataProcessor()
        
        models_loaded = True
        logging.info("✅ Models loaded successfully")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error loading models: {e}")
        logging.warning("Falling back to mock mode")
        return False

def mock_predict(text):
    """Mock prediction khi models chưa sẵn sàng"""
    # Phân tích đơn giản dựa trên keywords
    text_lower = text.lower()
    
    # Intent detection
    intent_keywords = {
        'send-mess': ['gửi', 'tin nhắn', 'nhắn tin', 'sms'],
        'set-alarm': ['đặt', 'báo thức', 'alarm', 'thức dậy'],
        'call': ['gọi', 'điện', 'phone', 'liên lạc'],
        'check-weather': ['thời tiết', 'weather', 'mưa', 'nắng'],
        'play-media': ['phát', 'nhạc', 'video', 'play', 'music'],
        'check-health-status': ['sức khỏe', 'health', 'bệnh', 'đau'],
        'read-news': ['tin tức', 'news', 'đọc', 'báo'],
        'set-reminder': ['nhắc nhở', 'reminder', 'nhớ', 'lịch']
    }
    
    detected_intent = 'general-conversation'  # default
    for intent, keywords in intent_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_intent = intent
            break
    
    # Command mapping
    command_mapping = {
        'send-mess': 'send_message',
        'set-alarm': 'set_alarm',
        'call': 'make_call',
        'check-weather': 'check_weather',
        'play-media': 'play_media',
        'check-health-status': 'check_health',
        'read-news': 'read_news',
        'set-reminder': 'set_reminder',
        'general-conversation': 'unknown'
    }
    
    detected_command = command_mapping.get(detected_intent, 'unknown')
    
    # Entity extraction (đơn giản)
    entities = []
    if 'mẹ' in text or 'bố' in text or 'anh' in text or 'chị' in text:
        entities.append({
            "text": "mẹ" if "mẹ" in text else "bố" if "bố" in text else "anh" if "anh" in text else "chị",
            "label": "RECEIVER"
        })
    
    if any(word in text for word in ['5 giờ', '6 giờ', '7 giờ', '8 giờ', '9 giờ', '10 giờ']):
        time_match = next(word for word in ['5 giờ', '6 giờ', '7 giờ', '8 giờ', '9 giờ', '10 giờ'] if word in text)
        entities.append({
            "text": time_match,
            "label": "TIME"
        })
    
    if ':' in text and len(text.split(':')) > 1:
        message_part = text.split(':')[1].strip()
        if message_part:
            entities.append({
                "text": message_part,
                "label": "MESSAGE"
            })
    
    return {
        "input": text,
        "intent": detected_intent,
        "command": detected_command,
        "entities": entities,
        "confidence": {
            "intent": 0.7 if detected_intent != 'general-conversation' else 0.3,
            "command": 0.7 if detected_command != 'unknown' else 0.3
        },
        "timestamp": datetime.now().isoformat(),
        "model_version": "PhoBERT_SAM_v1.0",
        "mode": "mock"
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "mode": "production" if models_loaded else "mock"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint dự đoán chính"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Thực hiện dự đoán
        if models_loaded and inference_engine:
            result = inference_engine.predict(text)
            result['mode'] = 'production'
        else:
            result = mock_predict(text)
        
        # Thêm metadata
        result['timestamp'] = datetime.now().isoformat()
        result['model_version'] = 'PhoBERT_SAM_v1.0'
        
        logging.info(f"Prediction for: '{text}' -> Intent: {result['intent']}, Command: {result['command']}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Phân tích chi tiết text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Phân tích cơ bản
        analysis = {
            "text": text,
            "text_length": len(text),
            "word_count": len(text.split()),
            "characters": len(text),
            "timestamp": datetime.now().isoformat()
        }
        
        # Thêm tokenization nếu có data processor
        if data_processor:
            try:
                tokens = data_processor.tokenizer.tokenize(text)
                analysis['tokens'] = tokens
                analysis['token_count'] = len(tokens)
            except:
                analysis['tokens'] = text.split()
                analysis['token_count'] = len(text.split())
        else:
            analysis['tokens'] = text.split()
            analysis['token_count'] = len(text.split())
        
        return jsonify(analysis)
        
    except Exception as e:
        logging.error(f"Error in text analysis: {e}")
        return jsonify({
            "error": f"Text analysis failed: {str(e)}"
        }), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Thông tin về hệ thống"""
    return jsonify({
        "system": "PhoBERT_SAM - Vietnamese NLP System",
        "version": "1.0.0",
        "description": "Intent Recognition, Entity Extraction, Command Processing",
        "status": "production" if models_loaded else "mock",
        "models_loaded": models_loaded,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analyze": "/analyze",
            "info": "/info"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/test', methods=['POST'])
def test_prediction():
    """Test endpoint với các mẫu"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                "error": "'texts' must be a list"
            }), 400
        
        results = []
        for text in texts:
            if models_loaded and inference_engine:
                result = inference_engine.predict(text)
                result['mode'] = 'production'
            else:
                result = mock_predict(text)
            results.append(result)
        
        response = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "total_texts": len(texts),
            "mode": "production" if models_loaded else "mock"
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in test prediction: {e}")
        return jsonify({
            "error": f"Test prediction failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Load models khi khởi động
    load_models()
    
    print("🚀 Starting Official PhoBERT_SAM API Server...")
    print("📡 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Single text prediction")
    print("  POST /analyze - Text analysis")
    print("  POST /test - Batch test prediction")
    print("  GET  /info - System information")
    print(f"\n🌐 Server will start on http://localhost:5000")
    print(f"📊 Mode: {'Production' if models_loaded else 'Mock'}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
