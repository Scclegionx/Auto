

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Kiểm tra cài đặt
```bash
python -c "import torch; import transformers; print('Cài đặt thành công!')"
```

## Huấn luyện mô hình

### 1. Chuẩn bị dữ liệu
Hệ thống đã có sẵn dataset `nlp_command_dataset.json` với 3062 mẫu dữ liệu tiếng Việt.

### 2. Huấn luyện tất cả mô hình
```bash
python main.py --mode train
```

Quá trình này sẽ huấn luyện:
- Intent Recognition Model (5 epochs)
- Entity Extraction Model (5 epochs)  
- Command Processing Model (5 epochs)
- Unified Model (5 epochs)

### 3. Kiểm tra kết quả
Sau khi huấn luyện, các file model sẽ được lưu trong thư mục `models/`:
- `best_intent_model.pth` (~515MB)
- `best_entity_model.pth` (~515MB)
- `best_command_model.pth` (~515MB)
- `best_unified_model.pth` (~515MB)

**Lưu ý**: Các file model lớn (>100MB) đã được loại trừ khỏi Git repository. Xem `MODELS_SETUP.md` để biết cách setup models sau khi clone.

## Khởi động API Server

### 1. Khởi động server
```bash
python api_server.py
```

Server sẽ chạy tại: `http://localhost:5000`

### 2. Kiểm tra trạng thái
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-12T12:49:43.292",
  "models_loaded": true,
  "mode": "production"
}
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Prediction (Chính)
```bash
POST /predict
Content-Type: application/json

{
  "text": "gửi tin nhắn cho mẹ: yêu mẹ nhiều"
}
```

## Sử dụng Web Interface

### 1. Mở giao diện web
Sau khi khởi động API server, mở file `static/index.html` trong trình duyệt hoặc truy cập:
```
http://localhost:5000/static/index.html
```

## Cấu hình

### Model Configuration (config.py)
```python
# PhoBERT model
model_name = "vinai/phobert-base"
max_length = 128
batch_size = 8
learning_rate = 2e-5
num_epochs = 5

# Intent labels (17 loại)
intent_labels = [
    "call", "check-health-status", "check-weather", 
    "express-emotion", "express-fatigue", "find-information",
    "general-conversation", "general-request", "play-media",
    "read-news", "report-symptom", "request-comfort",
    "request-entertainment", "request-instruction", 
    "send-mess", "set-alarm", "set-reminder"
]

# Command labels (18 loại)
command_labels = [
    "make_call", "check_health_status", "check_weather",
    "express_emotion", "express_fatigue", "find_information",
    "general_conversation", "general_request", "play_media",
    "read_news", "report_symptom", "request_comfort",
    "request_entertainment", "request_instruction",
    "send_message", "set_alarm", "set_reminder", "unknown"
]
```

## Performance

### Training Results
- Intent Recognition: ~74% accuracy
- Entity Extraction: ~96% F1-score
- Command Processing: ~70% accuracy
- Unified Model: Combined performance

### Inference Speed
- CPU: ~2-3 seconds per prediction
- GPU: ~0.5-1 second per prediction



## Cấu trúc thư mục

```
Auto_NLP/
├── api_server.py              # Flask API server
├── main.py                    # Training script
├── inference.py               # Inference engine
├── config.py                  # Configuration
├── utils.py                   # Utilities
├── requirements.txt           # Dependencies
├── nlp_command_dataset.json   # Training dataset
├── models/                    # Trained models
│   ├── best_intent_model.pth
│   ├── best_entity_model.pth
│   ├── best_command_model.pth
│   └── best_unified_model.pth
├── training/                  # Training modules
│   ├── __init__.py
│   └── trainer.py
├── data/                      # Data processing
│   ├── __init__.py
│   └── data_processor.py
├── static/                    # Web interface
│   └── index.html
└── logs/                      # Training logs
```
