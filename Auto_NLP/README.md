# PhoBERT_SAM - Hệ Thống NLP Cho Người Cao Tuổi

Hệ thống Intent Recognition và Entity Extraction sử dụng PhoBERT cho người cao tuổi, hỗ trợ nhận diện ý định và trích xuất thông tin từ câu nói tiếng Việt.

## 🚀 Tính Năng Chính

- **Intent Recognition**: Nhận diện ý định người dùng (gọi điện, gửi tin nhắn, đặt báo thức, v.v.)
- **Entity Extraction**: Trích xuất thông tin quan trọng (người nhận, thời gian, tin nhắn, địa điểm)
- **Command Mapping**: Chuyển đổi intent thành command thực thi
- **Value Generation**: Tạo mô tả hành động cụ thể
- **Web Interface**: Giao diện web đẹp mắt để test API
- **RESTful API**: API hoàn chỉnh với FastAPI

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- RAM: 8GB+ (khuyến nghị 16GB)
- Disk: 5GB+ trống
- OS: Windows/Linux/macOS

## 🛠️ Cài Đặt

### 1. Clone Repository
```bash
git clone <repository-url>
cd Auto_NLP
```

### 2. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Kiểm Tra Cài Đặt
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

## 🎯 Sử Dụng Nhanh

### Bước 1: Khởi động API Server
```bash
python api_server.py
```
Server sẽ chạy tại: http://localhost:8000

### Bước 2: Mở Giao Diện Web
- Mở file `web_interface.html` trong trình duyệt
- Hoặc truy cập: http://localhost:8000/docs (API documentation)

### Bước 3: Test Hệ Thống
Nhập các câu mẫu:
- "nhắc tôi lúc 5 giờ chiều"
- "alo cho bố"
- "gửi tin nhắn cho mẹ rằng tối con sẽ về ăn cơm"
- "đặt báo thức lúc 7 giờ sáng"

## 🔧 Training Model (Nếu Cần)

### 1. Chuẩn Bị Dataset
Hệ thống đã có sẵn dataset:
- `elderly_command_dataset_reduced.json`: 1,951 samples
- `nlp_command_dataset.json`: 3,062 samples

### 2. Data Augmentation (Tùy chọn)
```bash
python data_augmentation.py
```

### 3. Training Model
```bash
python simple_train.py
```

**Lưu ý**: Training có thể mất 30-60 phút tùy cấu hình máy.

## 📁 Cấu Trúc Dự Án

```
Auto_NLP/
├── api_server.py              # FastAPI server chính
├── web_interface.html         # Giao diện web
├── simple_train.py            # Script training đơn giản
├── config.py                  # Cấu hình hệ thống
├── data_augmentation.py       # Tăng cường dữ liệu
├── utils.py                   # Tiện ích
├── main.py                    # Training pipeline đầy đủ
├── requirements.txt           # Dependencies
├── README.md                  # Hướng dẫn này
├── TRAINING_GUIDE.md          # Hướng dẫn training chi tiết
├── models/                    # Thư mục chứa model
│   ├── best_simple_intent_model.pth  # Model đã train
│   ├── intent_model.py        # Model architecture
│   ├── entity_model.py        # Entity extraction model
│   ├── command_model.py       # Command processing model
│   └── unified_model.py       # Unified model
├── data/                      # Data processing
│   └── data_processor.py      # Data processing utilities
├── training/                  # Training utilities
│   └── trainer.py             # Training class
├── logs/                      # Training logs
└── datasets/                  # Dataset files
    ├── elderly_command_dataset_reduced.json
    └── nlp_command_dataset.json
```

## 🔌 API Endpoints

### Chính
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Predict intent và extract entities
- `GET /intents` - Lấy danh sách intents
- `GET /entities` - Lấy entity patterns
- `POST /batch_predict` - Predict nhiều texts

### Request Format
```json
{
  "text": "gửi tin nhắn cho mẹ rằng tối con sẽ về ăn cơm",
  "confidence_threshold": 0.3
}
```

### Response Format
```json
{
  "input_text": "gửi tin nhắn cho mẹ rằng tối con sẽ về ăn cơm",
  "intent": "send-mess",
  "confidence": 0.643,
  "command": "send_message",
  "entities": {
    "RECEIVER": "mẹ",
    "MESSAGE": "tối con sẽ về ăn cơm"
  },
  "value": "Gửi tin nhắn cho mẹ: tối con sẽ về ăn cơm",
  "processing_time": 0.123,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🎯 Intent Types

| Intent | Command | Mô Tả |
|--------|---------|-------|
| `call` | `make_call` | Gọi điện thoại |
| `send-mess` | `send_message` | Gửi tin nhắn |
| `set-alarm` | `set_alarm` | Đặt báo thức |
| `set-reminder` | `set_reminder` | Đặt nhắc nhở |
| `check-weather` | `check_weather` | Kiểm tra thời tiết |
| `play-media` | `play_media` | Phát nhạc/phim |
| `read-news` | `read_news` | Đọc tin tức |
| `check-health-status` | `check_health` | Kiểm tra sức khỏe |
| `general-conversation` | `chat` | Trò chuyện thông thường |

## 🔍 Entity Types

- **RECEIVER**: Người nhận (bố, mẹ, anh, chị, v.v.)
- **TIME**: Thời gian (5 giờ, 7:30, sáng, chiều, v.v.)
- **MESSAGE**: Nội dung tin nhắn
- **LOCATION**: Địa điểm

## ⚙️ Cấu Hình

### Model Config (`config.py`)
```python
model_size: "base"              # "base" hoặc "large"
max_length: 128                 # Độ dài tối đa input
batch_size: 8                   # Batch size cho training
learning_rate: 1e-5             # Learning rate
num_epochs: 15                  # Số epochs training
```

### Training Config
```python
device: "cpu"                   # "cpu" hoặc "cuda"
confidence_threshold: 0.3       # Ngưỡng tin cậy
```

## 🚨 Troubleshooting

### Lỗi Thường Gặp

1. **"Model not found"**
   - Chạy `python simple_train.py` để train model
   - Đảm bảo file `models/best_simple_intent_model.pth` tồn tại

2. **"CUDA out of memory"**
   - Giảm `batch_size` trong `config.py`
   - Sử dụng CPU training: `device: "cpu"`

3. **"API connection failed"**
   - Kiểm tra server đã chạy: `python api_server.py`
   - Kiểm tra port 8000 không bị chiếm

4. **"Import error"**
   - Cài đặt lại dependencies: `pip install -r requirements.txt`
   - Kiểm tra Python version >= 3.8

### Performance Tips

- **CPU Training**: Giảm batch_size, tăng epochs
- **Memory Issues**: Giảm max_length, batch_size
- **Speed**: Sử dụng GPU nếu có

## 📊 Performance

- **Model Size**: PhoBERT-base (500MB)
- **Training Time**: 30-60 phút (CPU)
- **Inference Time**: ~0.1-0.3s/câu
- **Accuracy**: ~74% (validation)

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🙏 Acknowledgments

- PhoBERT model từ VINAI
- FastAPI framework
- Transformers library từ Hugging Face

---

**Lưu ý**: Đây là hệ thống demo, model đã được train sẵn và sẵn sàng sử dụng. Nếu cần train lại, hãy làm theo hướng dẫn trong phần Training Model.
