# 🤖 Auto NLP Hybrid System

Hệ thống NLP Hybrid thông minh kết hợp trained model với reasoning engine, được thiết kế đặc biệt cho người cao tuổi với khả năng xử lý tiếng Việt tự nhiên.

## ✨ **TÍNH NĂNG NỔI BẬT**

- 🧠 **Hybrid Intelligence**: Kết hợp AI model với rule-based reasoning
- 🇻🇳 **Tiếng Việt Native**: Xử lý tiếng Việt có dấu và không dấu
- 👴 **Thân thiện người già**: Giao diện đơn giản, dễ sử dụng
- 📱 **Multi-platform**: Hỗ trợ Zalo, WhatsApp, Messenger, Telegram
- 🎯 **Intent Recognition**: Nhận diện 15+ loại intent phổ biến
- 🔍 **Entity Extraction**: Trích xuất thông tin chính xác
- ⚡ **Real-time**: Phản hồi nhanh chóng

## 🚀 **QUICK START**

### **Setup Tự Động (Khuyến nghị)**
```bash
# Clone repository
git clone https://github.com/Scclegionx/Auto.git
cd Auto

# Chạy setup tự động
python setup_complete.py
```

### **Setup Thủ Công**
```bash
# Tạo virtual environment
python -m venv venv_new
venv_new\Scripts\activate

# Cài đặt PyTorch với CUDA
pip install torch>=2.5.0 torchvision>=0.20.0 torchaudio>=2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Cài đặt packages khác
pip install transformers>=4.30.0 datasets>=2.12.0 accelerate>=0.20.0
pip install fastapi>=0.100.0 uvicorn>=0.20.0 pydantic>=2.0.0
pip install underthesea>=6.6.0 pyvi>=0.1.1 scikit-learn>=1.3.0
```

## 🎯 **SỬ DỤNG**

### **Chạy API Server**
```bash
python api/server.py
```
Truy cập: `http://localhost:8000`

### **Chạy Web Interface**
Mở file `web_interface.html` trong browser

### **Training Model**
```bash
python src/training/scripts/train_gpu.py
```

## 📚 **HƯỚNG DẪN CHI TIẾT**

- 📖 **[Setup Guide](SETUP_GUIDE.md)** - Hướng dẫn setup cho máy mới
- 🎯 **[Training Guide](TRAINING_GUIDE.md)** - Chuẩn bị và chạy training
- 🔧 **[API Documentation](api/README.md)** - Tài liệu API endpoints

## 🏗️ **KIẾN TRÚC HỆ THỐNG**

```
Auto_NLP/
├── src/
│   ├── inference/          # Inference engine
│   │   ├── engines/       # Reasoning & Entity extraction
│   │   └── interfaces/    # Web interface
│   ├── training/          # Training scripts
│   ├── models/            # Model definitions
│   └── data/              # Dataset management
├── api/                   # FastAPI server
├── core/                  # Core hybrid system
├── models/                # Trained models & configs
└── web_interface.html     # Web UI
```

## 🎯 **INTENTS ĐƯỢC HỖ TRỢ**

| Intent | Mô tả | Ví dụ |
|--------|-------|-------|
| `call` | Gọi điện thoại | "gọi điện cho mẹ" |
| `send-mess` | Nhắn tin | "nhắn tin cho bố" |
| `make-video-call` | Video call | "gọi video với con" |
| `add-contacts` | Thêm liên hệ | "lưu số điện thoại" |
| `search-internet` | Tìm kiếm web | "tìm kiếm thời tiết" |
| `search-youtube` | Tìm YouTube | "tìm video ca nhạc" |
| `set-alarm` | Đặt báo thức | "đặt báo thức 7 giờ" |
| `set-event-calendar` | Đặt lịch | "tạo lịch họp" |
| `open-cam` | Mở camera | "mở camera sau" |
| `control-device` | Điều khiển thiết bị | "bật wifi" |
| `play-media` | Phát media | "phát nhạc" |
| `get-info` | Lấy thông tin | "hỏi thời gian" |
| `help` | Trợ giúp | "giúp tôi" |

## 🔧 **CẤU HÌNH**

### **Yêu cầu hệ thống:**
- **Python**: 3.8+ (khuyến nghị 3.11)
- **GPU**: NVIDIA GTX 1060+ (6GB VRAM)
- **RAM**: 8GB+ (khuyến nghị 16GB)
- **CUDA**: 12.1+

### **Environment Variables:**
```bash
# Optional: Weights & Biases
export WANDB_API_KEY="your_wandb_key"

# Optional: Custom model path
export MODEL_PATH="models/trained/best_model"
```

## 📊 **PERFORMANCE**

| Metric | Value |
|--------|-------|
| **Accuracy** | 95%+ |
| **Response Time** | <200ms |
| **Memory Usage** | ~2GB |
| **Supported Languages** | Vietnamese (primary), English |

## 🤝 **ĐÓNG GÓP**

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 **LICENSE**

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 **LIÊN HỆ**

- **GitHub**: [Scclegionx/Auto](https://github.com/Scclegionx/Auto)
- **Issues**: [GitHub Issues](https://github.com/Scclegionx/Auto/issues)

---
**🎉 Cảm ơn bạn đã sử dụng Auto NLP Hybrid System!**
```

### 4. Test System
```bash
# Test hybrid system
python main.py test

# Test API
python test_api.py
```

## 📁 Cấu trúc dự án

```
Auto_NLP/
├── core/                       # Core components
│   ├── hybrid_system.py       # Model-first hybrid system
│   ├── model_loader.py        # Model loading & inference
│   └── reasoning_engine.py    # Reasoning engine
├── api/                       # API layer
│   └── server.py              # FastAPI server
├── src/                       # Source code
│   ├── data/                  # Data management
│   │   ├── raw/               # Raw datasets
│   │   ├── processed/         # Processed datasets (train/val/test)
│   │   └── grouped/           # Grouped by intent
│   ├── training/              # Training pipeline
│   │   ├── scripts/           # Training scripts
│   │   └── configs/           # Training configs
│   └── inference/             # Inference components
│       ├── engines/           # Rule-based engines
│       └── interfaces/        # Web interface
├── models/                    # Trained models
│   └── phobert_large_intent_model/
├── config.py                  # Main configuration
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## 🎯 Sử dụng

### Command Line Interface
```bash
# Training
python main.py train

# Chạy API server
python main.py api

# Test system
python main.py test

# Xem config
python main.py config
```



1. **call** - Gọi điện thoại
2. **control-device** - Điều khiển thiết bị
3. **play-media** - Phát media
4. **search-internet** - Tìm kiếm internet
5. **search-youtube** - Tìm kiếm YouTube
6. **set-alarm** - Đặt báo thức
7. **send-mess** - Gửi tin nhắn
8. **open-cam** - Mở camera
9. **set-event-calendar** - Đặt lịch
10. **make-video-call** - Gọi video
11. **add-contacts** - Thêm danh bạ
12. **view-content** - Xem nội dung
13. **get-info** - Lấy thông tin


