# Auto NLP Hybrid System

Hệ thống NLP Hybrid kết hợp trained model với reasoning engine cho người cao tuổi.

## 🚀 Quick Start

### Luồng Clone và Setup (Thống Nhất)
```bash
# 1. Clone repository
git clone <repository-url>
cd Auto_NLP

# 2. Setup tự động (khuyến nghị)
python setup_new_machine.py

# 3. Chạy training
python src/training/scripts/train_gpu.py
```

📋 **Xem hướng dẫn chi tiết**: [CLONE_SETUP_GUIDE.md](CLONE_SETUP_GUIDE.md)

### 2. Training Model
```bash

python src/training/scripts/train_gpu.py

# Hoặc sử dụng main.py
python main.py train
```

### 3. Chạy API Server
```bash
# Chạy API server
python main.py api

# Hoặc trực tiếp
python api/server.py
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


