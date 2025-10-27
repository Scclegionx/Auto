# Auto NLP Hybrid System

Hệ thống NLP Hybrid kết hợp trained model với reasoning engine cho người cao tuổi.

## 🚀 Quick Start

### Setup Tự Động (Khuyến nghị)
```bash
# Clone repository
git clone <repository-url>
cd Auto_NLP

# Chạy setup tự động
python setup_complete.py
```

### Setup Thủ Công (Nếu cần)
```bash
# Clone repository
git clone <repository-url>
cd Auto_NLP

# Tạo virtual environment
python -m venv venv_new
venv_new\Scripts\activate

# Cài đặt dependencies
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.20.0 scikit-learn>=1.0.0 seqeval>=1.2.0 tqdm>=4.60.0 numpy>=1.21.0 regex>=2021.0.0 fastapi>=0.70.0 uvicorn>=0.15.0 pydantic>=2.0.0

# Tải PhoBERT-large model
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large', force_download=True); AutoModel.from_pretrained('vinai/phobert-large', force_download=True)"
```

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


