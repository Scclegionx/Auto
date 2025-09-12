## Cài Đặt

### 1. Tạo virtual environment
```bash
python -m venv .venv
```

### 2. Kích hoạt virtual environment
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Kiểm tra hệ thống (tùy chọn)
```bash
# Test toàn diện
python test_training_complete.py

# Test cấu hình tối ưu
python test_optimal_config.py
```

## Training

### Chạy Training
```bash
# Cách 1: Script tiện ích (khuyến nghị)
python run_training.py

# Cách 2: Trực tiếp
python src/training/scripts/train_gpu.py
```

### Thông tin Training
- **Model**: PhoBERT-Large (vinai/phobert-large)
- **Dataset**: 1000 samples tiếng Việt
- **Intents**: 28 loại intent
- **Device**: Auto-detect (CPU/GPU)
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 3e-5

### Lưu ý
- Training sẽ tự động lưu model tại `models/phobert_large_intent_model/`
- Có thể dừng training bằng Ctrl+C
- Model sẽ được lưu sau mỗi epoch

## API Server

### Chạy API
```bash
# Cách 1: Script tiện ích (khuyến nghị)
python run_api.py

# Cách 2: Trực tiếp
python src/inference/api/api_server.py
```

### Sử dụng API
- **URL**: http://localhost:8000
- **Web Interface**: Mở `src/inference/interfaces/web_interface.html` trong browser
- **API Docs**: http://localhost:8000/docs (Swagger UI)

### Endpoints chính
- `POST /predict` - Dự đoán intent từ text
- `GET /health` - Kiểm tra trạng thái API
- `GET /intents` - Danh sách các intent được hỗ trợ

## Cấu trúc Dự án

```
Auto_NLP/
├── src/
│   ├── data/           # Dataset và xử lý dữ liệu
│   ├── models/         # Mô hình AI
│   ├── training/       # Scripts training
│   ├── inference/      # API và engines
│   └── utils/          # Tiện ích
├── models/             # Model đã train
├── logs/               # Log files
├── requirements.txt    # Dependencies
├── run_training.py     # Script training
└── run_api.py         # Script API
```

## Yêu cầu Hệ thống

- **Python**: 3.8+
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Storage**: Tối thiểu 5GB cho model và cache
- **GPU**: Tùy chọn (hỗ trợ CUDA)
