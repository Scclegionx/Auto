# Auto_NLP - Vietnamese NLP System

## Cài Đặt

### 1. Requirements
```bash
pip install torch transformers numpy scikit-learn tqdm datasets
```

## Cấu Hình

### Config Files
- **`src/training/configs/config.py`** - Config mặc định (tiết kiệm tài nguyên)
- **`src/training/configs/config_optimal.py`** - Config tối ưu (GPU khỏe)

### Tham Số Quan Trọng
```python
# Trong ModelConfig
self.num_epochs = 5              # Số epochs
self.batch_size = 16             # Batch size
self.learning_rate = 2e-5        # Learning rate
self.max_length = 256            # Độ dài input
self.dropout = 0.2               # Dropout rate
self.freeze_layers = 6           # Layers đóng băng
```

### Điều Chỉnh Config
```python
# File: src/training/configs/config.py
# Dòng 28: Thay đổi epochs
self.num_epochs = 10

# Dòng 20: Thay đổi batch size
self.batch_size = 32

# Dòng 22: Thay đổi learning rate  
self.learning_rate = 3e-5
```

## Training

### Chạy Training
```bash
# Cách 1: Script tiện ích
python run_training.py

# Cách 2: Trực tiếp
python src/training/scripts/train_gpu.py
```

### Config Phù Hợp
- **GPU yếu/RAM ít**: Dùng `config.py` (mặc định)
- **GPU khỏe/RAM nhiều**: Dùng `config_optimal.py`

## API Server

### Chạy API
```bash
# Cách 1: Script tiện ích
python run_api.py

# Cách 2: Trực tiếp
python src/inference/api/main.py
```

### Sử Dụng API
```bash
# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "gọi cho mẹ tôi"}'
```

## Data Processing

### Xử Lý Dataset
```bash
# Process data
python src/data/processed/data_processor.py

# Augment data
python src/data/augmented/data_augmentation.py
```

## Troubleshooting

### Lỗi Import
```bash
# Sửa imports
python fix_imports.py
```

### Lỗi Memory
- Giảm `batch_size` trong config
- Bật `gradient_checkpointing = True`
- Giảm `max_length`

### Lỗi GPU
- Kiểm tra CUDA: `torch.cuda.is_available()`
- Giảm `batch_size` nếu GPU yếu
- Sử dụng `config.py` thay vì `config_optimal.py`
