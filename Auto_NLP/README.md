## Cài Đặt

### 1. Requirements
```bash
pip install torch transformers numpy scikit-learn tqdm datasets
```

## Cấu Hình

### Config Files
- **`src/training/configs/config.py`** - Config mặc định (tiết kiệm tài nguyên)
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
## API Server

### Chạy API
```bash
# Cách 1: Script tiện ích
python run_api.py

# Cách 2: Trực tiếp
python src/inference/api/api_server.py
# start src/inference/interfaces/web_interface.html (giao diện)
```
