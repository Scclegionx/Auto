
### 1. Cài Đặt Dependencies
```bash
# Clone repository
git clone <repository-url>
cd Auto_NLP

# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra GPU (nếu có)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Khởi Động API Server
```bash
python api_server.py
```
Server sẽ chạy tại: **http://localhost:8000**

### 3. Sử Dụng Hệ Thống

#### A. Qua Giao Diện Web
- Mở file `web_interface.html` trong trình duyệt

### 1. Chuẩn Bị Dataset
Hệ thống đã có sẵn dataset:
- `elderly_command_dataset_reduced.json` - Dataset cơ bản
- `nlp_command_dataset.json` - Dataset mở rộng

### 2. Training với PhoBERT-Large (Khuyến Nghị)
```bash
# Bước 1: Mở rộng dataset (Tùy chọn - dataset đã có sẵn)
python expand_dataset.py

# Bước 2: Training với PhoBERT-Large trên GPU
python train_gpu.py

### 3. Training với PhoBERT-Base (Tùy chọn)
```bash
# Training đơn giản với Base model
python simple_train.py
```



## ⚙️ Cấu Hình Hệ Thống

### Model Config (`config.py`)
```python
# PhoBERT-Large Configuration (Khuyến nghị)
model_size: "large"             # "base" hoặc "large"
max_length: 256                 # Độ dài tối đa input
batch_size: 8                   # Batch size cho training
learning_rate: 2e-5             # Learning rate tối ưu
num_epochs: 15                  # Số epochs training
device: "auto"                  # Auto-detect GPU/CPU
use_fp16: True                  # Mixed precision cho GPU
gradient_checkpointing: True    # Tiết kiệm memory
```
### Lỗi Thường Gặp

#### 1. GPU Không Được Nhận Diện
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Cài đặt lại PyTorch với CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

