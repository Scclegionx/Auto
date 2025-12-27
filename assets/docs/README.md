
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





