# Hướng dẫn Setup cho Máy Mới

## 🚀 Các bước setup dự án Auto_NLP trên máy mới

### 1. Clone dự án
```bash
git clone <repository-url>
cd Auto_NLP
```

### 2. Kiểm tra môi trường
```bash
python check_training_issues.py
```

### 3. Setup tự động (nếu cần)
```bash
python setup_new_machine.py
```

### 4. Chạy training
```bash
python src/training/scripts/train_gpu.py
```

## 🔧 Các vấn đề đã được sửa

### 1. Import Issues
- **Vấn đề**: `import regex` có thể gây lỗi trên một số hệ thống
- **Giải pháp**: Đã sửa thành `import re as regex`

### 2. Đường dẫn Model Cache
- **Vấn đề**: Đường dẫn `../../model_cache` có thể không đúng trên máy khác
- **Giải pháp**: Đã sửa thành `model_cache` (relative path)

### 3. Dependencies
- **Vấn đề**: Một số package có thể thiếu
- **Giải pháp**: Script `setup_new_machine.py` sẽ tự động cài đặt

## 📋 Requirements cho máy mới

### Python Version
- Python 3.8+ (khuyến nghị 3.9+)

### Required Packages
```
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
seqeval>=1.2.0
tqdm>=4.60.0
numpy>=1.21.0
regex>=2021.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
```

### Hardware Requirements
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị 6GB+ VRAM)
- **RAM**: 8GB+ RAM
- **Storage**: 5GB+ free space

## 🐛 Troubleshooting

### Lỗi Import
```bash
# Nếu thiếu package
pip install <package-name>

# Nếu lỗi regex
pip install regex
```

### Lỗi Model Loading
```bash
# Nếu thiếu model cache
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large'); AutoModel.from_pretrained('vinai/phobert-large')"
```

### Lỗi CUDA
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu không có CUDA, training sẽ chạy trên CPU
```

### Lỗi Data Files
```bash
# Kiểm tra data files
python -c "import json; print(len(json.load(open('src/data/processed/train.json'))))"
```

## 📊 Kiểm tra sau khi setup

### 1. Kiểm tra training script
```bash
python -c "
import sys
sys.path.append('src/training/scripts')
from train_gpu import GPUTrainer
print('✅ Training script OK')
"
```

### 2. Kiểm tra config
```bash
python -c "
import sys
sys.path.append('src/training/configs')
from config import ModelConfig
config = ModelConfig()
print(f'✅ Config OK: {config.model_name}')
"
```

### 3. Kiểm tra data
```bash
python -c "
import json
with open('src/data/processed/train.json') as f:
    data = json.load(f)
print(f'✅ Data OK: {len(data)} samples')
"
```

## 🎯 Chạy Training

### Training cơ bản
```bash
python src/training/scripts/train_gpu.py
```

### Training với custom config
```bash
# Chỉnh sửa src/training/configs/config.py
# Sau đó chạy
python src/training/scripts/train_gpu.py
```

### Monitoring Training
- Logs sẽ được lưu trong file `training_*.log`
- Model checkpoints sẽ được lưu trong `models/phobert_large_intent_model/`

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Chạy `python check_training_issues.py` để kiểm tra
2. Xem log file để debug
3. Kiểm tra requirements và dependencies
4. Đảm bảo có đủ tài nguyên hệ thống

## ✅ Checklist Setup

- [ ] Python 3.8+ installed
- [ ] All required packages installed
- [ ] Model cache downloaded
- [ ] Data files present
- [ ] Training script runs without import errors
- [ ] GPU/CUDA available (optional)
- [ ] Sufficient disk space
- [ ] Sufficient RAM

Sau khi hoàn thành checklist này, dự án sẽ sẵn sàng để training!
