# 🚀 Hướng dẫn Setup Hoàn Chỉnh cho Máy Mới

## 📋 Luồng Setup Thống Nhất

### Bước 1: Clone dự án
```bash
git clone <repository-url>
cd Auto_NLP
```

### Bước 2: Tạo Virtual Environment
```bash
# Tạo virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Bước 3: Cài đặt Dependencies
```bash
# Cài đặt packages cần thiết
pip install torch>=2.5.0
pip install transformers>=4.20.0
pip install scikit-learn>=1.0.0
pip install seqeval>=1.2.0
pip install tqdm>=4.60.0
pip install numpy>=1.21.0
pip install regex>=2021.0.0
pip install fastapi>=0.70.0
pip install uvicorn>=0.15.0
pip install pydantic>=2.0.0
```

### Bước 4: Tải Model
```bash
# Tải PhoBERT-large model
python -c "
from transformers import AutoTokenizer, AutoModel
print('Tải PhoBERT-large...')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')
model = AutoModel.from_pretrained('vinai/phobert-large')
print('✅ Model đã tải xong')
"
```

### Bước 5: Cấu hình IDE
1. **VS Code**: Chọn Python interpreter từ virtual environment
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Chọn `venv\Scripts\python.exe`

2. **PyCharm**: Cấu hình Project Interpreter
   - File → Settings → Project → Python Interpreter
   - Add → Existing Environment → Chọn `venv\Scripts\python.exe`

### Bước 6: Kiểm tra Setup
```bash
# Test imports
python -c "
import torch
import transformers
import fastapi
import pydantic
print('✅ Tất cả imports OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### Bước 7: Chạy Training hoặc API
```bash
# Training
python src/training/scripts/train_gpu.py

# API Server
python api/server.py
```

## 🔧 Troubleshooting

### Lỗi Import trong IDE
- **Nguyên nhân**: IDE không sử dụng đúng Python interpreter
- **Giải pháp**: Chọn interpreter từ virtual environment

### Lỗi Model Loading
- **Nguyên nhân**: Model chưa được tải
- **Giải pháp**: Chạy Bước 4 để tải model

### Lỗi CUDA
- **Nguyên nhân**: Không có GPU hoặc CUDA
- **Giải pháp**: Training sẽ chạy trên CPU (chậm hơn)

### Lỗi Dependencies
- **Nguyên nhân**: Package version không tương thích
- **Giải pháp**: Cài đặt lại với version cụ thể

## 📊 Requirements

### Hardware
- **RAM**: 8GB+ (khuyến nghị 16GB+)
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị 6GB+ VRAM)
- **Storage**: 5GB+ free space

### Software
- **Python**: 3.8+ (khuyến nghị 3.9+)
- **CUDA**: 11.0+ (nếu có GPU)

## ✅ Checklist Setup

- [ ] Dự án được clone
- [ ] Virtual environment được tạo và activate
- [ ] Tất cả dependencies được cài đặt
- [ ] PhoBERT-large model được tải
- [ ] IDE được cấu hình đúng interpreter
- [ ] Test imports thành công
- [ ] Training script chạy được
- [ ] API server chạy được

## 🎯 Quick Start Commands

```bash
# Setup nhanh (copy-paste)
git clone <repository-url> && cd Auto_NLP
python -m venv venv && venv\Scripts\activate
pip install torch transformers scikit-learn seqeval tqdm numpy regex fastapi uvicorn pydantic
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large'); AutoModel.from_pretrained('vinai/phobert-large')"
python src/training/scripts/train_gpu.py
```

Sau khi setup xong, dự án sẽ sẵn sàng để training và chạy API! 🎉
