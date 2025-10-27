# 🚀 Hướng Dẫn Clone và Setup Dự Án Auto_NLP

## 📋 Luồng Thống Nhất Khi Clone Dự Án

### Bước 1: Clone Dự Án
```bash
git clone <repository-url>
cd Auto_NLP
```

### Bước 2: Setup Tự Động (Khuyến nghị)
```bash
python setup_new_machine.py
```

### Bước 3: Kiểm Tra Setup
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### Bước 4: Chạy Training
```bash
python src/training/scripts/train_gpu.py
```

---

## 🔍 Chi Tiết Từng Bước

### Bước 1: Clone Dự Án
```bash
# Clone repository
git clone https://github.com/Scclegionx/Auto.git
cd Auto/Auto_NLP

# Kiểm tra cấu trúc
ls -la
```

**Kết quả mong đợi:**
```
Auto_NLP/
├── src/
├── models/
├── api/
├── core/
├── setup_new_machine.py
├── SETUP_NEW_MACHINE.md
└── README.md
```

### Bước 2: Setup Tự Động
```bash
python setup_new_machine.py
```

**Script này sẽ:**
- ✅ Cài đặt Python packages cần thiết
- ✅ Kiểm tra model cache
- ✅ Tải PhoBERT-large model (nếu thiếu)
- ✅ Tạo các thư mục cần thiết
- ✅ Test model loading

**Output mong đợi:**
```
🚀 Setup cho máy mới...
📦 Cài đặt requirements...
✅ Đã cài torch>=2.5.0
✅ Đã cài transformers>=4.20.0
...
🔍 Kiểm tra model cache...
❌ model_cache/models--vinai--phobert-large: MISSING
🔄 Sẽ tự động tải model...
🤖 Tải PhoBERT-large model...
✅ Đã tải PhoBERT-large model
✅ Setup hoàn thành!
```

### Bước 3: Kiểm Tra Setup
```bash
# Kiểm tra PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Kiểm tra CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Kiểm tra model
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')
print('✅ Model OK')
"
```

**Kết quả mong đợi:**
```
PyTorch: 2.5.1+cu121
CUDA: True
✅ Model OK
```

### Bước 4: Chạy Training
```bash
python src/training/scripts/train_gpu.py
```

**Output mong đợi:**
```
Starting GPU-optimized training with PhoBERT
Model: vinai/phobert-large
Model size: large
Max length: 512
Batch size: 8
Learning rate: 0.00001
Epochs: 4
...
🎯 Initialized multi-task model with:
   - Intent labels: 13
   - Entity labels: 30
   - Command labels: 13
...
📅 Epoch 1/4
✅ Loaded train data: 8148 samples
✅ Loaded val data: 1018 samples
```

---

## 🐛 Troubleshooting

### Lỗi 1: Thiếu Python Packages
```bash
# Cài đặt thủ công
pip install torch>=2.5.0 transformers>=4.20.0 scikit-learn seqeval tqdm numpy regex fastapi uvicorn
```

### Lỗi 2: Model Cache Missing
```bash
# Tải model thủ công
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('vinai/phobert-large')
AutoModel.from_pretrained('vinai/phobert-large')
"
```

### Lỗi 3: CUDA Not Available
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, training sẽ chạy trên CPU (chậm hơn)
```

### Lỗi 4: Data Files Missing
```bash
# Kiểm tra data files
ls src/data/processed/
# Cần có: train.json, val.json, test.json
```

---

## 📊 Yêu Cầu Hệ Thống

### Tối Thiểu
- **Python**: 3.8+
- **RAM**: 8GB+
- **Storage**: 5GB+
- **OS**: Windows/Linux/macOS

### Khuyến Nghị
- **Python**: 3.9+
- **RAM**: 16GB+
- **GPU**: NVIDIA với CUDA support
- **VRAM**: 6GB+ (cho training)

---

## ✅ Checklist Setup

- [ ] Repository cloned successfully
- [ ] Python 3.8+ installed
- [ ] `setup_new_machine.py` runs without errors
- [ ] PyTorch 2.5.0+ installed
- [ ] PhoBERT-large model downloaded
- [ ] Data files present (train.json, val.json, test.json)
- [ ] CUDA available (optional)
- [ ] Training script runs without import errors

---

## 🎯 Sau Khi Setup Thành Công

### Chạy Training
```bash
python src/training/scripts/train_gpu.py
```

### Chạy API Server
```bash
python api/server.py
```

### Mở Web Interface
```bash
# Mở file trong browser
src/inference/interfaces/web_interface.html
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Chạy `python setup_new_machine.py` để tự động fix
2. Kiểm tra log files để debug
3. Đảm bảo có đủ tài nguyên hệ thống
4. Kiểm tra kết nối internet (để tải model)

**Luồng này đã được test và hoạt động ổn định!** 🎉
