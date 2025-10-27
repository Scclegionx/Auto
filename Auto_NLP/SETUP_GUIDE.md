# 🚀 AUTO NLP - HƯỚNG DẪN SETUP CHO MÁY MỚI

## 📋 **YÊU CẦU HỆ THỐNG**

### **Phần cứng tối thiểu:**
- **CPU**: Intel i5/AMD Ryzen 5 trở lên
- **RAM**: 8GB (khuyến nghị 16GB)
- **GPU**: NVIDIA GTX 1060/RTX 2060 trở lên (6GB VRAM)
- **Ổ cứng**: 20GB trống

### **Phần mềm:**
- **Python**: 3.8+ (khuyến nghị 3.11)
- **CUDA**: 12.1+ (cho GPU NVIDIA)
- **Git**: Để clone repository

## 🔧 **SETUP TỰ ĐỘNG (KHUYẾN NGHỊ)**

### **Bước 1: Clone repository**
```bash
git clone https://github.com/Scclegionx/Auto.git
cd Auto
```

### **Bước 2: Chạy setup tự động**
```bash
python setup_complete.py
```

**Script này sẽ tự động:**
- ✅ Kiểm tra Python version
- ✅ Tạo virtual environment mới
- ✅ Cài đặt tất cả packages cần thiết
- ✅ Xóa model cache cũ
- ✅ Tải PhoBERT-large model
- ✅ Test training script

### **Bước 3: Activate virtual environment**
```bash
# Windows
venv_new\Scripts\activate

# Linux/Mac
source venv_new/bin/activate
```

## 🎯 **SETUP THỦ CÔNG (NẾU CẦN)**

### **Bước 1: Tạo virtual environment**
```bash
python -m venv venv_new
```

### **Bước 2: Activate và cài đặt packages**
```bash
# Windows
venv_new\Scripts\activate
pip install --upgrade pip

# Linux/Mac
source venv_new/bin/activate
pip install --upgrade pip
```

### **Bước 3: Cài đặt PyTorch với CUDA**
```bash
pip install torch>=2.5.0 torchvision>=0.20.0 torchaudio>=2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

### **Bước 4: Cài đặt packages khác**
```bash
pip install transformers>=4.30.0 datasets>=2.12.0 accelerate>=0.20.0
pip install scikit-learn>=1.3.0 seqeval>=1.2.2 numpy>=1.24.0 pandas>=2.0.0
pip install fastapi>=0.100.0 uvicorn>=0.20.0 pydantic>=2.0.0
pip install underthesea>=6.6.0 pyvi>=0.1.1
pip install faiss-cpu>=1.7.0 rapidfuzz>=3.0.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0
pip install wandb>=0.15.0 tensorboard>=2.13.0
```

## 🧪 **KIỂM TRA SETUP**

### **Test 1: Kiểm tra imports**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### **Test 2: Test model loading**
```bash
python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large'); print('✅ Model OK')"
```

### **Test 3: Test training script**
```bash
python src/training/scripts/train_gpu.py --help
```

## 🚀 **CHẠY HỆ THỐNG**

### **Chạy API Server**
```bash
python api/server.py
```
Truy cập: `http://localhost:8000`

### **Chạy Web Interface**
Mở file `web_interface.html` trong browser

### **Chạy Training**
```bash
python src/training/scripts/train_gpu.py
```

## 🔧 **TROUBLESHOOTING**

### **Lỗi CUDA không có**
```bash
# Kiểm tra CUDA
nvidia-smi

# Cài đặt lại PyTorch CPU-only
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **Lỗi memory không đủ**
```bash
# Giảm batch size trong training
python src/training/scripts/train_gpu.py --batch_size 8
```

### **Lỗi model không tải được**
```bash
# Xóa cache và tải lại
rm -rf model_cache/
python setup_complete.py
```

## 📞 **HỖ TRỢ**

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra log files trong `models/logs/`
2. Chạy `python setup_complete.py` để reset
3. Tạo issue trên GitHub repository

---
**🎉 Chúc bạn setup thành công!**
