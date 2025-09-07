# Hướng dẫn Setup Auto_NLP

## 1. Cài đặt Python Environment

```bash
# Tạo virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

## 2. Kiểm tra GPU (nếu có)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 3. Test toàn bộ hệ thống

```bash
python test_training_complete.py
```

## 4. Chạy Training

```bash
# Training với GPU
python run_training.py

# Hoặc training với CPU
python src/training/scripts/train_gpu.py
```

## 5. Chạy API

```bash
python run_api.py
```

## Troubleshooting

### Lỗi CUDA
- Kiểm tra driver NVIDIA
- Cài đặt PyTorch với CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Lỗi Memory
- Giảm batch_size trong config
- Giảm max_length
- Sử dụng gradient_checkpointing=True

### Lỗi Model Download
- Kiểm tra kết nối internet
- Model sẽ được cache tự động
