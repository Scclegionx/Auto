# 🎯 AUTO NLP - HƯỚNG DẪN CHUẨN BỊ TRAINING DATA

## 📊 **TỔNG QUAN DATASET**

### **Cấu trúc dữ liệu hiện tại:**
```
src/data/
├── raw/                    # Dữ liệu gốc
│   ├── *.json             # Các file dataset gốc
│   └── *.md               # Documentation
├── processed/              # Dữ liệu đã xử lý
│   ├── *.json             # Dataset đã clean
│   └── *.py               # Scripts xử lý
├── grouped/               # Dữ liệu theo nhóm intent
│   ├── add-contacts.json  # Intent: thêm liên hệ
│   ├── call.json          # Intent: gọi điện
│   ├── send-mess.json     # Intent: nhắn tin
│   └── ...                # Các intent khác
└── augmented/             # Dữ liệu mở rộng
    └── expand_dataset.py  # Script tạo dữ liệu mở rộng
```

## 🔧 **CHUẨN BỊ DỮ LIỆU CHO TRAINING**

### **Bước 1: Kiểm tra dữ liệu hiện có**
```bash
# Kiểm tra số lượng samples
python -c "
import json
import os
from pathlib import Path

data_dir = Path('src/data/grouped')
total_samples = 0

for file in data_dir.glob('*.json'):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = len(data)
    print(f'{file.name}: {samples} samples')
    total_samples += samples

print(f'Total: {total_samples} samples')
"
```

### **Bước 2: Chuẩn bị dataset cho training**
```bash
# Chạy script chuẩn bị dữ liệu
python src/data/augmented/expand_dataset.py
```

**Script này sẽ:**
- ✅ Load tất cả datasets từ `src/data/grouped/`
- ✅ Chuẩn hóa format dữ liệu
- ✅ Tạo train/validation split
- ✅ Lưu vào `src/data/processed/`

### **Bước 3: Kiểm tra chất lượng dữ liệu**
```bash
# Kiểm tra distribution của labels
python -c "
import json
from collections import Counter

# Load processed data
with open('src/data/processed/train_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Count labels
labels = [item['intent'] for item in train_data]
label_counts = Counter(labels)

print('Label distribution:')
for label, count in label_counts.most_common():
    print(f'  {label}: {count} samples')

print(f'Total samples: {len(train_data)}')
print(f'Unique labels: {len(label_counts)}')
"
```

## 🎯 **CẤU HÌNH TRAINING**

### **File cấu hình: `models/configs/training_config.json`**
```json
{
  "model_name": "vinai/phobert-large",
  "max_length": 128,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 1,
  "save_steps": 1000,
  "eval_steps": 500,
  "logging_steps": 100,
  "output_dir": "models/trained/",
  "cache_dir": "model_cache/"
}
```

### **Tùy chỉnh cấu hình:**
```bash
# Chỉnh batch size cho GPU nhỏ
python src/training/scripts/train_gpu.py --batch_size 8

# Chỉnh learning rate
python src/training/scripts/train_gpu.py --learning_rate 1e-5

# Chỉnh số epochs
python src/training/scripts/train_gpu.py --num_epochs 15
```

## 🚀 **CHẠY TRAINING**

### **Training cơ bản:**
```bash
# Activate virtual environment
venv_new\Scripts\activate

# Chạy training
python src/training/scripts/train_gpu.py
```

### **Training với monitoring:**
```bash
# Với TensorBoard
python src/training/scripts/train_gpu.py --logging_dir models/logs/

# Với Weights & Biases
python src/training/scripts/train_gpu.py --use_wandb
```

### **Training với validation:**
```bash
# Chạy với validation set
python src/training/scripts/train_gpu.py --do_eval --eval_steps 500
```

## 📈 **MONITORING TRAINING**

### **TensorBoard:**
```bash
# Khởi động TensorBoard
tensorboard --logdir models/logs/

# Truy cập: http://localhost:6006
```

### **Weights & Biases:**
```bash
# Login W&B
wandb login

# Chạy training với W&B
python src/training/scripts/train_gpu.py --use_wandb --project_name auto-nlp
```

## 🔍 **EVALUATION**

### **Test model sau training:**
```bash
# Test trên validation set
python src/training/scripts/train_gpu.py --do_eval --eval_only

# Test với custom data
python -c "
from src.models.inference.trained_model_inference import TrainedModelInference

# Load trained model
inference = TrainedModelInference('models/trained/best_model')

# Test samples
test_samples = [
    'nhắn tin cho mẹ là con về nhà',
    'gọi điện cho bố',
    'tìm kiếm trên google về thời tiết'
]

for text in test_samples:
    result = inference.predict(text)
    print(f'Text: {text}')
    print(f'Intent: {result[\"intent\"]} (confidence: {result[\"confidence\"]:.3f})')
    print()
"
```

## 🎯 **OPTIMIZATION**

### **Tối ưu cho GPU nhỏ:**
```bash
# Giảm batch size
python src/training/scripts/train_gpu.py --batch_size 4

# Sử dụng gradient accumulation
python src/training/scripts/train_gpu.py --gradient_accumulation_steps 4

# Sử dụng mixed precision
python src/training/scripts/train_gpu.py --fp16
```

### **Tối ưu cho CPU:**
```bash
# Training trên CPU
python src/training/scripts/train_gpu.py --no_cuda

# Giảm max_length
python src/training/scripts/train_gpu.py --max_length 64
```

## 📊 **ANALYSIS**

### **Phân tích kết quả training:**
```bash
# Xem training logs
python -c "
import json
import matplotlib.pyplot as plt

# Load training logs
with open('models/logs/training_logs.json', 'r') as f:
    logs = json.load(f)

# Plot training curves
epochs = logs['epochs']
train_loss = logs['train_loss']
val_loss = logs['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.savefig('models/logs/training_curves.png')
plt.show()
"
```

## 🔧 **TROUBLESHOOTING**

### **Lỗi Out of Memory:**
```bash
# Giảm batch size
python src/training/scripts/train_gpu.py --batch_size 2

# Sử dụng gradient checkpointing
python src/training/scripts/train_gpu.py --gradient_checkpointing
```

### **Lỗi Data Loading:**
```bash
# Kiểm tra format dữ liệu
python -c "
import json
with open('src/data/processed/train_dataset.json', 'r') as f:
    data = json.load(f)
print('Sample:', data[0])
print('Keys:', data[0].keys())
"
```

### **Lỗi Model Loading:**
```bash
# Kiểm tra model cache
ls -la model_cache/

# Clear cache và tải lại
rm -rf model_cache/
python setup_complete.py
```

## 📋 **CHECKLIST TRƯỚC KHI TRAINING**

- [ ] ✅ Dữ liệu đã được chuẩn bị và clean
- [ ] ✅ Dataset có đủ samples cho mỗi intent (>50 samples)
- [ ] ✅ Train/validation split đã được tạo
- [ ] ✅ Model cache đã được tải
- [ ] ✅ GPU memory đủ cho batch size
- [ ] ✅ Cấu hình training đã được set
- [ ] ✅ Logging directory đã được tạo
- [ ] ✅ Output directory đã được tạo

---
**🎉 Chúc bạn training thành công!**
