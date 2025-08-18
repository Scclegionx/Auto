# Hướng Dẫn Training Model PhoBERT_SAM cho Người Cao Tuổi

## Tổng quan
Hướng dẫn này sẽ giúp bạn train model PhoBERT_SAM chuyên biệt cho người cao tuổi với dataset đã được tối ưu hóa.

## Chuẩn bị

### 1. Kiểm tra môi trường
```bash
# Kiểm tra Python version
python --version  # Python 3.8+

# Kiểm tra PyTorch
python -c "import torch; print(torch.__version__)"

# Kiểm tra CUDA (nếu có)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Kiểm tra dataset
```bash
# Kiểm tra dataset tồn tại
ls -la elderly_command_dataset_reduced.json

# Kiểm tra kích thước dataset
python -c "import json; data=json.load(open('elderly_command_dataset_reduced.json')); print(f'Dataset size: {len(data)} samples')"
```

## Training

### 1. Training cơ bản
```bash
# Training với cấu hình mặc định
python train_elderly.py

# Training với tham số tùy chỉnh
python train_elderly.py \
    --dataset elderly_command_dataset_reduced.json \
    --output models/elderly \
    --epochs 15 \
    --batch_size 16 \
    --lr 2e-5 \
    --seed 42
```

### 2. Training với cấu hình nâng cao
```bash
# Training với batch size lớn hơn (nếu có GPU)
python train_elderly.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-5

# Training với learning rate thấp hơn
python train_elderly.py \
    --epochs 25 \
    --lr 5e-6
```

### 3. Training từng model riêng biệt
```bash
# Chỉ train Intent Recognition
python train_elderly.py --task intent

# Chỉ train Entity Extraction  
python train_elderly.py --task entity

# Chỉ train Command Processing
python train_elderly.py --task command
```

## Cấu hình Training

### Tham số quan trọng

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `--epochs` | 10 | Số epoch training |
| `--batch_size` | 16 | Kích thước batch |
| `--lr` | 2e-5 | Learning rate |
| `--seed` | 42 | Random seed |
| `--output` | models/elderly | Thư mục lưu model |

### Cấu hình cho GPU
```bash
# Nếu có GPU, tăng batch size
python train_elderly.py --batch_size 32

# Nếu GPU memory lớn
python train_elderly.py --batch_size 64
```

### Cấu hình cho CPU
```bash
# Giảm batch size cho CPU
python train_elderly.py --batch_size 8

# Tăng epochs để bù đắp
python train_elderly.py --batch_size 8 --epochs 20
```

## Monitoring Training

### 1. Logs
Training logs được lưu trong thư mục `logs/`:
```bash
# Xem log mới nhất
tail -f logs/training_*.log

# Tìm log theo ngày
ls logs/training_*.log | grep $(date +%Y%m%d)
```

### 2. Metrics được theo dõi
- **Train Loss**: Loss trong quá trình training
- **Validation Loss**: Loss trên validation set
- **Accuracy**: Độ chính xác trên validation set
- **Early Stopping**: Dừng training khi validation loss không giảm

### 3. Model Checkpoints
Models được lưu tự động trong thư mục `models/elderly/`:
```
models/elderly/
├── best_intent_model.pth
├── best_intent_model_metadata.json
├── best_entity_model.pth
├── best_entity_model_metadata.json
├── best_command_model.pth
├── best_command_model_metadata.json
├── best_unified_model.pth
└── best_unified_model_metadata.json
```

## Đánh giá Model

### 1. Kiểm tra metadata
```bash
# Xem thông tin model
cat models/elderly/best_intent_model_metadata.json | python -m json.tool
```

### 2. Test model đã train
```bash
# Test với API
python demo_api.py --mode interactive

# Test với inference
python inference.py --text "Gọi cho mẹ tôi"
```

### 3. Benchmark performance
```bash
# Test accuracy
python test_api.py

# Test speed
python -c "
import time
from api_wrapper import PhoBERTSAMAPI
api = PhoBERTSAMAPI('models/elderly/best_unified_model.pth')
start = time.time()
for _ in range(100):
    api.process_text('Gọi cho mẹ')
print(f'Average time: {(time.time() - start) / 100:.3f}s')
"
```

## Troubleshooting

### 1. Lỗi Out of Memory
```bash
# Giảm batch size
python train_elderly.py --batch_size 8

# Hoặc sử dụng gradient accumulation
python train_elderly.py --batch_size 4 --gradient_accumulation_steps 4
```

### 2. Training quá chậm
```bash
# Tăng batch size (nếu có GPU)
python train_elderly.py --batch_size 32

# Giảm số epochs
python train_elderly.py --epochs 5
```

### 3. Model không converge
```bash
# Giảm learning rate
python train_elderly.py --lr 1e-5

# Tăng epochs
python train_elderly.py --epochs 20
```

### 4. Overfitting
```bash
# Tăng weight decay
python train_elderly.py --weight_decay 0.1

# Giảm epochs
python train_elderly.py --epochs 5
```

## Tối ưu hóa

### 1. Hyperparameter Tuning
```bash
# Thử các learning rate khác nhau
for lr in 1e-5 2e-5 5e-5; do
    python train_elderly.py --lr $lr --output models/elderly_lr_$lr
done

# Thử các batch size khác nhau
for bs in 8 16 32; do
    python train_elderly.py --batch_size $bs --output models/elderly_bs_$bs
done
```

### 2. Data Augmentation
```bash
# Tăng augmentation factor
python train_elderly.py --augmentation_factor 0.5

# Giảm augmentation factor
python train_elderly.py --augmentation_factor 0.1
```

### 3. Model Size
```bash
# Sử dụng PhoBERT-large (nếu có đủ memory)
python train_elderly.py --model_size large

# Sử dụng PhoBERT-base (mặc định)
python train_elderly.py --model_size base
```

## Production Deployment

### 1. Export model
```bash
# Tạo production model
python -c "
import torch
from models import create_model
model = create_model('unified')
model.load_state_dict(torch.load('models/elderly/best_unified_model.pth'))
torch.save(model, 'models/elderly/production_model.pt')
"
```

### 2. Test production model
```bash
# Test với production model
python api_wrapper.py --model_path models/elderly/production_model.pt --text "Gọi cho mẹ"
```

### 3. Performance monitoring
```bash
# Monitor memory usage
python -c "
import psutil
import time
from api_wrapper import PhoBERTSAMAPI
api = PhoBERTSAMAPI('models/elderly/best_unified_model.pth')
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Kết quả mong đợi

### Metrics tốt
- **Intent Accuracy**: >90%
- **Entity F1-score**: >85%
- **Command Accuracy**: >95%
- **Inference Time**: <100ms
- **Memory Usage**: <2GB

### Cải thiện so với rule-based
- **Accuracy**: Tăng từ ~47% lên >90%
- **Confidence**: Tăng từ 0.5-0.7 lên 0.8-0.95
- **Robustness**: Xử lý được nhiều cách diễn đạt khác nhau

## Lưu ý quan trọng

1. **Dataset Quality**: Đảm bảo dataset có chất lượng tốt và cân bằng
2. **Validation**: Luôn validate model trên test set riêng biệt
3. **Monitoring**: Theo dõi training để tránh overfitting
4. **Backup**: Backup models và logs thường xuyên
5. **Documentation**: Ghi chép lại các tham số training tốt nhất

## Hỗ trợ

Nếu gặp vấn đề trong quá trình training:
1. Kiểm tra logs trong thư mục `logs/`
2. Kiểm tra metadata của model
3. Test với dataset nhỏ trước
4. Đảm bảo đủ memory và disk space
