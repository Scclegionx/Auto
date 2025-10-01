# Auto NLP - Vietnamese Intent Recognition System

## 🎯 Overview
Hệ thống nhận diện ý định tiếng Việt sử dụng PhoBERT-Large cho ứng dụng chăm sóc người cao tuổi.

## ✨ Features
- **Multi-task learning**: Intent, Entity, Value, Command
- **PhoBERT-Large**: Kiến trúc mạnh mẽ cho tiếng Việt
- **GPU-optimized**: Tối ưu cho GPU 6GB
- **RESTful API**: Dễ dàng tích hợp
- **High Accuracy**: 84% accuracy, 0.8331 F1 score

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training
```bash
python run_training.py
```

### 3. API Server
```bash
python run_api.py
```

## 📊 Performance
- **Accuracy**: 84.00%
- **F1 Score**: 0.8331
- **Training Time**: 41.70 phút
- **Model Size**: PhoBERT-Large
- **Intent Classes**: 26 classes

## 🏗️ Project Structure
```
Auto_NLP/
├── src/                    # Source code
│   ├── training/           # Training modules
│   ├── inference/          # Inference modules
│   ├── data/              # Data processing
│   └── utils/             # Utilities
├── models/                 # Trained models
├── data/                   # Dataset files
└── scripts/               # Management scripts
```

## 🔧 Configuration
- **Model**: vinai/phobert-large
- **Max Length**: 64 tokens
- **Batch Size**: 2
- **Learning Rate**: 1e-5 (encoder), 3e-4 (heads)
- **GPU Memory**: 6GB optimized

## 📋 Intent Classes (26 classes)
1. **call** - Gọi điện
2. **send-mess** - Gửi tin nhắn
3. **check-weather** - Kiểm tra thời tiết
4. **play-content** - Phát nội dung
5. **set-reminder** - Đặt nhắc nhở
6. **make-video-call** - Gọi video
7. **search-content** - Tìm kiếm nội dung
8. **check-device-status** - Kiểm tra trạng thái thiết bị
9. **play-media** - Phát media
... và 17 classes khác

## 💻 Usage Examples

### Training
```bash
python run_training.py
```

### API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Gọi điện cho mẹ"}'
```

### Data Management
```bash
python scripts/management/organize_dataset.py
```

## 📚 Documentation
- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - Chi tiết hệ thống
- [QUICK_START.md](QUICK_START.md) - Hướng dẫn nhanh

## 🔧 Requirements
- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (optional)

## 📈 Results
Model đã được training thành công với:
- **84% accuracy** trên validation set
- **0.8331 F1 score** - chỉ số cân bằng tốt
- **26 intent classes** được nhận diện chính xác
- **Real-time inference** với API server

## 🎯 Next Steps
1. Deploy model to production
2. Add more intent classes
3. Improve accuracy with more data
4. Add entity extraction
5. Multi-language support

## 📄 License
MIT