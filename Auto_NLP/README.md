# 🤖 Auto NLP Hybrid System

Hệ thống NLP Hybrid thông minh kết hợp trained model với reasoning engine, được thiết kế đặc biệt cho người cao tuổi với khả năng xử lý tiếng Việt tự nhiên.

## ✨ **TÍNH NĂNG NỔI BẬT**

- 🧠 **Hybrid Intelligence**: Kết hợp AI model với rule-based reasoning
- 🇻🇳 **Tiếng Việt Native**: Xử lý tiếng Việt có dấu và không dấu
- 👴 **Thân thiện người già**: Giao diện đơn giản, dễ sử dụng
- 📱 **Multi-platform**: Hỗ trợ Zalo, WhatsApp, Messenger, Telegram
- 🎯 **Intent Recognition**: Nhận diện 15+ loại intent phổ biến
- 🔍 **Entity Extraction**: Trích xuất thông tin chính xác
- ⚡ **Real-time**: Phản hồi nhanh chóng

## 🚀 **QUICK START**

### **1. Clone Repository**
```bash
git clone <repository-url>
cd Auto_NLP
```

### **2. Setup Environment**
```bash
# Tạo virtual environment
python -m venv venv_new
source venv_new/bin/activate  # Linux/Mac
# hoặc
venv_new\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### **3. Download Model Files**
```bash
# Xem hướng dẫn chi tiết trong:
cat models/MODEL_SETUP.md
```

### **4. Configure Environment**
```bash
# Copy & edit environment file
cp env.example .env
# Edit .env với text editor
```

### **5. Start Server**
```bash
# Set PYTHONPATH
export PYTHONPATH="$PWD/src:$PWD"  # Linux/Mac
# hoặc
$env:PYTHONPATH="$PWD\src;$PWD"    # Windows PowerShell

# Run API server
python api/server.py
```

Server sẽ chạy tại: `http://localhost:8000`  
API Docs: `http://localhost:8000/docs`

## 🎯 **SỬ DỤNG**

### **API Request Example**
```bash
# Test với curl
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Nhắn tin cho mẹ hỏi ăn cơm chưa"}'
```

### **Python Client Example**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"input_text": "Gọi điện cho con gái"}
)
print(response.json())
```

## 📚 **HƯỚNG DẪN CHI TIẾT**

- 🚀 **[DEPLOYMENT.md](DEPLOYMENT.md)** - Hướng dẫn deploy lên server chi tiết
- 📦 **[models/MODEL_SETUP.md](models/MODEL_SETUP.md)** - Hướng dẫn download & setup model files
- 🔧 **API Endpoints**: `http://localhost:8000/docs` - FastAPI auto-generated docs

## 🏗️ **KIẾN TRÚC HỆ THỐNG**

```
Auto_NLP/
├── api/                      # FastAPI REST API
│   └── server.py            # Main API server
├── core/                     # Core business logic
│   ├── hybrid_system.py     # Model-First Hybrid System
│   ├── reasoning_engine.py  # Rule-based reasoning
│   ├── entity_contracts.py  # Entity validation & whitelisting
│   └── *.json               # Knowledge base, patterns, rules
├── src/
│   ├── inference/
│   │   └── engines/         # Specialized entity extractors
│   ├── models/              # Model definitions (PhoBERT, etc.)
│   ├── training/            # Training scripts (optional)
│   └── data/                # Dataset configs
├── models/                   # Model files & configs
│   ├── phobert_multitask/   # Trained model (download separately)
│   └── configs/             # Label maps, training configs
├── resources/                # Vietnamese accent maps, etc.
├── scripts/                  # Utility & visualization scripts
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # This file
```

### **Hybrid Architecture**
1. **Input** → Voice-to-Text (Frontend) → NLP API
2. **Intent Prediction** → PhoBERT Multi-task Model
3. **Intent Guard** → 3-tier heuristic validation
4. **Entity Extraction** → Specialized extractors (confidence-based)
5. **Entity Validation** → Whitelist filtering & clarity scoring
6. **Output** → Clean JSON → Frontend execution

## 🎯 **INTENTS ĐƯỢC HỖ TRỢ**

| Intent | Mô tả | Ví dụ |
|--------|-------|-------|
| `add-contacts` | Thêm liên hệ | "lưu số điện thoại" |
| `call` | Gọi điện thoại | "gọi điện cho mẹ" |
| `control-device` | Điều khiển thiết bị | "bật wifi" |
| `get-info` | Lấy thông tin | "hỏi thời gian" |
| `make-video-call` | Video call | "gọi video với con" |
| `open-cam` | Mở camera | "mở camera sau" |
| `search-internet` | Tìm kiếm web | "tìm kiếm thời tiết" |
| `search-youtube` | Tìm YouTube | "tìm video ca nhạc" |
| `send-mess` | Nhắn tin | "nhắn tin cho bố" |
| `set-alarm` | Đặt báo thức | "đặt báo thức 7 giờ" |
| `help` | Trợ giúp | "giúp tôi" |

## 🔧 **CẤU HÌNH**

### **Yêu cầu hệ thống:**
- **Python**: 3.8+ (khuyến nghị 3.11)
- **GPU**: NVIDIA GTX 1060+ (6GB VRAM)
- **RAM**: 8GB+ (khuyến nghị 16GB)
- **CUDA**: 12.1+

### **Environment Variables:**
```bash
# Optional: Weights & Biases
export WANDB_API_KEY="your_wandb_key"

# Optional: Custom model path
export MODEL_PATH="models/trained/best_model"
```

## 📊 **PERFORMANCE**

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | 95%+ |
| **Entity F1 Score** | 88%+ |
| **Response Time** | 300-800ms |
| **Memory Usage** | ~2-4GB (with model loaded) |
| **Model Size** | ~1.4GB (PhoBERT-based) |
| **Supported Languages** | Vietnamese (primary) |

### **Specialized Extractors Confidence**
- **send-mess** (MESSAGE/RECEIVER): ≥0.80
- **set-alarm** (TIME/DATE): ≥0.80
- **control-device** (ACTION/DEVICE): ≥0.85

## 🤝 **ĐÓNG GÓP**

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 **LICENSE**

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 **LIÊN HỆ**

- **GitHub**: [Scclegionx/Auto](https://github.com/Scclegionx/Auto)
- **Issues**: [GitHub Issues](https://github.com/Scclegionx/Auto/issues)

---

**🎉 Cảm ơn bạn đã sử dụng Auto NLP Hybrid System!**
