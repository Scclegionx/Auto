# 🚀 DEPLOYMENT GUIDE - Auto NLP Hybrid System

## 📋 System Overview
Vietnamese NLP Hybrid System combining trained PhoBERT multitask model with specialized entity extractors and reasoning engine for intent classification and entity extraction.

**Supported Intents:**
- `call` - Gọi điện thoại
- `send-mess` - Gửi tin nhắn (SMS/Zalo/Facebook)
- `set-alarm` - Đặt báo thức
- `control-device` - Điều khiển thiết bị (wifi, bluetooth, flash, etc.)
- `search-internet` - Tìm kiếm web
- `search-youtube` - Tìm kiếm YouTube
- `open-cam` - Mở camera
- `add-contacts` - Thêm contact
- `make-video-call` - Gọi video
- `get-info` - Lấy thông tin

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.8 - 3.10 (recommended: 3.9)
- **RAM**: Minimum 4GB, Recommended 8GB
- **Disk**: 2-3GB (models + cache)
- **GPU**: Optional (CUDA-capable for faster inference)

### Dependencies
All dependencies are in `requirements.txt`:
- PyTorch (CPU or GPU version)
- Transformers (HuggingFace)
- FastAPI & Uvicorn
- sentence-transformers
- scikit-learn

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd Auto_NLP
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv_new
.\venv_new\Scripts\activate

# Linux/macOS
python3 -m venv venv_new
source venv_new/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in project root (see `.env.example`):

```env
# Model Configuration
MODEL_PATH=models/phobert_multitask_lr1e5
MODEL_CACHE_DIR=model_cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_FILE=reasoning_engine.log

# Python Path
PYTHONPATH=src;.

# Optional: Performance
TORCH_NUM_THREADS=4
OMP_NUM_THREADS=4
```

### For Windows PowerShell
```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
```

### For Linux/macOS
```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
```

---

## 🚀 Running the API Server

### Development Mode
```bash
# Windows
cd Auto_NLP
$env:PYTHONPATH="$PWD\src;$PWD"
.\venv_new\Scripts\python.exe api\server.py

# Linux/macOS
cd Auto_NLP
export PYTHONPATH="$(pwd)/src:$(pwd)"
python api/server.py
```

### Production Mode (with Uvicorn)
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Expected Output
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
🚀 Initializing Hybrid System...
✅ Trained model loaded successfully
✅ Reasoning engine initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Startup Time**: ~20-40 seconds (model loading)

---

## 📡 API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-27T22:00:00.000000",
  "version": "1.0.0",
  "hybrid_system_status": {
    "model_loaded": true,
    "reasoning_loaded": true,
    "device": "cuda"
  },
  "system_stats": {
    "total_predictions": 100,
    "avg_processing_time": 0.35,
    "avg_confidence": 0.89,
    "success_rate": 0.98
  }
}
```

---

### 2. Predict Intent & Entities
**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "text": "Gọi cho mẹ"
}
```

**Response**:
```json
{
  "input_text": "Gọi cho mẹ",
  "intent": "call",
  "confidence": 0.98,
  "command": "call",
  "entities": {
    "RECEIVER": "mẹ"
  },
  "method": "hybrid",
  "processing_time": 0.25,
  "timestamp": "2025-12-27T22:00:00.000000",
  "entity_clarity_score": 1.0,
  "nlp_response": null,
  "decision_reason": "high_model_confidence_with_specialized_entities"
}
```

**Field Descriptions**:
- `intent`: Classified intent (command type)
- `confidence`: Model confidence score (0-1)
- `entities`: Extracted entities relevant to intent
- `entity_clarity_score`: Entity extraction quality (0-1, target > 0.8)
- `method`: Processing method (`hybrid`, `model`, `reasoning`)
- `decision_reason`: Why this intent was chosen

---

### 3. Example Requests

#### Send Message
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Nhắn tin cho chị Mai tối nay con bận"}'
```

Response:
```json
{
  "intent": "send-mess",
  "entities": {
    "RECEIVER": "chị Mai",
    "MESSAGE": "tối nay con bận",
    "PLATFORM": "sms"
  }
}
```

#### Set Alarm
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Đặt báo thức 6 rưỡi sáng mai"}'
```

Response:
```json
{
  "intent": "set-alarm",
  "entities": {
    "TIME": "06:30",
    "DATE": "2025-12-28",
    "TIMESTAMP": "2025-12-28T06:30:00"
  }
}
```

#### Control Device
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bật wifi"}'
```

Response:
```json
{
  "intent": "control-device",
  "entities": {
    "DEVICE": "wifi",
    "ACTION": "ON"
  }
}
```

---

## 📊 Monitoring & Performance

### Health Monitoring
```bash
# Quick health check
curl http://127.0.0.1:8000/health

# Watch health continuously
watch -n 5 curl -s http://127.0.0.1:8000/health
```

### Performance Metrics
- **Target Latency**: < 500ms per request
- **Target Intent Accuracy**: > 95%
- **Target Entity Clarity**: > 0.85
- **Memory Usage**: ~1.5-2GB (with model loaded)

### Logs
- **API Logs**: Console output (Uvicorn)
- **Reasoning Logs**: `reasoning_engine.log` (if enabled)
- **Check Logs**:
  ```bash
  tail -f reasoning_engine.log
  ```

---

## 🐛 Troubleshooting

### Issue: Model Not Found
**Error**: `FileNotFoundError: models/phobert_multitask_lr1e5/best_model.pt`

**Solution**:
1. Verify model files exist in `models/phobert_multitask_lr1e5/`
2. Check `MODEL_PATH` in config
3. Ensure you're running from project root

---

### Issue: Import Errors
**Error**: `ModuleNotFoundError: No module named 'inference'`

**Solution**:
Set PYTHONPATH correctly:
```bash
# Windows
$env:PYTHONPATH="$PWD\src;$PWD"

# Linux/macOS
export PYTHONPATH="$(pwd)/src:$(pwd)"
```

---

### Issue: Slow Inference
**Symptoms**: Processing time > 1s

**Solutions**:
1. **Use GPU** (if available):
   - Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
   - Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Optimize environment**:
   ```bash
   export TORCH_NUM_THREADS=4
   export OMP_NUM_THREADS=4
   ```

3. **Reduce model size** (if needed):
   - Use quantization or distilled models

---

### Issue: Out of Memory
**Error**: `CUDA out of memory` or system RAM exhausted

**Solutions**:
1. Use CPU instead of GPU:
   ```python
   # In config.py
   DEVICE = "cpu"
   ```

2. Reduce batch size or model cache

3. Close other memory-intensive applications

---

## 🔒 Security Considerations

### Production Deployment
1. **Never expose API directly to internet** without authentication
2. **Use reverse proxy** (nginx, Apache) with rate limiting
3. **Enable HTTPS** for secure communication
4. **Sanitize inputs** to prevent injection attacks
5. **Set up monitoring** for anomalous requests

### Example nginx config
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🧪 Testing

### Quick Test
```bash
# Test all endpoints
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Gọi cho bà"}'
```

### Integration Test
Run the full test suite to verify all intents:
```bash
python scripts/test_final_deployment.py
```

---

## 📝 API Response Schema

### Intent Response
```typescript
{
  input_text: string;        // Original input text
  intent: string;            // Classified intent
  confidence: float;         // Confidence score (0-1)
  command: string;           // Command to execute
  entities: object;          // Extracted entities
  method: string;            // Processing method
  processing_time: float;    // Time in seconds
  timestamp: string;         // ISO format timestamp
  entity_clarity_score: float; // Entity quality (0-1)
  nlp_response: string | null; // Additional response
  decision_reason: string;   // Decision explanation
}
```

---

## 🎯 Best Practices

1. **Always check `/health` before sending requests**
2. **Monitor `entity_clarity_score`** (target > 0.8)
3. **Handle low confidence** (< 0.7) predictions with fallback
4. **Log failed predictions** for model improvement
5. **Rate limit client requests** to avoid overload

---

## 📞 Support

For issues or questions:
- Check logs: `reasoning_engine.log`
- Review troubleshooting section
- Contact: [your-contact-info]

---

## 🎉 Ready to Deploy!

Your API is now ready for production deployment. Monitor the `/health` endpoint and enjoy high-quality Vietnamese NLP! 🚀

