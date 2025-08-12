# 🚀 PhoBERT_SAM - Vietnamese NLP System

Hệ thống NLP tiếng Việt với 3 thành phần chính: **Intent Recognition**, **Entity Extraction**, và **Command Processing**.

## **🎯 Tính năng chính**

- **🎯 Intent Recognition**: Nhận diện ý định người dùng (17 loại intent)
- **🏷️ Entity Extraction**: Trích xuất thông tin từ text (47 loại entity)
- **⚙️ Command Processing**: Xử lý và thực thi lệnh tương ứng

## **🚀 Khởi động nhanh**

### **1. Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

### **2. Khởi động API Server**
```bash
python api_server.py
```

### **3. Khởi động Client GUI**
```bash
python phobert_client.py
```

## **📁 Cấu trúc dự án**

## **🧪 Test API**

### **Health Check**
```bash
curl http://localhost:5000/health
```

### **Prediction**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "gửi tin nhắn cho mẹ: yêu mẹ nhiều"}'
```

## **📊 Modes**

- **Mock Mode**: Dự đoán dựa trên keywords (khi chưa train models)
- **Production Mode**: Sử dụng AI models đã train

## **🎯 Supported Intents**

1. `send-mess` - Gửi tin nhắn
2. `set-alarm` - Đặt báo thức
3. `call` - Gọi điện
4. `check-weather` - Kiểm tra thời tiết
5. `play-media` - Phát media
6. `check-health-status` - Kiểm tra sức khỏe
7. `read-news` - Đọc tin tức
8. `set-reminder` - Đặt nhắc nhở
9. Và 8 loại khác...

## **🏷️ Supported Entities**

- **RECEIVER**: Người nhận
- **TIME**: Thời gian
- **MESSAGE**: Nội dung tin nhắn
- **LOCATION**: Địa điểm
- **ARTIST**: Nghệ sĩ
- **WEATHER**: Thời tiết
- Và 41 loại khác...

## **📖 Documentation**

Xem `CLIENT_USAGE_GUIDE.md` để biết chi tiết cách sử dụng.

## **🚀 Development**

### **Train Models**
```bash
python main.py --mode train
```

### **Test Models**
```bash
python main.py --mode test
```

## **✅ Status**

- ✅ API Server hoàn chỉnh
- ✅ Client GUI hoàn chỉnh
- ✅ Dataset đã sẵn sàng (465 samples)
- ✅ Config đã cập nhật đầy đủ
- ✅ Ready for production

**PhoBERT_SAM - Vietnamese NLP System** 🚀
