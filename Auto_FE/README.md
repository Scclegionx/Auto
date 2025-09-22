# Auto FE - Ứng dụng Tự động hóa Thao tác Điện thoại

## Mô tả
Auto FE là ứng dụng Android hỗ trợ tự động hóa các thao tác trên điện thoại thông qua lệnh giọng nói. Ứng dụng sử dụng AI NLP để hiểu lệnh và thực hiện các tác vụ tự động như gửi tin nhắn, gọi điện.

## Tính năng chính
- 🎤 Ghi âm lệnh bằng giọng nói
- 🤖 Xử lý lệnh bằng AI NLP
- 📱 Tự động hóa SMS và cuộc gọi
- 🪟 Cửa sổ nổi tiện lợi
- ✅ Xác nhận lệnh trước khi thực hiện

## Cấu trúc dự án
```
app/src/main/java/com/auto_fe/auto_fe/
├── audio/              # Xử lý âm thanh và giọng nói
│   ├── AudioRecorder.kt
│   └── AudioManager.kt
├── automation/         # Tự động hóa thao tác
│   ├── msg/
│   │   └── SMSAutomation.kt
│   └── phone/
│       └── PhoneAutomation.kt
├── core/              # Xử lý lệnh chính
│   └── CommandProcessor.kt
├── service/           # Dịch vụ giao tiếp
│   └── NLPService.kt
├── ui/                # Giao diện người dùng
│   └── FloatingWindow.kt
├── utils/             # Tiện ích hỗ trợ
│   └── PermissionManager.kt
└── MainActivity.kt    # Activity chính
```

## Yêu cầu hệ thống
- Android 11 (API 30) trở lên
- Quyền truy cập microphone
- Quyền gửi SMS
- Quyền gọi điện
- Quyền hiển thị trên các ứng dụng khác
- Server NLP chạy trên localhost:8000

## Cài đặt và sử dụng

### 1. Cài đặt ứng dụng
```bash
# Build ứng dụng
./gradlew assembleDebug

# Cài đặt APK
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 2. Cấp quyền
Khi chạy ứng dụng lần đầu, cần cấp các quyền sau:
- Microphone (ghi âm)
- SMS (gửi tin nhắn)
- Phone (gọi điện)
- Display over other apps (cửa sổ nổi)

### 3. Sử dụng
1. Mở ứng dụng
2. Cấp quyền cần thiết
3. Cửa sổ nổi "Auto FE" sẽ xuất hiện
4. Nhấn vào cửa sổ nổi
5. Chọn "Ghi âm lệnh"
6. Nói lệnh của bạn (ví dụ: "Nhắn tin cho mẹ là con sắp về")
7. Xác nhận lệnh
8. Ứng dụng sẽ thực hiện lệnh tự động

## Cấu hình Server NLP

### Yêu cầu Server
Server NLP cần chạy trên `localhost:8000` với endpoint `/infer`

### Format Request
```json
{
  "input": "nhắn tin cho mẹ là con sắp về"
}
```

### Format Response
```json
{
  "command": "sms",
  "ent": "{\"recipient\": \"mẹ\"}",
  "val": "{\"message\": \"con sắp về\"}"
}
```

## API Automation

### SMS Automation
- Gửi tin nhắn đến số điện thoại
- Hỗ trợ tin nhắn dài (tự động chia nhỏ)

### Phone Automation
- Gọi điện đến số điện thoại
- Quay số (mở dialer)

## Xử lý lỗi
- Kiểm tra quyền ứng dụng
- Kiểm tra kết nối mạng
- Kiểm tra server NLP
- Kiểm tra microphone

## Phát triển

### Thêm tính năng mới
1. Tạo class automation trong package `automation`
2. Implement interface callback
3. Thêm logic xử lý trong `CommandProcessor`
4. Cập nhật NLP response format

### Debug
```bash
# Xem log
adb logcat | grep "Auto_FE"

# Kiểm tra permissions
adb shell dumpsys package com.auto_fe.auto_fe
```

## License
MIT License

## Đóng góp
Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.
