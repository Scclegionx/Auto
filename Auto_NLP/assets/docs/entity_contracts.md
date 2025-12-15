### Hợp đồng entity giữa backend AI và FE

Tài liệu này mô tả **các trường entity mà backend AI cam kết trả về** cho từng `command`
(sau tầng hybrid / post‑processing).  
FE có thể dựa vào đây để parse JSON và xử lý nghiệp vụ.

> Ghi chú:
> - **Bắt buộc (hard)**: đa số câu đúng phải có, nếu thiếu nên coi là lỗi / cần fallback.
> - **Khuyến nghị (soft)**: dùng được nếu có, thiếu thì vẫn xử lý được.

---

### 1. `send-mess` – Gửi tin nhắn

- **Bắt buộc (ít nhất)**:
  - Một trong các nhóm:
    - `RECEIVER` **hoặc** `CONTACT_NAME` **hoặc** `PHONE`
    - `MESSAGE`
- **Khuyến nghị**:
  - `PLATFORM` (zalo, messenger, sms, …)

JSON ví dụ:

```json
{
  "command": "send-mess",
  "entities": {
    "RECEIVER": "con trai",
    "PLATFORM": "zalo",
    "MESSAGE": "hôm nay ông bận không sang chơi được"
  }
}
```

---

### 2. `set-alarm` – Báo thức / nhắc nhở

- **Bắt buộc (hard)**:
  - `TIME` – dạng chuẩn `HH:MM` (ví dụ `"06:30"`).
  - `TIMESTAMP` – thời điểm tuyệt đối ISO, theo `Asia/Bangkok`.
- **Khuyến nghị (soft)**:
  - `DATE` – ngày ISO `YYYY-MM-DD`.
  - `MESSAGE` – nội dung nguyên văn mà người dùng nói.
  - `REMINDER_CONTENT` – thường `= MESSAGE`, FE có thể dùng trực tiếp.

JSON ví dụ:

```json
{
  "command": "set-alarm",
  "entities": {
    "TIME": "06:30",
    "DATE": "2025-12-12",
    "TIMESTAMP": "2025-12-12T06:30:00+07:00",
    "MESSAGE": "6 rưỡi sáng mai nhắc uống thuốc",
    "REMINDER_CONTENT": "6 rưỡi sáng mai nhắc uống thuốc"
  }
}
```

---

### 3. `control-device` – Điều khiển thiết bị

- **Bắt buộc (hard)**:
  - `DEVICE` – **tên thiết bị thuần** hoặc tham số điều khiển (`đèn`, `đèn pin`, `quạt trần`, `âm lượng`, `độ sáng`, `wifi`, …) – **không bao gồm giá trị số**.
  - Ít nhất một trong:
    - `ACTION` (`bật`/`tắt`/`+`/`-`/`mở`/`đóng`…)
    - `MODE` (`on`/`off`/`up`/`down`…)
- **Khuyến nghị**:
  - `LEVEL` – mức tăng/giảm hoặc giá trị phần trăm (`small`, `medium`, `"10"`, `"30"`…).
  - `VALUE` – giá trị tuyệt đối cho thiết bị (ví dụ: `26` trong `"26 độ"` cho điều hòa).
  - `LOCATION` – nếu suy ra được (phòng khách, phòng ngủ…).

---

### 4. `call` – Gọi thoại

- **Bắt buộc (ít nhất)**:
  - Một trong các nhóm:
    - `CONTACT_NAME` **hoặc** `RECEIVER`
    - `PHONE`
- **Khuyến nghị**:
  - `PLATFORM` – nếu là cuộc gọi qua ứng dụng (zalo, messenger…), nếu không có thì coi là cuộc gọi thường.

---

### 5. `make-video-call` – Gọi video

- **Bắt buộc (hard)**:
  - `PLATFORM` – ứng dụng gọi video (mặc định `zalo` nếu không phát hiện được).
- **Bắt buộc (ít nhất)**:
  - Một trong:
    - `CONTACT_NAME`
    - `RECEIVER`
    - `PHONE`

---

### 6. `open-cam` – Mở camera

- **Bắt buộc (hard)**:
  - `ACTION` – `"mở"`/`"tắt"`.
  - `MODE` – `"trước"` hoặc `"sau"`.
  - `CAMERA_TYPE` – `"image"` (chụp ảnh) hoặc `"video"` (quay video).

---

### 7. `search-internet` – Tìm kiếm trên web

- **Bắt buộc (hard)**:
  - `QUERY` – nội dung cần tìm.
  - `PLATFORM` – công cụ tìm kiếm, mặc định `"google"`.

---

### 8. `search-youtube` – Tìm video trên YouTube

- **Bắt buộc (hard)**:
  - `QUERY` – nội dung video cần xem.
  - `PLATFORM` – luôn `"youtube"`.

---

### 9. `add-contacts` – Thêm liên hệ

- **Bắt buộc (ít nhất)**:
  - `PHONE`.
- **Khuyến nghị**:
  - `CONTACT_NAME`.



