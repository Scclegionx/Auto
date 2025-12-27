# Hướng dẫn gán nhãn & phong cách câu cho dự án Auto_NLP

## 1. Mục tiêu câu & cách gán nhãn

- **Mục tiêu**: câu ngắn, tự nhiên, giống cách người lớn tuổi nói chuyện với trợ lý. Giữ nội dung rõ ràng, không cần “mềm hóa” quá mức.
- **IOB2**: 
  - Token đầu thực thể → `B-ENTITY`.
  - Các token tiếp theo → `I-ENTITY`.
  - Không bao giờ bắt đầu bằng `I-ENTITY`.
- **Những lưu ý chính**:
  - `CONTACT_NAME`: tên lưu trong danh bạ (“Phan Thanh Chi”).
  - `RECEIVER`: cách xưng hô/quan hệ (“cậu Sáu”, “mẹ”, “dì Bảy”). Có thể dùng song song với `CONTACT_NAME`, nhưng không gán chồng lên nhau.
  - `PHONE`: chấp nhận chữ lẫn số (ví dụ “không chín năm…”).
  - `TIME`/`DATE`: gán cả cụm (“8 giờ sáng”, “chiều nay”, “thứ Hai tới”).
  - `MESSAGE`/`QUERY`/`REMINDER_CONTENT`: chỉ gán phần nội dung cần thiết sau động từ, không “nuốt” toàn bộ câu nếu chỉ một đoạn là message.

### Ví dụ IOB2
```
Nhắn tin cho cậu Sáu là chiều nay đến bệnh viện nhé.
RECEIVER = “cậu Sáu”          → B-RECEIVER I-RECEIVER
TIME     = “chiều nay”        → B-TIME I-TIME
LOCATION = “bệnh viện”        → B-LOCATION I-LOCATION
```

- **Chính tả & dấu câu**:
  - Giữ nguyên dấu nếu người dùng nói có.
  - Không bắt buộc thêm dấu chấm/phẩy ở cuối câu.

## 2. Từ điển gợi ý (seed lexicon)

> Chỉ dùng hỗ trợ annotate/augment, không phải ràng buộc cứng.

- **Xưng hô/quan hệ (`RECEIVER`)**: mẹ, ba/bố, má, cậu, dì, mợ, cô, chú, bác, anh, chị, em, ông nội, bà ngoại, anh rể, em dâu, con, cháu, thím…
- **Nền tảng (`PLATFORM`)**: Zalo, Messenger, SMS, Viber, WhatsApp, YouTube.
- **Thiết bị (`DEVICE`)**: đèn, quạt, TV, điều hòa, loa, rèm, camera, âm lượng, độ sáng, wifi, máy lọc không khí…
- **Hành động (`ACTION`)**: bật, tắt, tăng, giảm, mở, đóng, chuyển, đổi, tạm dừng, tiếp tục.
- **Chế độ (`MODE`)**: on, off, up, down (chuẩn hóa sau NER), im lặng, ban đêm, tiết kiệm.
- **Mức độ (`LEVEL`)**: một chút, nhẹ nhàng, thêm hai mức, 20%, 50%, 80%.
- **Loại camera (`CAMERA_TYPE`)**: trước, sau, selfie.
- **Thời gian/chu kỳ (`TIME`/`DATE`/`FREQUENCY`)**: 8 giờ sáng, 6 giờ chiều, chiều nay, tối nay, ngày mai, thứ Hai, tuần sau, hàng ngày, mỗi ngày, Th2–6, cuối tuần.
- **Địa danh (`LOCATION`)**: Hà Nội, TP.HCM, Đà Nẵng, Huế, Cần Thơ, Bệnh viện Bạch Mai, Chợ Rẫy…
- **Nhắc nhở/nội dung (`REMINDER_CONTENT`/`MESSAGE`/`QUERY`)**: uống thuốc huyết áp, hẹn bác sĩ, đo đường huyết, bài tập thở, đau lưng nhẹ…

## 3. Kịch bản augment chuyên biệt

Các phép biến đổi chữ (an toàn với giọng nói):

1. **Filler injection**: chèn “ờ/à/giúp tôi/làm ơn/nhé/nhá/ạ…” vào đầu, cuối hoặc sát động từ chính.
2. **Diacritic dropout**: bỏ dấu toàn phần/một phần (20–30%) cho đoạn không phải entity.
3. **Digit ↔ chữ**: đổi “0985 383 569” ↔ “không chín tám năm ba tám ba năm sáu chín”; “8 giờ rưỡi” ↔ “8:30”.
4. **Synonym hẹp**: gửi/nhắn/soạn; bật/mở; video/gọi hình; phát/chạy.
5. **Noise chính tả nhẹ**: 1 ký tự sai cho mỗi 10–12 token, không chạm span entity.
6. **Ràng buộc entity**: sau augment vẫn giữ đủ entity bắt buộc (ví dụ send-mess phải còn `MESSAGE`).

> Luôn chạy validator sau khi augment để đảm bảo thống kê và ràng buộc không bị phá vỡ.

