# Raw Data

Thư mục này lưu trữ các bộ dữ liệu gốc phục vụ huấn luyện và suy luận cho hệ thống NLP tiếng Việt dành cho người cao tuổi.

## Dataset chính

- `elderly_commands_master.json`
  - ~34.897 câu lệnh (train ~33k, val 928, test 969)
  - Intent cân bằng: `add-contacts`, `call`, `control-device`, `get-info`, `make-video-call`, `open-cam`, `search-internet`, `search-youtube`, `send-mess`, `set-alarm`
  - Entity chuẩn: `ACTION`, `CAMERA_TYPE`, `CONTACT_NAME`, `DATE`, `DEVICE`, `FREQUENCY`, `LEVEL`, `LOCATION`, `MESSAGE`, `MODE`, `PHONE`, `PLATFORM`, `QUERY`, `RECEIVER`, `REMINDER_CONTENT`, `TIME`
  - Được refactor/chuẩn hóa cuối cùng (MESSAGE/QUERY/DEVICE, spans, IOB2) và là **file dataset gốc duy nhất** dùng để sinh bộ processed (`train.json`, `val.json`, `test.json`) hiện hành.

## Dataset legacy / lưu trữ

- Các phiên bản cũ (`elderly_command_dataset_FINAL.json`, `elderly_command_dataset_MERGED_13C_VITEXT.json`, …) đã được di chuyển vào `src/data/raw/archive/` để phục vụ đối chiếu lịch sử. Chi tiết xem thêm trong `src/data/raw/archive/README.md`.

## Các tệp hỗ trợ

- `entity_vocab_clean.json`: từ vựng entity gốc dùng tham chiếu đối chiếu.
- `normalize_numbers_vi_dataset.py`: script chuẩn hóa số trong dữ liệu thô.
- Thư mục `archive/`: lưu các snapshot dữ liệu lịch sử phục vụ đối chiếu.

## Sử dụng

- Quá trình tiền xử lý & tách tập đã được tự động hóa trong pipeline, không cần chỉnh sửa thủ công.
- Khi cần tái sinh bộ processed, chạy các script tương ứng tại `src/data/processed/`.

_Last updated: 2025-11-17 (cập nhật FINAL dataset)_
