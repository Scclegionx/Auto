# Raw Data

Thư mục này lưu trữ các bộ dữ liệu gốc phục vụ huấn luyện và suy luận cho hệ thống NLP tiếng Việt dành cho người cao tuổi.

## Dataset chính

- `elderly_command_dataset_MERGED_13C_VITEXT.json`
  - 10.186 câu lệnh đã chuẩn hóa theo chuẩn IOB2
  - 13 intent chính: `add-contacts`, `call`, `control-device`, `get-info`, `make-video-call`, `open-cam`, `play-media`, `search-internet`, `search-youtube`, `send-mess`, `set-alarm`, `set-event-calendar`, `view-content`
  - 13 nhóm entity: `ACTION`, `CONTACT_NAME`, `CONTENT_TYPE`, `DATE`, `DEVICE`, `LEVEL`, `LOCATION`, `MEDIA_TYPE`, `PHONE`, `PLATFORM`, `QUERY`, `TIME`, `TITLE`
  - Là nguồn đầu vào cho pipeline xử lý và sinh bộ processed (`train.json`, `val.json`, `test.json`)

## Các tệp hỗ trợ

- `entity_vocab_clean.json`: từ vựng entity gốc dùng tham chiếu đối chiếu.
- `normalize_numbers_vi_dataset.py`: script chuẩn hóa số trong dữ liệu thô.
- Thư mục `archive/`: lưu các snapshot dữ liệu lịch sử phục vụ đối chiếu.

## Sử dụng

- Quá trình tiền xử lý & tách tập đã được tự động hóa trong pipeline, không cần chỉnh sửa thủ công.
- Khi cần tái sinh bộ processed, chạy các script tương ứng tại `src/data/processed/`.

_Last updated: 2025-10-30_
