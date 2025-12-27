# Báo cáo Đánh giá Mô hình

## 1. Bối cảnh và câu hỏi nghiên cứu
- Hệ trợ lý ngôn ngữ tiếng Việt cần nhận diện Intent, NER và trả lệnh chính xác.
- Mục tiêu: mô hình tổng quát hoá tốt, độ trễ thấp, ổn định qua nhiều tình huống.

## 2. Quy trình Evaluation chuẩn
![evaluation_flow](evaluation_flow.png)

- **Three-way Holdout**: train (60%), validation (20%), test (20%).
- **Model Selection**: chạy PhoBERT-base/large, XLM-R-base, ViT5-base trên train + validation.
- **Model Evaluation**: chỉ dùng test một lần sau khi chốt mô hình & hyperparameters.
- **Kiểm tra chéo (CV)**: lưu scoring từng fold để ước lượng độ ổn định.

## 3. Thước đo và lý do chọn
- **Intent Accuracy, Macro/Weighted F1**: giữ cân bằng class mất cân đối.
- **NER micro F1 (seqeval)**: đo chất lượng chuỗi entity.
- **Per-class F1 + confusion matrix**: định vị lớp khó.
- **Latency (batch=1, max_len=16/32/64)**: đảm bảo đáp ứng thời gian thực.
- **Throughput (batch=8)**: đánh giá suy luận song song.
- **VRAM peak, số tham số, hidden size**: đánh đổi tài nguyên.
- **Std/CI 95% qua 5 folds**: chứng minh ổn định.
- **Epoch-to-best, thời gian/epoch**: đảm bảo early stopping hợp lý.

## 4. Kết quả tổng hợp
| Mô hình | Intent Weighted F1 | NER micro F1 | Latency@64 (ms) | Throughput@8 (câu/s) | VRAM peak (GB) | Params | Hidden |
|---------|--------------------|--------------|-----------------|----------------------|----------------|--------|--------|
| PhoBERT-base | 0.912 | 0.884 | 28.5 | 142 | 1.9 | ≈135M | 768 |
| PhoBERT-large | 0.934 | 0.903 | 41.2 | 108 | 3.4 | ≈355M | 1024 |
| XLM-R-base | 0.896 | 0.861 | 35.7 | 126 | 2.6 | ≈278M | 768 |
| ViT5-base | 0.874 | 0.842 | 52.4 | 95 | 4.1 | ≈220M | 768 |

![benchmark_metrics](benchmark_metrics.png)

## 5. Ổn định & hội tụ
- Boxplot epoch-best (5-fold) + đường thời gian/epoch chứng minh mô hình lựa chọn hội tụ quanh 9–12 epoch.
- CI 95% của F1 không chồng chéo giữa PhoBERT-large và các mô hình khác → khác biệt đáng kể.

![training_dynamics](training_dynamics.png)

## 6. Tối ưu hoá vận hành
- PhoBERT-large cần VRAM ≈3.4 GB và latency cao hơn base → cân nhắc server GPU.
- ViT5-base encoder-decoder → latency cao hơn rõ rệt, phù hợp tác vụ sinh mà không phải intent chính.
- Đề xuất runtime: PhoBERT-base cho edge; PhoBERT-large cho backend GPU tối ưu.

## 7. Checklist tuân thủ
- [x] Test set chỉ dùng một lần (sau khi chốt model).
- [x] Có cross-validation để chọn hyperparameters.
- [x] Báo cáo both method + metric.
- [x] Lưu learning curve, confusion matrix, per-class F1 để phân tích lỗi.
- [ ] Thu thập throughput thực tế trên hạ tầng deployment.
- [ ] Ghi lại seed và cấu hình chuẩn để tái lập kết quả.

