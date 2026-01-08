# Dataset Report Scripts

Bộ scripts để tổng hợp và trực quan hóa dataset.

## Scripts

### 1. `aggregate_datasets.py`
Tổng hợp thông tin từ master dataset và train/val/test splits.

**Usage:**
```bash
python scripts/data/aggregate_datasets.py
```

**Output:**
- `reports/dataset_aggregate_report.json` - Báo cáo JSON chi tiết

**Tính năng:**
- Phân tích master dataset
- Phân tích train/val/test splits
- Kiểm tra consistency (master = train+val+test)
- Thống kê intent/entity distribution
- Tính toán các metrics (avg entities, sentence length, etc.)

---

### 2. `create_dataset_dashboard.py`
Tạo dashboard HTML với visualizations đẹp và dễ nhìn.

**Usage:**
```bash
python scripts/data/create_dataset_dashboard.py
```

**Output:**
- `reports/dataset_dashboard.html` - Dashboard HTML
- `reports/dashboard_*.png` - Các biểu đồ visualization

**Visualizations:**
- Intent Distribution (Bar & Pie Chart)
- Train/Val/Test Split
- Entity Type Distribution
- Sentence Length Distribution
- Entities per Sample
- Entity by Intent Heatmap

**Tính năng:**
- Dashboard HTML responsive, đẹp mắt
- Tất cả visualizations trong một trang
- Bảng thống kê chi tiết
- Consistency check

---

### 3. `generate_dataset_report.py` (Wrapper)
Chạy cả aggregate và dashboard cùng lúc.

**Usage:**
```bash
# Chạy đầy đủ
python scripts/data/generate_dataset_report.py

# Chỉ tạo dashboard (nếu đã có aggregate report)
python scripts/data/generate_dataset_report.py --skip-aggregate

# Chỉ tạo aggregate report
python scripts/data/generate_dataset_report.py --skip-dashboard
```

---

## Output Files

### JSON Report
`reports/dataset_aggregate_report.json`
- Metadata về các file dataset
- Consistency check
- Phân tích chi tiết cho master, train, val, test
- Intent/Entity distributions
- Statistics

### HTML Dashboard
`reports/dataset_dashboard.html`
- Dashboard trực quan với tất cả visualizations
- Responsive design
- Dễ xem và chia sẻ

### Visualizations
`reports/dashboard_*.png`
- `dashboard_intent_bar.png` - Intent distribution (bar)
- `dashboard_intent_pie.png` - Intent distribution (pie)
- `dashboard_split.png` - Train/Val/Test split
- `dashboard_entity_dist.png` - Entity distribution
- `dashboard_sentence_length.png` - Sentence length histogram
- `dashboard_entities_per_sample.png` - Entities per sample histogram
- `dashboard_entity_intent_heatmap.png` - Entity by intent heatmap

---

## Examples

### Tạo đầy đủ report và dashboard:
```bash
python scripts/data/generate_dataset_report.py
```

### Chỉ tạo aggregate report:
```bash
python scripts/data/aggregate_datasets.py --output reports/my_report.json
```

### Chỉ tạo dashboard với custom paths:
```bash
python scripts/data/create_dataset_dashboard.py \
    --master src/data/raw/elderly_commands_master.json \
    --train src/data/processed/train.json \
    --val src/data/processed/val.json \
    --test src/data/processed/test.json \
    --output reports/my_dashboard.html
```

---

## Requirements

- Python 3.8+
- matplotlib
- seaborn
- numpy
- json (built-in)

---

## Notes

- Tất cả scripts hỗ trợ Windows encoding (UTF-8)
- Dashboard HTML có thể mở trực tiếp trong browser
- Visualizations được lưu với DPI 150 cho chất lượng tốt
- Aggregate report có thể được sử dụng cho các analysis khác

