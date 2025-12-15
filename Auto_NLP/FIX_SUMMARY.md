# 🔧 Fix Model Collapse - Summary

## ❌ Vấn Đề Trước Đó
- Entity predictions: 100% O (không dự đoán entity)
- Loss: NaN (gradient explosion)
- Intent/Command: Collapse về add-contacts
- Root cause: Hard masking (-inf) + Lambda quá cao

---

## ✅ Các Fix Đã Áp Dụng

### 1. Soft Masking
```python
# BEFORE:
entity_logits.masked_fill(mask, float('-inf'))
→ Gây NaN trong softmax/gradient

# AFTER:
entity_logits.masked_fill(mask, -1e9)
→ Vẫn suppress invalid labels, nhưng không NaN
```

**Hiệu quả**:
- ✅ Tránh NaN trong loss
- ✅ Gradient ổn định
- ✅ Softmax vẫn gần 0 cho invalid labels

---

### 2. Lambda Cân Bằng
```python
# BEFORE:
LAMBDA_ENTITY = 0.8  (quá cao)
LAMBDA_INTENT = 0.15
LAMBDA_COMMAND = 0.05

# AFTER:
LAMBDA_ENTITY = 0.5  (vừa phải)
LAMBDA_INTENT = 0.3
LAMBDA_COMMAND = 0.2
```

**Lý do**:
- Entity loss có thể lớn ban đầu
- Lambda = 0.8 làm model chỉ focus entity
- Intent/Command bị bỏ quên → collapse

---

### 3. Focal Loss Gamma
```python
# BEFORE: gamma = 3.0 (quá mạnh)
# AFTER:  gamma = 2.0 (chuẩn)
```

**Lý do**:
- gamma = 3.0: (1-p)³ → focus cực mạnh vào hard examples
- Có thể bỏ qua easy examples quá mức
- gamma = 2.0: Cân bằng hơn (theo paper gốc)

---

## 🎯 Cấu Hình Hiện Tại

```yaml
Model:
  - Soft masking: -1e9 cho invalid I- labels
  - Invalid labels: I-PHONE, I-ACTION, I-MODE, I-PLATFORM...

Loss:
  - Entity: Focal Loss (α=0.25, γ=2.0) + class weights
  - O weight: 0.1
  - Entity weight: up to 5.0

Lambda:
  - Intent: 0.3 (30%)
  - Entity: 0.5 (50%) ← Focus chính
  - Command: 0.2 (20%)
  
Warmup:
  - Epoch 0: λ_entity = 0.3
  - Epoch 1: λ_entity = 0.4
  - Epoch 2+: λ_entity = 0.5
```

---

## 📊 Kỳ Vọng

| Metric | Trước (Collapse) | Sau (Dự kiến) |
|--------|------------------|---------------|
| **Entity F1** | 0.0 | 0.2-0.4 |
| **pred_non_O** | 0% | 20-30% |
| **Loss** | NaN | Stable (0.8-2.0) |
| **I-PHONE pred** | 0 → Collapsed | 0 (masked) |
| **Intent F1** | 0.0 (collapse) | 0.3-0.5 |

---

## ⚠️ Monitor Points

### 1. Loss Values
```
✅ Healthy:
  - Total loss: 0.5 - 2.0
  - Entity loss: 0.2 - 1.0
  - Intent/Command: 0.3 - 2.5

❌ Problems:
  - NaN: Gradient explosion
  - > 5.0: Model không học
  - Giảm đột ngột 0: Collapse
```

### 2. Prediction Distribution
```
✅ Healthy:
  - pred_non_O: 15-35%
  - Intent: Phân bố đều (5-15% mỗi class)
  - Entity: Diverse predictions

❌ Problems:
  - pred_non_O: 0% hoặc 100%
  - Intent: >80% vào 1 class
  - Entity: Chỉ predict O hoặc 1 label
```

### 3. Gradient Flow
```
✅ Healthy:
  - Grad norm: 0.1 - 2.0
  - Clipped occasionally

❌ Problems:
  - Grad norm > 10: Explosion
  - Grad norm ~ 0: Vanishing
  - Clipped mọi batch: Too aggressive
```

---

**Ready to retrain!** 🚀

