# Chiến Lược Trả Lời Câu Hỏi Về Thiết Kế & Implementation

## Tổng Quan

Khi nhận được các câu hỏi về **"Tại sao lại làm như này?"**, cần có chiến lược trả lời có cấu trúc, dựa trên hiểu biết sâu về codebase và rationale đằng sau các quyết định.

---

## 1. Câu Hỏi Về Thuật Toán / Design Decisions

### Pattern: "Tại sao thuật toán lại như này như kia?"

#### Chiến Lược Trả Lời:

1. **Xác định Context & Component**
   - Tìm component/module liên quan trong codebase
   - Xác định vị trí trong architecture (xem `ARCHITECTURE_ANALYSIS.md`)
   - Hiểu mục đích và responsibility của component đó

2. **Giải Thích Rationale**
   - **Model-First Hybrid Approach**: Giải thích tại sao chọn model-first thay vì rule-first
     - Model có độ chính xác cao hơn với dữ liệu đã train
     - Reasoning engine chỉ làm validation/enhancement, không override
     - Giảm rủi ro khi model đã được train tốt
   
   - **Entity Contracts**: Giải thích tại sao cần whitelist filtering
     - Đảm bảo output schema nhất quán cho frontend
     - Tránh entity leakage giữa các commands
     - Validate required entities theo từng intent

3. **Tham Chiếu Code Thực Tế**
   - Trích dẫn file và line number cụ thể
   - Giải thích logic trong code
   - Ví dụ: `core/hybrid_system.py:956-1024` - hybrid decision logic

4. **Trade-offs & Alternatives**
   - So sánh với các approach khác
   - Giải thích tại sao chọn approach này
   - Đề cập limitations nếu có

#### Ví Dụ Trả Lời:

**Q: "Tại sao lại dùng Model-First thay vì Rule-First?"**

**A:**
```
Hệ thống sử dụng Model-First Hybrid Approach vì:

1. **Model đã được train tốt**: PhoBERT multi-task model đã được train 
   trên dataset lớn và có độ chính xác cao (xem `models/phobert_multitask/`)

2. **Reasoning engine làm validation**: Reasoning engine chỉ validate và 
   enhance kết quả từ model, không override khi model confidence cao 
   (xem `core/hybrid_system.py:979-995`)

3. **Giảm rủi ro**: Model-first đảm bảo kết quả ổn định, chỉ can thiệp 
   khi thực sự cần thiết (low confidence, missing entities)

4. **Performance**: Model inference nhanh hơn rule-based matching cho 
   các case thông thường

Trade-off: Có thể miss một số edge cases mà rule-based sẽ catch tốt hơn,
nhưng được bù đắp bởi reasoning engine validation.
```

---

## 2. Câu Hỏi Về Kết Quả / Output

### Pattern: "Tại sao kết quả lại như này như kia?"

#### Chiến Lược Trả Lời:

1. **Trace Execution Flow**
   - Xác định input → processing steps → output
   - Trace qua các layers: Model → Reasoning → Hybrid Decision → Post-processing
   - Xem `core/hybrid_system.py:predict()` method

2. **Giải Thích Decision Logic**
   - Confidence thresholds và khi nào override
   - Entity extraction và filtering logic
   - Heuristic overrides (video call, set-alarm, etc.)

3. **Tham Chiếu Configuration**
   - Entity contracts và whitelist rules
   - Context rules trong reasoning engine
   - Normalization mappings

4. **Debug Information**
   - Kiểm tra `decision_reason` trong output
   - Xem `model_result` vs `reasoning_result`
   - Kiểm tra entity clarity score

#### Ví Dụ Trả Lời:

**Q: "Tại sao intent là 'send-mess' nhưng không có MESSAGE entity?"**

**A:**
```
Kết quả này có thể do:

1. **Model prediction**: Model có thể predict đúng intent nhưng extract 
   entity không đầy đủ (xem `core/hybrid_system.py:677-763`)

2. **Entity filtering**: Entity contracts có thể filter out MESSAGE nếu 
   không đạt quality threshold (xem `core/entity_contracts.py:filter_entities()`)

3. **Specialized extractor**: Specialized extractor có thể không tìm thấy 
   MESSAGE với confidence >= 0.8 (xem `core/hybrid_system.py:589-607`)

4. **Check decision_reason**: Xem field `decision_reason` trong output để 
   hiểu tại sao hệ thống quyết định như vậy

Để debug:
- Kiểm tra `model_result.entities` vs `final_result.entities`
- Xem `entity_clarity_score` - nếu < 0.7 thì có vấn đề
- Check logs trong `reasoning_engine.log` để xem reasoning engine extract 
  được gì
```

---

## 3. Câu Hỏi Về Training Strategy

### Pattern: "Tại sao lại training mô hình như này?"

#### Chiến Lược Trả Lời:

1. **Architecture Choice**
   - Multi-task learning: Intent + Command + NER cùng lúc
   - PhoBERT-large: Pre-trained model cho tiếng Việt
   - Shared encoder với task-specific heads

2. **Training Configuration**
   - Batch size, learning rate, epochs
   - Gradient accumulation cho GPU memory
   - Freeze encoder epochs strategy
   - Class weights cho imbalanced data

3. **Data Strategy**
   - Data augmentation techniques
   - Train/val/test split
   - Subset sampling cho quick experiments

4. **Loss Function & Metrics**
   - Multi-task loss combination
   - Entity class weights
   - Evaluation metrics

#### Ví Dụ Trả Lời:

**Q: "Tại sao lại dùng multi-task learning thay vì train riêng từng task?"**

**A:**
```
Multi-task learning được chọn vì:

1. **Shared representations**: Intent, Command, và NER có liên quan chặt chẽ
   - Intent "call" thường có entity "person"
   - Command "send-mess" cần extract "RECEIVER" và "MESSAGE"
   - Shared encoder học được features chung tốt hơn

2. **Data efficiency**: Tận dụng tối đa training data, không cần train 
   3 models riêng biệt (xem `src/models/base/multitask_model.py`)

3. **Consistency**: Đảm bảo predictions giữa các tasks nhất quán
   - Intent "call" → Command "call" → Entity "CONTACT_NAME"

4. **Performance**: Một model inference nhanh hơn 3 models riêng

Trade-off: Có thể compromise một chút accuracy cho từng task riêng lẻ,
nhưng overall system performance tốt hơn và consistent hơn.

Xem training config: `src/training/configs/config.py`
```

---

## 4. Framework Trả Lời Tổng Quát

### Template Trả Lời:

```
1. **Context**: [Component/Module nào, ở đâu trong architecture]

2. **Rationale**: [Tại sao quyết định này được đưa ra]
   - Primary reason
   - Secondary reasons
   - Trade-offs considered

3. **Implementation**: [Code thực tế như thế nào]
   - File: line numbers
   - Key logic
   - Configuration

4. **Alternatives**: [Các approach khác đã được xem xét]
   - Why not chosen
   - When might be better

5. **References**: [Tài liệu liên quan]
   - ARCHITECTURE_ANALYSIS.md
   - Code comments
   - Training configs
```

---

## 5. Best Practices

### ✅ Nên Làm:

1. **Luôn tham chiếu code thực tế**
   - File path + line numbers
   - Method/function names
   - Configuration values

2. **Giải thích trade-offs**
   - Không có solution perfect
   - Mỗi quyết định có pros/cons

3. **Cung cấp context**
   - Vị trí trong architecture
   - Mục đích của component
   - Dependencies và interactions

4. **Đề xuất improvements nếu phù hợp**
   - Nếu có limitations rõ ràng
   - Nếu có better alternatives
   - Nhưng phải justify

### ❌ Không Nên:

1. **Đoán mò**
   - Nếu không chắc, nên search codebase trước
   - Không đưa ra lý do không có trong code

2. **Quá technical mà thiếu context**
   - Giải thích phải dễ hiểu
   - Cung cấp background khi cần

3. **Phủ nhận design decisions**
   - Trừ khi có bug rõ ràng
   - Respect original design intent
   - Suggest improvements, don't criticize

---

## 6. Resources Để Tham Khảo

Khi trả lời, nên tham chiếu:

1. **`core/ARCHITECTURE_ANALYSIS.md`**: Tổng quan architecture
2. **`core/hybrid_system.py`**: Main hybrid system logic
3. **`core/reasoning_engine.py`**: Reasoning engine implementation
4. **`core/entity_contracts.py`**: Entity validation rules
5. **`src/training/configs/config.py`**: Training configuration
6. **`src/training/scripts/train_gpu.py`**: Training script
7. **`models/phobert_multitask/`**: Trained model checkpoints

---

## 7. Ví Dụ Câu Hỏi Thường Gặp

### Q1: "Tại sao confidence threshold là 0.7?"

**A:** 
```
Threshold 0.7 được chọn dựa trên:
- Model performance trên validation set
- Balance giữa accuracy và coverage
- Xem `core/hybrid_system.py:979` - high confidence >= 0.7
- Có thể adjust trong config nếu cần
```

### Q2: "Tại sao entity extractor có nhiều layers?"

**A:**
```
Specialized entity extractor có nhiều layers vì:
- Model prediction → Reasoning extraction → Specialized extraction
- Mỗi layer bù đắp limitations của layer trước
- Xem `core/hybrid_system.py:970-1027` - entity enhancement flow
```

### Q3: "Tại sao training dùng gradient accumulation?"

**A:**
```
Gradient accumulation cho phép:
- Train với effective batch size lớn hơn GPU memory limit
- Xem `src/training/scripts/train_gpu.py:317` - grad_accum parameter
- Useful cho PhoBERT-large với limited GPU memory
```

### Q4: "Tại sao quyết định define giá trị loss weighting bằng này?" (0.45, 0.25, 0.2)

**A:**
```
Giá trị loss weighting được quyết định dựa trên tầm quan trọng nghiệp vụ và độ khó của từng task:

1. **Intent (0.45) - Quan trọng nhất:**
   - Intent Classification là nền tảng của hệ thống
   - Nếu Intent sai → Entity và Command cũng sẽ sai
   - Ảnh hưởng trực tiếp đến user experience
   - Xem `src/training/configs/config.py:32` - LAMBDA_INTENT = 0.45

2. **Entity (0.25) - Quan trọng vừa:**
   - Entity loss thường lớn hơn Intent loss (nhiều tokens, nhiều labels IOB2)
   - Nếu không có weighting → Entity loss sẽ "lấn át" Intent loss
   - Entity có thể được bổ sung bởi rule-based system trong hybrid system
   - Xem `src/training/configs/config.py:33` - LAMBDA_ENTITY = 0.25

3. **Command (0.2) - Quan trọng ít hơn:**
   - Command phụ thuộc vào Intent (thường map 1-1)
   - Command có thể được suy luận từ Intent
   - Command có ít labels hơn, dễ học hơn
   - Xem `src/training/configs/config.py:34` - LAMBDA_COMMAND = 0.2

**Quy trình quyết định:**
- Bắt đầu với giá trị đều nhau (0.33, 0.33, 0.33)
- Quan sát loss values và điều chỉnh dựa trên business priority
- Test và validate trên validation set
- Fine-tune dựa trên kết quả

**Tổng không bằng 1.0 (0.9):**
- Đây là relative weights, không phải probability
- Cho phép điều chỉnh linh hoạt
- Optimizer sẽ tự điều chỉnh learning rate

Xem chi tiết: `src/training/LOSS_WEIGHTING_RATIONALE.md`
```

---

## Kết Luận

Chiến lược trả lời hiệu quả = **Context + Rationale + Code Reference + Trade-offs**

Luôn nhớ: Mỗi design decision đều có lý do. Nhiệm vụ là tìm hiểu và giải thích rõ ràng, không phải phán xét.
