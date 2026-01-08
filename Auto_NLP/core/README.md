# Core Module - Hybrid NLP System

## 📦 Overview

Folder `core/` chứa các component chính của **Model-First Hybrid NLP System**, kết hợp trained model với rule-based reasoning để tạo ra hệ thống NLP robust và accurate.

## 🗂️ File Structure

```
core/
├── hybrid_system.py          # Main orchestrator (69KB, 1491 lines)
├── reasoning_engine.py       # Rule-based reasoning engine (91KB, 1880 lines)
├── model_loader.py           # Model loading wrapper (7.1KB, 194 lines)
├── entity_contracts.py       # Entity validation contracts (4.8KB, 158 lines)
├── semantic_patterns.json    # Regex patterns for intent detection
├── knowledge_base.json       # Knowledge base for semantic understanding
├── intent_fallback.json      # Fallback rules for low confidence
├── context_rules.json        # Multi-turn conversation rules
└── ARCHITECTURE_ANALYSIS.md  # Detailed architecture analysis
```

## 🎯 Core Components

### 1. **ModelFirstHybridSystem** (`hybrid_system.py`)
**Orchestrator chính** - điều phối toàn bộ hệ thống

**Responsibilities:**
- Load và quản lý trained model (PRIMARY)
- Initialize reasoning engine (SECONDARY)
- Decision logic: khi nào dùng model, khi nào dùng reasoning
- Entity enhancement: kết hợp entities từ nhiều nguồn
- Post-processing: làm sạch và chuẩn hóa output
- Heuristic overrides: xử lý edge cases

**Key Methods:**
- `predict(text)` - Main prediction method
- `_make_hybrid_decision()` - Decision logic
- `_enhance_entities()` - Entity enhancement
- `_apply_heuristic_overrides()` - Special rules

---

### 2. **ReasoningEngine** (`reasoning_engine.py`)
**Rule-based reasoning engine** với semantic understanding

**Features:**
- Semantic similarity với PhoBERT embeddings
- Fuzzy matching với rapidfuzz
- FAISS-based vector search
- Pattern matching với regex
- Knowledge base integration
- Multi-turn conversation context

**Components:**
- `ReasoningCache` - Cache embeddings và results
- `FuzzyMatcher` - Fuzzy keyword matching
- `VectorStore` - FAISS semantic search
- `EntityExtractor` - Rule-based entity extraction

---

### 3. **TrainedModelInference** (`model_loader.py`)
**Wrapper** để load và sử dụng trained model

**Features:**
- Load trained PhoBERT multi-task model
- Entity cleaning và normalization
- Platform whitelist filtering
- Message/Query merging
- Output format standardization

**Entity Processing:**
- Remove special tokens (`<s>`, `</s>`, `[PAD]`)
- Platform whitelist (Zalo, Messenger, Facebook)
- Merge multiple MESSAGE/QUERY spans
- Filter trigger verbs
- Select best entity spans

---

### 4. **Entity Contracts** (`entity_contracts.py`)
**Validation layer** - đảm bảo output quality

**Features:**
- Entity whitelist per intent
- Required entities validation
- Entity filtering
- Clarity score calculation

**Key Functions:**
- `filter_entities()` - Chỉ giữ entities hợp lệ
- `validate_entities()` - Kiểm tra required entities
- `calculate_entity_clarity_score()` - Tính điểm chất lượng (0-1)

**Entity Whitelist:**
```python
ENTITY_WHITELIST = {
    "send-mess": {
        "required": ["MESSAGE", "RECEIVER"],
        "optional": ["PLATFORM"]
    },
    "call": {
        "required": ["RECEIVER"],
        "optional": ["CONTACT_NAME", "PHONE"]
    },
    # ... cho tất cả 10 intents
}
```

---

## 📊 Processing Flow

```
User Input
    ↓
[ModelFirstHybridSystem]
    ↓
┌─────────────────────────┐
│ 1. Model Prediction     │ ← PRIMARY (TrainedModelInference)
│    - Intent + Entities  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 2. Reasoning Validation │ ← SECONDARY (ReasoningEngine)
│    - Semantic similarity │
│    - Pattern matching   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 3. Entity Enhancement   │ ← SPECIALIZED (EntityExtractor)
│    - Merge entities     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 4. Contract Validation  │ ← VALIDATOR (EntityContracts)
│    - Whitelist filter   │
│    - Required check     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 5. Heuristic Overrides  │ ← RULES (Special cases)
│    - Video call rules   │
│    - Communication guard│
└─────────────────────────┘
    ↓
Final Output (Validated & Enhanced)
```

## 🎯 Design Principles

1. **Model-First**: Trained model là PRIMARY, reasoning là SECONDARY
2. **Enhancement, Not Override**: Reasoning enhance, không override model
3. **Validation Layers**: Nhiều lớp validation (contracts, whitelist, required)
4. **Quality Assurance**: Contracts đảm bảo output quality
5. **Flexibility**: Heuristic overrides cho edge cases

## 🔗 Dependencies

```
hybrid_system.py
    ├── model_loader.py (TrainedModelInference)
    ├── reasoning_engine.py (ReasoningEngine)
    ├── entity_contracts.py (filter_entities, validate_entities)
    └── src.inference.engines.entity_extractor (SpecializedEntityExtractor)

reasoning_engine.py
    ├── semantic_patterns.json (Regex patterns)
    ├── knowledge_base.json (Knowledge base)
    └── context_rules.json (Context rules)

hybrid_system.py
    └── intent_fallback.json (Fallback rules)
```

## 📈 Performance

- **Model-First Strategy**: Fast với fallback robust
- **Caching**: ReasoningCache giảm computation time
- **Entity Processing**: Cleaning và merging tối ưu

## 🚀 Usage

```python
from core.hybrid_system import ModelFirstHybridSystem

# Initialize system
hybrid_system = ModelFirstHybridSystem()

# Predict
result = hybrid_system.predict("gửi tin nhắn cho mẹ")

# Result structure:
{
    "intent": "send-mess",
    "command": "send-mess",
    "confidence": 0.95,
    "entities": {
        "MESSAGE": "...",
        "RECEIVER": "mẹ",
        "PLATFORM": "zalo"
    },
    "method": "hybrid",
    "decision_reason": "...",
    "entity_clarity_score": 0.9
}
```

## 📚 Documentation

- **ARCHITECTURE_ANALYSIS.md**: Detailed architecture analysis với diagrams và explanations

## 🔍 Key Insights

**Tại sao Hybrid?**
- Model có thể miss entities, sai intent
- Rules bù đắp model weaknesses
- Contracts đảm bảo output đúng format

**Tại sao Model-First?**
- Trained model tốt nhất (đã train trên dataset lớn)
- Reasoning chỉ enhance, không override
- Performance: Model nhanh hơn reasoning

**Tại sao Contracts?**
- Output validation: Đảm bảo đúng format
- Entity filtering: Chỉ giữ entities hợp lệ
- Quality scoring: Đánh giá chất lượng output
