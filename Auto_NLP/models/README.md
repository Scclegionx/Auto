# Trained Models

## Structure
- `trained/` - Trained model files (.pth)
- `tokenizers/` - Tokenizer files
- `configs/` - Model configs and metadata
- `logs/` - Training logs
- `exports/` - Exported models for deployment

## Usage
```python
# Load trained model
model_path = "models/trained/phobert_large_intent_model/model_best.pth"
model = load_trained_model(model_path)
```
