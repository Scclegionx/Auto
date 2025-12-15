# Models Package

## Structure
- `base/` - Định nghĩa mô hình cốt lõi (`MultiTaskModel` cho intent/entity/command).
- `inference/` - Tiện ích load checkpoint và suy luận (`load_multi_task_model`).
- `utils/` - Helper phục vụ huấn luyện/đánh giá.
- `checkpoints/` - Placeholder lưu checkpoint (nếu cần).

## Usage
```python
from models.base import MultiTaskModel
from models.inference.model_loader import load_multi_task_model
```
