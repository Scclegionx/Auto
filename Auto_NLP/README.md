## Cài Đặt

# Tạo virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Cài đặt requirements
pip install -r requirements.txt

# Chạy test toàn diện trước khi training
python test_training_complete.py

## Training

### Chạy Training
```bash
# Cách 1: Script tiện ích
python run_training.py

# Cách 2: Trực tiếp
python src/training/scripts/train_gpu.py
```

## API Server

### Chạy API
```bash
# Cách 1: Script tiện ích
python run_api.py

# Cách 2: Trực tiếp
python src/inference/api/api_server.py
# start src/inference/interfaces/web_interface.html (giao diện)
```
