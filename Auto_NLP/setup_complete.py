#!/usr/bin/env python3
"""
Script setup hoàn chỉnh cho người mới clone dự án về máy mới
Giải quyết tất cả vấn đề thường gặp khi setup lần đầu
"""

import os
import sys
import subprocess
import shutil
import warnings
from pathlib import Path

def print_step(step, description):
    """Print step with formatting"""
    print(f"\n{'='*60}")
    print(f"BƯỚC {step}: {description}")
    print('='*60)

def run_command(cmd, description=""):
    """Run command with error handling"""
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Thành công: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi: {description}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print_step(1, "KIỂM TRA PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Cần Python 3.8+ để chạy dự án")
        return False
    
    print("✅ Python version OK")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print_step(2, "TẠO VIRTUAL ENVIRONMENT")
    
    venv_path = Path("venv_new")
    
    if venv_path.exists():
        print("🗑️ Xóa venv cũ...")
        shutil.rmtree(venv_path)
    
    print("📦 Tạo venv mới...")
    if not run_command("python -m venv venv_new", "Tạo virtual environment"):
        return False
    
    return True

def install_packages():
    """Install required packages"""
    print_step(3, "CÀI ĐẶT PACKAGES")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    packages = [
        "torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121",
        "transformers>=4.20.0",
        "scikit-learn>=1.0.0", 
        "seqeval>=1.2.0",
        "tqdm>=4.60.0",
        "numpy>=1.21.0",
        "regex>=2021.0.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0"
    ]
    
    for package in packages:
        cmd = f"venv_new\\Scripts\\pip install {package}"
        if not run_command(cmd, f"Cài đặt {package.split()[0]}"):
            return False
    
    return True

def clear_model_cache():
    """Clear model cache"""
    print_step(4, "XÓA MODEL CACHE CŨ")
    
    cache_paths = [
        Path.home() / ".cache" / "huggingface",
        Path("model_cache"),
        Path("models")
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"🗑️ Xóa cache: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
    
    return True

def download_models():
    """Download models"""
    print_step(5, "TẢI MODEL PHOBERT-LARGE")
    
    # Create test script
    test_script = """
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    print("📥 Đang tải PhoBERT-large...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-large",
        cache_dir="model_cache",
        force_download=True
    )
    
    # Download model  
    model = AutoModel.from_pretrained(
        "vinai/phobert-large", 
        cache_dir="model_cache",
        force_download=True
    )
    
    # Test loading
    test_text = "Xin chào"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    
    print("✅ Model tải thành công!")
    print(f"   - Tokenizer: {type(tokenizer).__name__}")
    print(f"   - Model: {type(model).__name__}")
    print(f"   - Output shape: {outputs.last_hidden_state.shape}")
    
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    exit(1)
"""
    
    # Write and run test script
    with open("test_model_download.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    success = run_command("venv_new\\Scripts\\python test_model_download.py", "Tải và test model")
    
    # Cleanup
    if os.path.exists("test_model_download.py"):
        os.remove("test_model_download.py")
    
    return success

def test_training_script():
    """Test training script"""
    print_step(6, "KIỂM TRA TRAINING SCRIPT")
    
    # Check if training script exists
    train_script = Path("src/training/scripts/train_gpu.py")
    if not train_script.exists():
        print("❌ Không tìm thấy train_gpu.py")
        return False
    
    print("✅ Training script tồn tại")
    
    # Test imports
    test_imports = """
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    print("✅ All imports OK")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)
"""
    
    with open("test_imports.py", "w", encoding="utf-8") as f:
        f.write(test_imports)
    
    success = run_command("venv_new\\Scripts\\python test_imports.py", "Test imports")
    
    # Cleanup
    if os.path.exists("test_imports.py"):
        os.remove("test_imports.py")
    
    return success

def create_setup_guide():
    """Create setup guide"""
    print_step(7, "TẠO HƯỚNG DẪN SETUP")
    
    guide_content = """# HƯỚNG DẪN SETUP CHO NGƯỜI MỚI

## 🚀 Setup tự động (Khuyến nghị)
```bash
python setup_complete.py
```

## 📋 Setup thủ công

### 1. Tạo virtual environment
```bash
python -m venv venv_new
venv_new\\Scripts\\activate
```

### 2. Cài đặt packages
```bash
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.20.0 scikit-learn>=1.0.0 seqeval>=1.2.0 tqdm>=4.60.0 numpy>=1.21.0 regex>=2021.0.0 fastapi>=0.70.0 uvicorn>=0.15.0 pydantic>=2.0.0
```

### 3. Tải model
```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large'); AutoModel.from_pretrained('vinai/phobert-large')"
```

### 4. Training
```bash
python src/training/scripts/train_gpu.py
```

## 🔧 Troubleshooting

### Lỗi: 'NoneType' object has no attribute 'endswith'
```bash
# Xóa cache và tải lại
rmdir /s model_cache
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large', force_download=True); AutoModel.from_pretrained('vinai/phobert-large', force_download=True)"
```

### Lỗi import trong IDE
1. Chọn interpreter: `.\venv_new\Scripts\python.exe`
2. Restart IDE
3. Chờ IDE index packages

### Lỗi CUDA
- Đảm bảo có GPU NVIDIA
- Cài đặt CUDA toolkit
- Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`

## 📞 Hỗ trợ
Nếu gặp vấn đề, chạy:
```bash
python setup_complete.py
```
"""
    
    with open("SETUP_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("✅ Đã tạo SETUP_GUIDE.md")
    return True

def main():
    """Main setup function"""
    print("🚀 AUTO NLP - SETUP CHO NGƯỜI MỚI")
    print("=" * 60)
    print("Script này sẽ setup hoàn chỉnh dự án cho người mới clone về")
    print("=" * 60)
    
    steps = [
        check_python_version,
        create_virtual_environment, 
        install_packages,
        clear_model_cache,
        download_models,
        test_training_script,
        create_setup_guide
    ]
    
    for i, step_func in enumerate(steps, 1):
        if not step_func():
            print(f"\n❌ BƯỚC {i} THẤT BẠI!")
            print("Vui lòng kiểm tra lỗi và thử lại")
            return False
    
    print("\n" + "="*60)
    print("🎉 SETUP HOÀN THÀNH THÀNH CÔNG!")
    print("="*60)
    print("\n📋 BƯỚC TIẾP THEO:")
    print("1. Activate virtual environment:")
    print("   venv_new\\Scripts\\activate")
    print("\n2. Chạy training:")
    print("   python src/training/scripts/train_gpu.py")
    print("\n3. Hoặc chạy API:")
    print("   python api/server.py")
    print("\n📖 Xem hướng dẫn chi tiết: SETUP_GUIDE.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)