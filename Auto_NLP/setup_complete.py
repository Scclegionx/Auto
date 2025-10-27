#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import warnings
from pathlib import Path

def print_step(step, description):
    print(f"\n{'='*60}")
    print(f"BƯỚC {step}: {description}")
    print('='*60)

def run_command(cmd, description="", ignore_errors=False):
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Thành công: {description}")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️ Bỏ qua: {description}")
            return True
        else:
            print(f"❌ Lỗi: {description}")
            print(f"   Error: {e.stderr}")
            return False

def check_python_version():
    print_step(1, "KIỂM TRA PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Cần Python 3.8+ để chạy dự án")
        return False
    
    print("✅ Python version OK")
    return True

def create_virtual_environment():
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
    print_step(3, "CÀI ĐẶT PACKAGES")
    
    warnings.filterwarnings("ignore")
    
    # Uninstall packages cũ trước
    print("🗑️ Uninstall packages cũ...")
    old_packages = ["torch", "transformers", "scikit-learn", "seqeval", "tqdm", "numpy", "regex", "fastapi", "uvicorn", "pydantic"]
    
    for package in old_packages:
        cmd = f"venv_new\\Scripts\\pip uninstall {package} -y"
        run_command(cmd, f"Uninstall {package}", ignore_errors=True)
    
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
    print_step(5, "TẢI MODEL PHOBERT-LARGE")
    
    test_script = """
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    print("📥 Đang tải PhoBERT-large...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-large",
        cache_dir="model_cache",
        force_download=True
    )
    
    model = AutoModel.from_pretrained(
        "vinai/phobert-large", 
        cache_dir="model_cache",
        force_download=True
    )
    
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
    
    with open("test_model_download.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    success = run_command("venv_new\\Scripts\\python test_model_download.py", "Tải và test model")
    
    if os.path.exists("test_model_download.py"):
        os.remove("test_model_download.py")
    
    return success

def test_training_script():
    print_step(6, "KIỂM TRA TRAINING SCRIPT")
    
    train_script = Path("src/training/scripts/train_gpu.py")
    if not train_script.exists():
        print("❌ Không tìm thấy train_gpu.py")
        return False
    
    print("✅ Training script tồn tại")
    
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
    
    if os.path.exists("test_imports.py"):
        os.remove("test_imports.py")
    
    return success

def main():
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
        test_training_script
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
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)