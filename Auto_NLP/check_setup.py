#!/usr/bin/env python3
"""
Script kiểm tra nhanh setup dự án
"""

import os
import sys
import json

def check_python_version():
    """Kiểm tra Python version"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 8

def check_packages():
    """Kiểm tra packages"""
    print("📦 Packages:")
    
    packages = ['torch', 'transformers', 'sklearn', 'seqeval', 'tqdm', 'numpy', 'regex']
    all_ok = True
    
    for pkg in packages:
        try:
            if pkg == 'sklearn':
                import sklearn
                print(f"  ✅ {pkg}: OK")
            else:
                __import__(pkg)
                print(f"  ✅ {pkg}: OK")
        except ImportError:
            print(f"  ❌ {pkg}: MISSING")
            all_ok = False
    
    return all_ok

def check_files():
    """Kiểm tra files"""
    print("📁 Files:")
    
    files = [
        'src/training/scripts/train_gpu.py',
        'src/training/configs/config.py',
        'src/data/processed/train.json',
        'src/data/processed/val.json',
        'src/data/processed/test.json',
        'setup_new_machine.py'
    ]
    
    all_ok = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}: OK")
        else:
            print(f"  ❌ {file}: MISSING")
            all_ok = False
    
    return all_ok

def check_model_cache():
    """Kiểm tra model cache"""
    print("🤖 Model Cache:")
    
    cache_path = 'model_cache/models--vinai--phobert-large'
    if os.path.exists(cache_path):
        print(f"  ✅ {cache_path}: OK")
        return True
    else:
        print(f"  ❌ {cache_path}: MISSING")
        return False

def check_data():
    """Kiểm tra data"""
    print("📊 Data:")
    
    try:
        with open('src/data/processed/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"  ✅ train.json: {len(train_data)} samples")
        
        with open('src/data/processed/val.json', 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"  ✅ val.json: {len(val_data)} samples")
        
        return True
    except Exception as e:
        print(f"  ❌ Data error: {e}")
        return False

def check_cuda():
    """Kiểm tra CUDA"""
    print("🎮 CUDA:")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
        return True
    except Exception as e:
        print(f"  ❌ CUDA error: {e}")
        return False

def main():
    print("🔍 Kiểm tra setup dự án Auto_NLP...")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Packages", check_packages),
        ("Files", check_files),
        ("Model Cache", check_model_cache),
        ("Data", check_data),
        ("CUDA", check_cuda)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 50)
    print("📋 Tổng kết:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup hoàn hảo! Có thể chạy training:")
        print("   python src/training/scripts/train_gpu.py")
    else:
        print("⚠️ Setup chưa hoàn chỉnh. Chạy:")
        print("   python setup_new_machine.py")

if __name__ == "__main__":
    main()
