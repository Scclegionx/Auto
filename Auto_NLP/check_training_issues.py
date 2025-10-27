#!/usr/bin/env python3
"""
Script để kiểm tra và sửa các vấn đề khi clone dự án về máy khác
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("🐍 Kiểm tra phiên bản Python...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python version quá cũ! Cần Python 3.8+")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_required_packages():
    """Kiểm tra các package cần thiết"""
    print("\n📦 Kiểm tra các package cần thiết...")
    
    required_packages = [
        'torch',
        'transformers', 
        'sklearn',
        'seqeval',
        'tqdm',
        'numpy',
        'regex'  # Package này có thể gây vấn đề
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"✅ {package}: OK")
            elif package == 'regex':
                import regex
                print(f"✅ {package}: OK")
            else:
                __import__(package)
                print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Thiếu packages: {missing_packages}")
        print("Chạy lệnh sau để cài đặt:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ Tất cả packages đã có")
        return True

def check_file_structure():
    """Kiểm tra cấu trúc file"""
    print("\n📁 Kiểm tra cấu trúc file...")
    
    required_files = [
        'src/training/scripts/train_gpu.py',
        'src/training/configs/config.py',
        'src/data/processed/train.json',
        'src/data/processed/val.json',
        'src/data/processed/test.json',
        'models/label_maps.json'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: OK")
        else:
            print(f"❌ {file_path}: MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Thiếu files: {missing_files}")
        return False
    else:
        print("✅ Tất cả files cần thiết đã có")
        return True

def check_model_cache():
    """Kiểm tra model cache"""
    print("\n🤖 Kiểm tra model cache...")
    
    cache_paths = [
        'model_cache',
        'model_cache/models--vinai--phobert-base',
        'model_cache/models--vinai--phobert-large'
    ]
    
    missing_cache = []
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            print(f"✅ {cache_path}: OK")
        else:
            print(f"❌ {cache_path}: MISSING")
            missing_cache.append(cache_path)
    
    if missing_cache:
        print(f"\n❌ Thiếu model cache: {missing_cache}")
        print("Cần tải model cache hoặc cho phép download online")
        return False
    else:
        print("✅ Model cache OK")
        return True

def check_data_files():
    """Kiểm tra data files"""
    print("\n📊 Kiểm tra data files...")
    
    data_files = [
        'src/data/processed/train.json',
        'src/data/processed/val.json', 
        'src/data/processed/test.json'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ {data_file}: OK ({len(data)} samples)")
            except Exception as e:
                print(f"❌ {data_file}: CORRUPTED - {e}")
                return False
        else:
            print(f"❌ {data_file}: MISSING")
            return False
    
    print("✅ Data files OK")
    return True

def fix_import_issues():
    """Sửa các vấn đề import"""
    print("\n🔧 Sửa các vấn đề import...")
    
    train_gpu_path = 'src/training/scripts/train_gpu.py'
    
    if not os.path.exists(train_gpu_path):
        print(f"❌ Không tìm thấy {train_gpu_path}")
        return False
    
    # Đọc file
    with open(train_gpu_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sửa import regex thành re
    if 'import regex' in content:
        content = content.replace('import regex', 'import re as regex')
        print("✅ Đã sửa import regex")
    
    # Sửa đường dẫn model cache
    if '../../model_cache' in content:
        content = content.replace('../../model_cache', 'model_cache')
        print("✅ Đã sửa đường dẫn model cache")
    
    # Ghi lại file
    with open(train_gpu_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Đã sửa các vấn đề import")
    return True

def create_setup_script():
    """Tạo script setup cho máy mới"""
    print("\n📝 Tạo script setup...")
    
    setup_content = '''#!/usr/bin/env python3
"""
Script setup cho máy mới
"""

import subprocess
import sys
import os

def install_requirements():
    """Cài đặt requirements"""
    print("📦 Cài đặt requirements...")
    
    requirements = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'scikit-learn>=1.0.0',
        'seqeval>=1.2.0',
        'tqdm>=4.60.0',
        'numpy>=1.21.0',
        'regex>=2021.0.0',
        'fastapi>=0.70.0',
        'uvicorn>=0.15.0'
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✅ Đã cài {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Lỗi cài {req}")

def download_models():
    """Tải models"""
    print("🤖 Tải models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Tải PhoBERT base
        print("Tải PhoBERT base...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModel.from_pretrained("vinai/phobert-base")
        
        # Tải PhoBERT large
        print("Tải PhoBERT large...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        model = AutoModel.from_pretrained("vinai/phobert-large")
        
        print("✅ Đã tải models")
        
    except Exception as e:
        print(f"❌ Lỗi tải models: {e}")

def main():
    print("🚀 Setup cho máy mới...")
    
    # Tạo thư mục cần thiết
    os.makedirs('model_cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Cài đặt requirements
    install_requirements()
    
    # Tải models
    download_models()
    
    print("✅ Setup hoàn thành!")

if __name__ == "__main__":
    main()
'''
    
    with open('setup_new_machine.py', 'w', encoding='utf-8') as f:
        f.write(setup_content)
    
    print("✅ Đã tạo setup_new_machine.py")

def main():
    """Main function"""
    print("🔍 Kiểm tra dự án Auto_NLP cho máy mới...")
    print("=" * 50)
    
    all_ok = True
    
    # Kiểm tra Python version
    if not check_python_version():
        all_ok = False
    
    # Kiểm tra packages
    if not check_required_packages():
        all_ok = False
    
    # Kiểm tra file structure
    if not check_file_structure():
        all_ok = False
    
    # Kiểm tra model cache
    if not check_model_cache():
        all_ok = False
    
    # Kiểm tra data files
    if not check_data_files():
        all_ok = False
    
    # Sửa import issues
    if not fix_import_issues():
        all_ok = False
    
    # Tạo setup script
    create_setup_script()
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Tất cả kiểm tra đều OK! Dự án sẵn sàng để training.")
    else:
        print("❌ Có một số vấn đề cần sửa. Chạy setup_new_machine.py để tự động sửa.")
    
    print("\n📋 Hướng dẫn cho máy mới:")
    print("1. Chạy: python setup_new_machine.py")
    print("2. Chạy: python src/training/scripts/train_gpu.py")
    print("3. Nếu vẫn lỗi, kiểm tra log để debug")

if __name__ == "__main__":
    main()
