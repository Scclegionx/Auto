#!/usr/bin/env python3
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
    print("🤖 Tải PhoBERT-large model...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Chỉ tải PhoBERT large (model được sử dụng trong dự án)
        print("Tải PhoBERT-large...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        model = AutoModel.from_pretrained("vinai/phobert-large")
        
        print("✅ Đã tải PhoBERT-large model")
        
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        print("💡 Tip: Kiểm tra kết nối internet và thử lại")

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
