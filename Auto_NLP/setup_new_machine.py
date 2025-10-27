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
        'torch>=2.5.0',  # Compatible with current version
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

def check_model_cache():
    """Kiểm tra model cache"""
    print("🔍 Kiểm tra model cache...")
    
    cache_paths = [
        'model_cache',
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
        print(f"⚠️ Thiếu model cache: {missing_cache}")
        print("🔄 Sẽ tự động tải model...")
        return False
    else:
        print("✅ Model cache OK")
        return True

def download_models():
    """Tải models"""
    print("🤖 Tải PhoBERT-large model...")
    
    try:
        import warnings
        # Suppress vulnerability warnings for PyTorch < 2.6
        warnings.filterwarnings("ignore", message=".*vulnerability.*")
        
        from transformers import AutoTokenizer, AutoModel
        
        # Force download lại để đảm bảo có đầy đủ files
        print("Tải PhoBERT-large tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-large",
            force_download=True,  # Force download
            cache_dir="model_cache"
        )
        
        print("Tải PhoBERT-large model...")
        model = AutoModel.from_pretrained(
            "vinai/phobert-large",
            force_download=True,  # Force download
            cache_dir="model_cache"
        )
        
        print("✅ Đã tải PhoBERT-large model")
        
        # Test loading để đảm bảo hoạt động
        print("🧪 Test loading model...")
        test_text = "Xin chào"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Test OK: '{test_text}' -> {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        print("💡 Tip: Kiểm tra kết nối internet và thử lại")
        print("💡 Tip: Nếu gặp vulnerability warning, có thể bỏ qua")
        return False

def main():
    print("🚀 Setup cho máy mới...")
    
    # Tạo thư mục cần thiết
    os.makedirs('model_cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Cài đặt requirements
    install_requirements()
    
    # Kiểm tra model cache
    cache_ok = check_model_cache()
    
    # Tải models nếu thiếu
    if not cache_ok:
        success = download_models()
        if not success:
            print("❌ Setup thất bại - không thể tải model")
            return False
    
    print("✅ Setup hoàn thành!")
    print("\n📋 Bước tiếp theo:")
    print("1. Chạy: python src/training/scripts/train_gpu.py")
    print("2. Hoặc chạy: python api/server.py (cho API)")
    return True

if __name__ == "__main__":
    main()
