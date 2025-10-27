#!/usr/bin/env python3
"""
Script fix nhanh cho vấn đề model cache
"""

import subprocess
import sys
import os

def fix_pytorch_vulnerability():
    """Fix PyTorch vulnerability"""
    print("🔧 Fix PyTorch vulnerability...")
    
    try:
        # Upgrade PyTorch
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch>=2.6.0', '--upgrade'])
        print("✅ PyTorch upgraded to 2.6.0+")
        return True
    except Exception as e:
        print(f"❌ Lỗi upgrade PyTorch: {e}")
        return False

def force_download_model():
    """Force download model"""
    print("🔄 Force download PhoBERT-large...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Xóa cache cũ nếu có
        cache_path = "model_cache/models--vinai--phobert-large"
        if os.path.exists(cache_path):
            import shutil
            shutil.rmtree(cache_path)
            print("🗑️ Đã xóa cache cũ")
        
        # Download lại
        print("Tải tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-large",
            force_download=True,
            cache_dir="model_cache"
        )
        
        print("Tải model...")
        model = AutoModel.from_pretrained(
            "vinai/phobert-large", 
            force_download=True,
            cache_dir="model_cache"
        )
        
        print("✅ Model downloaded successfully")
        
        # Test
        test_text = "Xin chào"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Test OK: '{test_text}' -> {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi download model: {e}")
        return False

def main():
    print("🚀 Fix model cache issues...")
    print("=" * 40)
    
    # Fix PyTorch
    pytorch_ok = fix_pytorch_vulnerability()
    
    if pytorch_ok:
        # Force download model
        model_ok = force_download_model()
        
        if model_ok:
            print("\n✅ Fix hoàn thành!")
            print("Bây giờ có thể chạy training:")
            print("python src/training/scripts/train_gpu.py")
        else:
            print("\n❌ Fix thất bại - không thể download model")
    else:
        print("\n❌ Fix thất bại - không thể upgrade PyTorch")

if __name__ == "__main__":
    main()
