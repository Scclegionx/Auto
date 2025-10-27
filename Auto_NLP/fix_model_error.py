#!/usr/bin/env python3
"""
Script fix nhanh cho lỗi: 'NoneType' object has no attribute 'endswith'
Khi clone dự án về máy mới và gặp lỗi model loading
"""

import os
import sys
import shutil
import warnings
from pathlib import Path

def fix_model_loading_error():
    """Fix lỗi model loading"""
    print("🔧 Fixing model loading error...")
    print("Lỗi: 'NoneType' object has no attribute 'endswith'")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    try:
        # 1. Clear all caches
        print("🗑️ Clearing model caches...")
        cache_paths = [
            Path.home() / ".cache" / "huggingface",
            Path("model_cache"),
            Path("models")
        ]
        
        for cache_path in cache_paths:
            if cache_path.exists():
                print(f"   - Removing: {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
        
        # 2. Force download with explicit parameters
        print("📥 Force downloading PhoBERT-large...")
        
        from transformers import AutoTokenizer, AutoModel
        
        # Download tokenizer with explicit parameters
        print("   - Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-large",
            cache_dir="model_cache",
            force_download=True,
            local_files_only=False,
            trust_remote_code=False
        )
        
        # Download model with explicit parameters
        print("   - Downloading model...")
        model = AutoModel.from_pretrained(
            "vinai/phobert-large",
            cache_dir="model_cache", 
            force_download=True,
            local_files_only=False,
            trust_remote_code=False
        )
        
        # 3. Test loading
        print("🧪 Testing model loading...")
        test_text = "Xin chào"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        
        print("✅ Model loading test successful!")
        print(f"   - Tokenizer: {type(tokenizer).__name__}")
        print(f"   - Model: {type(model).__name__}")
        print(f"   - Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Alternative solutions:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade transformers")
        print("3. Try: pip install --upgrade torch")
        print("4. Run: python setup_complete.py")
        return False

def check_environment():
    """Check environment"""
    print("🔍 Checking environment...")
    
    try:
        import torch
        import transformers
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: python setup_complete.py")
        return False

def main():
    """Main function"""
    print("🚀 Auto NLP - Fix Model Loading Error")
    print("=" * 50)
    print("Fixing: 'NoneType' object has no attribute 'endswith'")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return False
    
    # Fix model loading
    if not fix_model_loading_error():
        return False
    
    print("\n🎉 Fix completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run training: python src/training/scripts/train_gpu.py")
    print("2. If still fails, run: python setup_complete.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
