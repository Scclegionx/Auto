#!/usr/bin/env python3
"""
Script kiểm tra nhanh model cache
"""

import os
import sys

def check_model_cache_detailed():
    """Kiểm tra chi tiết model cache"""
    print("🔍 Kiểm tra chi tiết model cache...")
    
    # Kiểm tra thư mục gốc
    if not os.path.exists('model_cache'):
        print("❌ Thư mục model_cache không tồn tại")
        print("💡 Chạy: mkdir model_cache")
        return False
    
    print("✅ Thư mục model_cache tồn tại")
    
    # Kiểm tra PhoBERT-large
    phobert_large_path = 'model_cache/models--vinai--phobert-large'
    if not os.path.exists(phobert_large_path):
        print(f"❌ {phobert_large_path} không tồn tại")
        print("💡 Cần tải PhoBERT-large model")
        return False
    
    print(f"✅ {phobert_large_path} tồn tại")
    
    # Kiểm tra nội dung thư mục
    try:
        contents = os.listdir(phobert_large_path)
        print(f"📁 Nội dung: {contents}")
        
        # Kiểm tra snapshots
        snapshots_path = os.path.join(phobert_large_path, 'snapshots')
        if os.path.exists(snapshots_path):
            snapshots = os.listdir(snapshots_path)
            print(f"📸 Snapshots: {snapshots}")
            
            if snapshots:
                snapshot_path = os.path.join(snapshots_path, snapshots[0])
                snapshot_contents = os.listdir(snapshot_path)
                print(f"📄 Snapshot contents: {snapshot_contents}")
                
                # Kiểm tra các file quan trọng
                important_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                for file in important_files:
                    file_path = os.path.join(snapshot_path, file)
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path) / (1024*1024)  # MB
                        print(f"✅ {file}: {size:.1f}MB")
                    else:
                        print(f"❌ {file}: MISSING")
        else:
            print("❌ Snapshots folder không tồn tại")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi đọc thư mục: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading model"""
    print("\n🧪 Test loading model...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print("Tải tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        print("✅ Tokenizer loaded")
        
        print("Tải model...")
        model = AutoModel.from_pretrained("vinai/phobert-large")
        print("✅ Model loaded")
        
        # Test tokenization
        test_text = "Xin chào"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test: '{test_text}' -> {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi loading model: {e}")
        return False

def main():
    print("🔍 Kiểm tra model cache chi tiết...")
    print("=" * 50)
    
    # Kiểm tra cache
    cache_ok = check_model_cache_detailed()
    
    if cache_ok:
        print("\n🧪 Test loading model...")
        loading_ok = test_model_loading()
        
        if loading_ok:
            print("\n✅ Model cache hoàn toàn OK!")
        else:
            print("\n❌ Model cache có vấn đề khi loading")
    else:
        print("\n❌ Model cache không đầy đủ")
        print("\n💡 Giải pháp:")
        print("1. Chạy: python setup_new_machine.py")
        print("2. Hoặc tải thủ công:")
        print("   python -c \"from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('vinai/phobert-large'); AutoModel.from_pretrained('vinai/phobert-large')\"")

if __name__ == "__main__":
    main()
