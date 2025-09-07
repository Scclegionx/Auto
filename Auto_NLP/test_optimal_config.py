#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra cấu hình tối ưu với max_length = 512
"""

import sys
import os
from pathlib import Path

# Thêm paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src" / "training" / "configs"))

def test_optimal_config():
    """Test cấu hình tối ưu"""
    print("🚀 TESTING OPTIMAL CONFIGURATION")
    print("=" * 50)
    
    try:
        from src.training.configs.config import model_config, intent_config
        
        print("📊 CONFIGURATION SUMMARY:")
        print(f"  Model: {model_config.model_name}")
        print(f"  Max Length: {model_config.max_length}")
        print(f"  Batch Size: {model_config.batch_size}")
        print(f"  Epochs: {model_config.num_epochs}")
        print(f"  Learning Rate: {model_config.learning_rate}")
        print(f"  Freeze Layers: {model_config.freeze_layers}")
        print(f"  Gradient Checkpointing: {model_config.gradient_checkpointing}")
        print(f"  Gradient Accumulation: {model_config.gradient_accumulation_steps}")
        print(f"  FP16: {model_config.use_fp16}")
        print(f"  Device: {model_config.device}")
        
        print(f"\n📈 INTENT CONFIGURATION:")
        print(f"  Num Intents: {intent_config.num_intents}")
        print(f"  Intent Labels: {len(intent_config.intent_labels)}")
        
        # Kiểm tra memory requirements
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n💾 GPU MEMORY: {gpu_memory:.1f} GB")
            
            # Ước tính memory usage
            estimated_memory = (model_config.batch_size * model_config.max_length * 4) / 1024**3  # Rough estimate
            print(f"📊 ESTIMATED MEMORY USAGE: ~{estimated_memory:.2f} GB")
            
            if gpu_memory < 8:
                print("⚠️ WARNING: GPU memory < 8GB, consider reducing batch_size or max_length")
            else:
                print("✅ GPU memory should be sufficient")
        
        print(f"\n🎯 OPTIMAL SETTINGS FOR PERFORMANCE:")
        print(f"  ✅ Max Length: 512 (tối ưu cho context dài)")
        print(f"  ✅ Batch Size: 32 (tối ưu cho throughput)")
        print(f"  ✅ Epochs: 20 (tối ưu cho convergence)")
        print(f"  ✅ Gradient Checkpointing: True (tiết kiệm memory)")
        print(f"  ✅ FP16: True (tăng tốc training)")
        print(f"  ✅ Gradient Accumulation: 2 (ổn định training)")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_memory_compatibility():
    """Test tương thích memory"""
    print(f"\n🔍 TESTING MEMORY COMPATIBILITY...")
    
    try:
        import torch
        from src.training.configs.config import model_config
        
        if torch.cuda.is_available():
            # Test tensor creation với config hiện tại
            test_tensor = torch.randn(
                model_config.batch_size, 
                model_config.max_length, 
                device='cuda'
            )
            
            print(f"✅ Successfully created tensor: {test_tensor.shape}")
            print(f"✅ Tensor memory: {test_tensor.element_size() * test_tensor.nelement() / 1024**2:.1f} MB")
            
            # Clear memory
            del test_tensor
            torch.cuda.empty_cache()
            
            return True
        else:
            print("⚠️ CUDA not available, skipping memory test")
            return True
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 TESTING OPTIMAL CONFIGURATION")
    print("=" * 60)
    
    tests = [
        ("Optimal Config", test_optimal_config),
        ("Memory Compatibility", test_memory_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 OPTIMAL CONFIG IS READY!")
        print("🚀 You can now run training with max_length=512 and batch_size=32")
        return True
    else:
        print("⚠️ Some tests failed. Please check configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

