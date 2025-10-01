
import sys
import os
import gc
import torch
from pathlib import Path

# Set environment variables for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src" / "training" / "configs"))
sys.path.insert(0, str(current_dir / "src" / "models" / "base"))
sys.path.insert(0, str(current_dir / "src" / "training" / "utils"))

def clean_old_data():
    """Clean old training data before starting new training"""
    print("🧹 Cleaning old training data...")
    
    # Clean model checkpoints
    model_dirs = ["models", "src/models/trained"]
    for model_dir in model_dirs:
        if Path(model_dir).exists():
            import shutil
            shutil.rmtree(model_dir)
            print(f"✅ Removed: {model_dir}")
    
    # Clean log files
    import glob
    log_files = glob.glob("*.log")
    for log_file in log_files:
        os.remove(log_file)
        print(f"✅ Removed: {log_file}")
    
    # Clean cache
    cache_dirs = ["model_cache", "__pycache__"]
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✅ Removed: {cache_dir}")
    
    # Create fresh directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("✅ Created fresh directories")

def main():
    print("🚀 CHẠY TRAINING TỪ CẤU TRÚC MỚI")
    print("=" * 50)
    
    # Clean old data first
    clean_old_data()
    
    # Cấu hình tối ưu cho GPU 6GB
    os.environ['MAX_LENGTH'] = '64'
    os.environ['BATCH_SIZE'] = '2'
    print("🎮 Optimized configuration for GPU 6GB:")
    print("   - Max Length: 64")
    print("   - Batch Size: 2")
    print("   - Gradient Accumulation: 4 (effective batch 8)")
    
    # Clear GPU cache before starting (only if CUDA is actually available)
    if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        except RuntimeError as e:
            print(f"⚠️ Could not clear GPU cache: {e}")
            print("🖥️ Continuing with CPU training...")
    else:
        print("🖥️ Using CPU training (CUDA disabled)")
    
    try:
        from src.training.scripts.train_gpu import main as train_main
        print("🔧 Starting training with improved error handling...")
        train_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("🔧 Thử chạy trực tiếp...")
        
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "src/training/scripts/train_gpu.py"
        ], cwd=current_dir)
        
        if result.returncode == 0:
            print("✅ Training hoàn thành!")
        else:
            print("❌ Training thất bại!")
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        print("🔧 Thử chạy với cấu hình an toàn hơn...")
        
        # Try with reduced batch size
        try:
            import subprocess
            env = os.environ.copy()
            env['BATCH_SIZE'] = '2'
            env['MAX_LENGTH'] = '64'
            
            result = subprocess.run([
                sys.executable, 
                "src/training/scripts/train_gpu.py"
            ], cwd=current_dir, env=env)
            
            if result.returncode == 0:
                print("✅ Training hoàn thành với cấu hình an toàn!")
            else:
                print("❌ Training vẫn thất bại!")
                
        except Exception as e2:
            print(f"❌ Lỗi cuối cùng: {e2}")

if __name__ == "__main__":
    main()
