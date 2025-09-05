
import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src" / "training" / "configs"))
sys.path.insert(0, str(current_dir / "src" / "models" / "base"))
sys.path.insert(0, str(current_dir / "src" / "training" / "utils"))

def main():
    print("🚀 CHẠY TRAINING TỪ CẤU TRÚC MỚI")
    print("=" * 50)
    
    try:
        from src.training.scripts.train_gpu import main as train_main
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

if __name__ == "__main__":
    main()
