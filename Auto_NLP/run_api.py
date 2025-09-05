"""
Script để chạy API server từ cấu trúc mới
"""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src" / "inference" / "api"))
sys.path.insert(0, str(current_dir / "src" / "inference" / "engines"))
sys.path.insert(0, str(current_dir / "src" / "models" / "base"))
sys.path.insert(0, str(current_dir / "src" / "utils"))

def main():
    """Chạy API server"""
    
    print("🚀 CHẠY API SERVER TỪ CẤU TRÚC MỚI")
    print("=" * 50)
    
    import subprocess
    result = subprocess.run([
        sys.executable, 
        "src/inference/api/api_server.py"
    ], cwd=current_dir)
    
    if result.returncode == 0:
        print("✅ API server đã dừng!")
    else:
        print("❌ API server gặp lỗi!")

if __name__ == "__main__":
    main()
