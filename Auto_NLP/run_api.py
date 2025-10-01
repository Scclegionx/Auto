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
    
    print("CHAY API SERVER TU CAU TRUC MOI")
    print("=" * 50)
    
    try:
        # Import và chạy trực tiếp
        from src.inference.api.api_server import app
        import uvicorn
        
        print("API Documentation: http://localhost:8000/docs")
        print("Predict Endpoint: POST http://localhost:8000/predict")
        print("=" * 50)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"Loi khoi dong API server: {e}")
        print("Thu chay truc tiep...")
        
        # Fallback: chạy trực tiếp file
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "src/inference/api/api_server.py"
        ], cwd=current_dir)
        
        if result.returncode == 0:
            print("API server da dung!")
        else:
            print("API server gap loi!")

if __name__ == "__main__":
    main()
