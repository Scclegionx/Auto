#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point
Điểm khởi đầu chính của hệ thống
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.hybrid_system import ModelFirstHybridSystem

def test_hybrid_system():
    """Test hybrid system"""
    print("🧪 Testing Hybrid System...")
    
    try:
        # Initialize system
        hybrid_system = ModelFirstHybridSystem()
        
        # Test cases
        test_cases = [
            "gọi điện cho mẹ",
            "bật đèn phòng khách", 
            "phát nhạc",
            "tìm kiếm nhạc trên youtube",
            "đặt báo thức 7 giờ sáng"
        ]
        
        print(f"\nTesting {len(test_cases)} cases...")
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_case}'")
            result = hybrid_system.predict(test_case)
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Method: {result['method']}")
        
        print("\n✅ Hybrid system test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def run_api():
    """Run API server"""
    print("🚀 Starting API Server...")
    
    try:
        import uvicorn
        from api.server import app
        
        uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            workers=config.API_WORKERS,
            log_level=config.LOG_LEVEL.lower()
        )
        
    except Exception as e:
        print(f"❌ API server failed: {e}")
        import traceback
        traceback.print_exc()

def train_model():
    """Train model"""
    print("🎯 Training Model...")
    
    try:
        import subprocess
        
        # Run training script
        result = subprocess.run([
            sys.executable, 
            str(config.TRAINING_DIR / "scripts" / "train_gpu.py")
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Auto NLP Hybrid System")
    parser.add_argument("command", choices=["test", "api", "train", "config"], 
                       help="Command to run")
    
    args = parser.parse_args()
    
    # Print config
    config.print_config()
    
    if args.command == "test":
        test_hybrid_system()
    elif args.command == "api":
        run_api()
    elif args.command == "train":
        train_model()
    elif args.command == "config":
        print("\n📋 Current Configuration:")
        config.print_config()

if __name__ == "__main__":
    main()
