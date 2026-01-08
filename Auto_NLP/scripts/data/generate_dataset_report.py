#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script wrapper để tạo đầy đủ dataset report và dashboard.
Chạy cả aggregate và visualization cùng lúc.

Usage:
    python scripts/data/generate_dataset_report.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tạo đầy đủ dataset report và dashboard.")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Bỏ qua bước aggregate (nếu đã có report).",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Bỏ qua bước tạo dashboard.",
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    print("="*70)
    print("DATASET REPORT GENERATOR")
    print("="*70)
    print()
    
    # Step 1: Aggregate datasets
    if not args.skip_aggregate:
        print("[STEP 1] Dang tong hop datasets...")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(script_dir / "aggregate_datasets.py")],
            capture_output=False,
        )
        if result.returncode != 0:
            print("[ERROR] Aggregate datasets failed!")
            return 1
        print()
    else:
        print("[SKIP] Bo qua aggregate datasets...")
        print()
    
    # Step 2: Create dashboard
    if not args.skip_dashboard:
        print("[STEP 2] Dang tao dashboard...")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(script_dir / "create_dataset_dashboard.py")],
            capture_output=False,
        )
        if result.returncode != 0:
            print("[ERROR] Create dashboard failed!")
            return 1
        print()
    else:
        print("[SKIP] Bo qua tao dashboard...")
        print()
    
    print("="*70)
    print("[OK] Hoan tat! Cac file da duoc tao:")
    print("  - reports/dataset_aggregate_report.json")
    print("  - reports/dataset_dashboard.html")
    print("  - reports/dashboard_*.png (visualizations)")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

