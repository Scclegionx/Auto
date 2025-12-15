#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/setup_env.py

Script setup môi trường cho máy mới sau khi pull code từ Git:
- Tạo virtualenv (mặc định: venv_new).
- Cài đặt dependencies từ requirements.txt.
- Chạy một số bước chuẩn hoá dữ liệu/dataset:
  - Đồng bộ lại grouped datasets từ elderly_commands_master.json.
- In hướng dẫn chạy full train và API server.

Cách dùng (từ thư mục gốc project):
    python scripts/setup_env.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / "venv_new"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Chạy lệnh subprocess, echo ra console."""
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd or PROJECT_ROOT))


def ensure_venv() -> Path:
    """Tạo (hoặc dùng lại) virtualenv tại venv_new, trả về đường dẫn python trong venv."""
    if not VENV_DIR.exists():
        print(f"[*] Creating virtualenv at {VENV_DIR} ...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"[*] Using existing virtualenv at {VENV_DIR}")

    if os.name == "nt":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"

    if not python_path.exists():
        raise RuntimeError(f"Không tìm thấy python trong venv: {python_path}")

    return python_path


def install_requirements(venv_python: Path) -> None:
    """Cài đặt requirements vào venv."""
    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"Không tìm thấy requirements.txt tại {REQUIREMENTS}")

    print("\n[*] Upgrading pip ...")
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    print("\n[*] Installing dependencies from requirements.txt ...")
    run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def sync_grouped_datasets(venv_python: Path) -> None:
    """Đồng bộ lại grouped datasets từ elderly_commands_master.json."""
    script_path = PROJECT_ROOT / "scripts" / "data" / "generate_grouped_from_master.py"
    if not script_path.exists():
        print(f"[WARN] Không tìm thấy script generate_grouped_from_master.py tại {script_path}, bỏ qua bước sync grouped.")
        return

    print("\n[*] Sync grouped datasets from elderly_commands_master.json ...")
    run([str(venv_python), str(script_path)])


def main() -> None:
    print("=== Auto NLP Hybrid System - Setup Environment ===")
    print(f"Project root : {PROJECT_ROOT}")

    venv_python = ensure_venv()
    print(f"[*] Using Python in venv: {venv_python}")

    install_requirements(venv_python)
    sync_grouped_datasets(venv_python)

    print("\n=== Setup hoàn tất ✅ ===")
    print("\nTiếp theo bạn có thể:")
    print("1) Chạy full train:")
    print(f"   {venv_python} src/training/scripts/train_gpu.py --epochs 10 --batch-size 8 --grad-accum 2 --freeze-epochs 2")
    print("\n2) Chạy API server:")
    print(f"   set PYTHONPATH={PROJECT_ROOT/'src'};{PROJECT_ROOT}  (trên Windows PowerShell dùng $env:PYTHONPATH)")
    print(f"   {venv_python} api/server.py")
    print("\nHoặc tự kích hoạt venv rồi chạy lệnh thủ công:")
    if os.name == "nt":
        print("   .\\venv_new\\Scripts\\activate")
    else:
        print("   source venv_new/bin/activate")


if __name__ == "__main__":
    main()


