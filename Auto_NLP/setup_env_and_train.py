
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parent  # Thư mục Auto_NLP


def run(cmd: list[str], env: Dict[str, str] | None = None, cwd: Path | None = None) -> None:
    """Chạy lệnh và hiển thị realtime output, raise nếu lỗi."""
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, env=env, cwd=str(cwd or PROJECT_ROOT))


def get_venv_python() -> Path:
    """Trả về đường dẫn python.exe trong venv_new (Windows-focused)."""
    if os.name == "nt":
        candidate = PROJECT_ROOT / "venv_new" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / "venv_new" / "bin" / "python"
    return candidate


def ensure_venv() -> Path:
    """Tạo venv_new nếu chưa có, trả về đường dẫn python trong venv."""
    venv_python = get_venv_python()
    if venv_python.exists():
        print(f"[INFO] Đã tìm thấy virtualenv tại: {venv_python}")
        return venv_python

    print("[INFO] Chưa có virtualenv `venv_new`, đang tạo...")
    run([sys.executable, "-m", "venv", "venv_new"])

    if not venv_python.exists():
        raise RuntimeError("Không tạo được virtualenv venv_new – kiểm tra lại Python / quyền ghi đĩa.")

    print(f"[INFO] Đã tạo virtualenv: {venv_python}")
    return venv_python


def install_requirements(venv_python: Path) -> None:
    """Cài đặt requirements vào venv."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {req_file}")

    # Nâng cấp pip và cài đặt phụ thuộc
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(venv_python), "-m", "pip", "install", "-r", str(req_file)])


def probe_hardware(venv_python: Path) -> Dict[str, Any]:
    """Lấy thông tin cơ bản về GPU/CPU thông qua PyTorch trong venv."""
    code = r"""
import json, torch, platform

info = {
    "python_version": platform.python_version(),
    "torch_version": torch.__version__ if hasattr(torch, "__version__") else None,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": [],
}

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info["gpus"].append({
            "index": idx,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
        })

print(json.dumps(info))
"""
    print("[INFO] Đang kiểm tra cấu hình GPU/CPU qua PyTorch...")
    out = subprocess.check_output([str(venv_python), "-c", code], cwd=str(PROJECT_ROOT), text=True)
    hw = json.loads(out)
    print("\n[HARDWARE INFO]")
    print(json.dumps(hw, indent=2, ensure_ascii=False))
    return hw


def select_training_config(hw: Dict[str, Any]) -> Dict[str, str]:
    """Chọn cấu hình train (batch size, max_length, epochs, grad_accum, AMP) dựa trên GPU."""
    cuda_available = hw.get("cuda_available", False)
    gpus = hw.get("gpus") or []

    # Giá trị mặc định an toàn
    cfg: Dict[str, str] = {
        "MODEL_NAME": "vinai/phobert-large",
        "MAX_LENGTH": "128",
        "EPOCHS": "6",
        "BATCH_SIZE": "8",
        "GRAD_ACCUM": "4",
        "USE_MIXED_PRECISION": "1",
    }

    if not cuda_available or not gpus:
        print("[WARN] Không phát hiện GPU CUDA – sẽ train trên CPU, có thể rất chậm.")
        # CPU: giữ cấu hình mặc định tương đối nhẹ
        return cfg

    # Lấy GPU 0
    gpu0 = gpus[0]
    mem_gb = gpu0.get("total_memory_gb", 0)
    print(f"[INFO] GPU 0: {gpu0['name']} ({mem_gb} GB VRAM)")

    # Heuristic đơn giản theo dung lượng VRAM
    if mem_gb >= 16:
        cfg["BATCH_SIZE"] = "32"
        cfg["GRAD_ACCUM"] = "1"
        cfg["MAX_LENGTH"] = "160"
    elif mem_gb >= 10:
        cfg["BATCH_SIZE"] = "24"
        cfg["GRAD_ACCUM"] = "1"
        cfg["MAX_LENGTH"] = "160"
    elif mem_gb >= 8:
        cfg["BATCH_SIZE"] = "16"
        cfg["GRAD_ACCUM"] = "2"
        cfg["MAX_LENGTH"] = "128"
    elif mem_gb >= 6:
        cfg["BATCH_SIZE"] = "12"
        cfg["GRAD_ACCUM"] = "3"
        cfg["MAX_LENGTH"] = "128"
    else:
        cfg["BATCH_SIZE"] = "8"
        cfg["GRAD_ACCUM"] = "4"
        cfg["MAX_LENGTH"] = "128"

    print("\n[TRAINING CONFIG ĐỀ XUẤT]")
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    return cfg


def run_training(venv_python: Path, train_env: Dict[str, str]) -> None:
    """Chạy script train_gpu.py với env và arguments đã cấu hình."""
    train_script = PROJECT_ROOT / "src" / "training" / "scripts" / "train_gpu.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Không tìm thấy script train: {train_script}")

    env = os.environ.copy()
    env.update(train_env)

    # Build command arguments from config
    cmd = [str(venv_python), str(train_script)]
    if "EPOCHS" in train_env:
        cmd.extend(["--epochs", train_env["EPOCHS"]])
    if "BATCH_SIZE" in train_env:
        cmd.extend(["--batch-size", train_env["BATCH_SIZE"]])
    if "GRAD_ACCUM" in train_env:
        cmd.extend(["--grad-accum", train_env["GRAD_ACCUM"]])

    print("\n[INFO] Bắt đầu chạy train_gpu.py với cấu hình trên...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    run(cmd, env=env)


def main() -> None:
    print("=== Auto_NLP Setup & Train Helper ===")
    print(f"Project root: {PROJECT_ROOT}")

    venv_python = ensure_venv()

    install_requirements(venv_python)

    hw_info = probe_hardware(venv_python)

    train_cfg = select_training_config(hw_info)

    run_training(venv_python, train_cfg)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Lệnh con thất bại với mã {e.returncode}")
        sys.exit(e.returncode)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


