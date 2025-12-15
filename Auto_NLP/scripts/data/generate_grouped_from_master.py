#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-command grouped datasets from the single master dataset.

Đọc `src/data/raw/elderly_commands_master.json` và ghi ra:
    - `src/data/grouped/add-contacts.json`
    - `src/data/grouped/call.json`
    - ...
    - `src/data/grouped/set-alarm.json`

Mỗi file là list các sample đầy đủ (giữ nguyên cấu trúc gốc), chỉ khác là đã lọc theo `command`.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "src" / "data" / "raw" / "elderly_commands_master.json"
GROUPED_DIR = PROJECT_ROOT / "src" / "data" / "grouped"


def load_master() -> List[Dict]:
    """Load master dataset."""
    with RAW_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("elderly_commands_master.json phải là list các sample.")
    return data


def group_by_command(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group samples by `command` field."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        cmd = sample.get("command") or sample.get("intent") or "unknown"
        groups[cmd].append(sample)
    return groups


def main() -> None:
    # Dùng log ASCII để tránh lỗi encoding trên Windows console
    print("Project root:", PROJECT_ROOT)
    print("Loading master dataset from:", RAW_PATH)
    samples = load_master()
    print(f"Loaded {len(samples)} samples from master dataset.")

    groups = group_by_command(samples)
    GROUPED_DIR.mkdir(parents=True, exist_ok=True)

    expected_commands = [
        "add-contacts",
        "call",
        "control-device",
        "get-info",
        "make-video-call",
        "open-cam",
        "search-internet",
        "search-youtube",
        "send-mess",
        "set-alarm",
    ]

    print("\nWriting grouped datasets to", GROUPED_DIR)
    for cmd in expected_commands:
        cmd_samples = groups.get(cmd, [])
        out_path = GROUPED_DIR / f"{cmd}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cmd_samples, f, ensure_ascii=False, indent=2)
        print(f"  - {cmd}.json: {len(cmd_samples)} samples")

    # Optional: report unknown / extra commands if có
    extra_cmds = sorted(set(groups.keys()) - set(expected_commands))
    if extra_cmds:
        print("\n[WARN] Found extra commands not in expected list:", ", ".join(extra_cmds))

    print("\nDone. Grouped datasets are now synced with elderly_commands_master.json.")


if __name__ == "__main__":
    main()


