#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build train/val/test từ các file grouped theo intent với số lượng cụ thể.
Usage:
    python scripts/data/build_from_grouped.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.entity_schema import canonicalize_entity_dict
from data.processed.data_processor import DataProcessor

# Số lượng mẫu mong muốn cho từng intent
TARGET_COUNTS = {
    "add-contacts": 3407,
    "call": 4371,
    "control-device": 3397,
    "get-info": 3496,
    "make-video-call": 3399,
    "open-cam": 3403,
    "search-internet": 3518,
    "search-youtube": 3481,
    "send-mess": 3599,
    "set-alarm": 3538,
}

GROUPED_DIR = PROJECT_ROOT / "src" / "data" / "grouped"
OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "processed"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_grouped_file(intent: str) -> List[Dict]:
    """Load file grouped theo intent."""
    filename = f"{intent.replace('-', '_')}_dataset.json"
    path = GROUPED_DIR / filename
    if not path.exists():
        print(f"WARNING: File không tồn tại: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"WARNING: File {filename} không phải là list")
        return []
    return data


def sample_intent(intent: str, target_count: int, seed: int) -> List[Dict]:
    """Sample đúng số lượng mẫu từ file grouped."""
    samples = load_grouped_file(intent)
    if len(samples) == 0:
        print(f"ERROR: Không có mẫu nào cho intent '{intent}'")
        return []
    
    if len(samples) < target_count:
        print(f"WARNING: Intent '{intent}' chỉ có {len(samples)} mẫu, cần {target_count}. Sẽ lấy hết + duplicate.")
        # Lấy hết + duplicate để đủ số
        rng = random.Random(seed)
        result = list(samples)
        while len(result) < target_count:
            result.append(rng.choice(samples))
        return result[:target_count]
    
    # Sample ngẫu nhiên
    rng = random.Random(seed)
    return rng.sample(samples, target_count)


def normalize_sample(sample: Dict, processor: DataProcessor) -> Dict:
    """Chuẩn hóa sample: canonicalize entities + sinh lại BIO."""
    sample = dict(sample)
    text = sample.get("input", "")
    entities = sample.get("entities", [])
    
    # Chuẩn hóa entities
    normalized_entities = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        canonical = canonicalize_entity_dict(ent)
        # Tìm lại start/end nếu thiếu hoặc không hợp lệ
        ent_text = canonical.get("text", "")
        start = canonical.get("start")
        end = canonical.get("end")
        
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            # Tìm lại trong text
            if ent_text:
                idx = text.find(ent_text)
                if idx >= 0:
                    canonical["start"] = idx
                    canonical["end"] = idx + len(ent_text)
                else:
                    # Không tìm thấy → skip
                    continue
            else:
                continue
        
        normalized_entities.append(canonical)
    
    sample["entities"] = normalized_entities
    sample["spans"] = normalized_entities
    
    # Sinh lại BIO
    sample["bio_labels"] = processor.align_labels(text, normalized_entities)
    
    return sample


def split_data(all_samples: List[Dict], seed: int) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia tách train/val/test theo tỉ lệ 80/10/10."""
    rng = random.Random(seed)
    rng.shuffle(all_samples)
    
    total = len(all_samples)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    
    train = all_samples[:train_size]
    val = all_samples[train_size:train_size + val_size]
    test = all_samples[train_size + val_size:]
    
    return train, val, test


def save_json(path: Path, data: List[Dict]) -> None:
    """Ghi JSON với indent 2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("Build train/val/test từ grouped dataset")
    print("=" * 60)
    
    processor = DataProcessor()
    all_samples: List[Dict] = []
    
    # Đọc và sample từng intent
    for intent, count in TARGET_COUNTS.items():
        print(f"\nIntent: {intent}")
        print(f"  Target: {count} mẫu")
        samples = sample_intent(intent, count, SEED)
        print(f"  Sampled: {len(samples)} mẫu")
        
        # Normalize
        normalized = []
        for sample in samples:
            try:
                norm = normalize_sample(sample, processor)
                normalized.append(norm)
            except Exception as e:
                print(f"  WARNING: Lỗi khi normalize sample: {e}")
                continue
        
        print(f"  Normalized: {len(normalized)} mẫu")
        all_samples.extend(normalized)
    
    print(f"\n{'='*60}")
    print(f"Tổng số mẫu: {len(all_samples)}")
    print(f"{'='*60}")
    
    # Split train/val/test
    train, val, test = split_data(all_samples, SEED)
    
    print(f"\nTách tập:")
    print(f"  Train: {len(train)} mẫu (~{len(train)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val)} mẫu (~{len(val)/len(all_samples)*100:.1f}%)")
    print(f"  Test:  {len(test)} mẫu (~{len(test)/len(all_samples)*100:.1f}%)")
    
    # Ghi file
    train_path = OUTPUT_DIR / "train.json"
    val_path = OUTPUT_DIR / "val.json"
    test_path = OUTPUT_DIR / "test.json"
    
    print(f"\nGhi file:")
    print(f"  {train_path}")
    save_json(train_path, train)
    print(f"  {val_path}")
    save_json(val_path, val)
    print(f"  {test_path}")
    save_json(test_path, test)
    
    print(f"\n✅ Hoàn tất!")


if __name__ == "__main__":
    main()






