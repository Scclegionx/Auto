#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix span alignment issues trong processed dataset
- Tìm lại entity text trong text hiện tại
- Update start/end positions
- Remove invalid entities/spans
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Add project root to path
_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.entity_schema import ENTITY_BASE_NAMES


def normalize_text_for_matching(text: str) -> str:
    """Normalize text để matching tốt hơn"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def find_entity_in_text(entity_text: str, text: str, start_hint: int = 0) -> Optional[Tuple[int, int]]:
    """
    Tìm entity text trong text, trả về (start, end) position
    Sử dụng multiple strategies để tìm tốt nhất
    """
    if not entity_text or not text:
        return None
    
    # Strategy 1: Exact match (case-insensitive)
    entity_normalized = normalize_text_for_matching(entity_text)
    text_normalized = normalize_text_for_matching(text)
    
    # Try exact match first
    idx = text_normalized.lower().find(entity_normalized.lower(), start_hint)
    if idx != -1:
        # Find actual positions in original text (accounting for spaces)
        # Map normalized position back to original
        original_start = idx
        original_end = idx + len(entity_normalized)
        
        # Adjust for actual text (handle spaces)
        actual_start = 0
        actual_end = 0
        norm_pos = 0
        
        for i, char in enumerate(text):
            if norm_pos == original_start:
                actual_start = i
            if norm_pos == original_end:
                actual_end = i
                break
            if char.strip():  # Non-space character
                norm_pos += 1
        
        if actual_end == 0:
            actual_end = len(text)
        
        # Verify the found text matches
        found_text = text[actual_start:actual_end].strip()
        if entity_normalized.lower() in found_text.lower() or found_text.lower() in entity_normalized.lower():
            return (actual_start, actual_end)
    
    # Strategy 2: Partial match (entity text contains in text or vice versa)
    if entity_normalized.lower() in text_normalized.lower():
        idx = text_normalized.lower().find(entity_normalized.lower())
        if idx != -1:
            # Find boundaries
            # Try to find word boundaries
            start = max(0, idx - 5)
            end = min(len(text_normalized), idx + len(entity_normalized) + 5)
            
            # Find actual positions
            actual_start = 0
            actual_end = 0
            norm_pos = 0
            
            for i, char in enumerate(text):
                if norm_pos == start:
                    actual_start = i
                if norm_pos == end:
                    actual_end = i
                    break
                if char.strip():
                    norm_pos += 1
            
            if actual_end == 0:
                actual_end = len(text)
            
            return (actual_start, actual_end)
    
    # Strategy 3: Word-by-word matching (for multi-word entities)
    entity_words = entity_normalized.split()
    if len(entity_words) > 1:
        # Try to find consecutive words
        text_words = text_normalized.split()
        for i in range(len(text_words) - len(entity_words) + 1):
            window = ' '.join(text_words[i:i+len(entity_words)])
            if window.lower() == entity_normalized.lower():
                # Find positions of these words in original text
                word_start_idx = 0
                word_count = 0
                for j, char in enumerate(text):
                    if char == ' ' or j == 0:
                        if word_count == i:
                            word_start_idx = j if j == 0 else j + 1
                            break
                        if char == ' ':
                            word_count += 1
                
                # Find end position
                word_end_idx = word_start_idx
                for j in range(word_start_idx, len(text)):
                    if j > word_start_idx and text[j-1] == ' ':
                        word_count += 1
                        if word_count >= i + len(entity_words):
                            word_end_idx = j
                            break
                
                if word_end_idx == word_start_idx:
                    word_end_idx = len(text)
                
                return (word_start_idx, word_end_idx)
    
    return None


def fix_entity_spans(sample: Dict) -> Tuple[Dict, Dict]:
    """
    Fix entity và span positions trong một sample
    Returns: (fixed_sample, stats)
    """
    text = sample.get("input", "")
    if not text:
        return sample, {"removed": 0, "fixed": 0, "kept": 0}
    
    text_len = len(text)
    stats = {"removed": 0, "fixed": 0, "kept": 0}
    
    # Fix entities
    fixed_entities = []
    entities = sample.get("entities", [])
    
    for entity in entities:
        if not isinstance(entity, dict):
            stats["removed"] += 1
            continue
        
        entity_text = entity.get("text", "").strip()
        label = entity.get("label", "")
        original_start = entity.get("start", -1)
        original_end = entity.get("end", -1)
        
        # Check if current position is valid
        if 0 <= original_start < original_end <= text_len:
            # Position is valid, keep it
            fixed_entities.append(entity)
            stats["kept"] += 1
            continue
        
        # Try to find entity in text
        if entity_text:
            found_pos = find_entity_in_text(entity_text, text, max(0, original_start - 10))
            
            if found_pos:
                new_start, new_end = found_pos
                # Verify it's within text bounds
                if 0 <= new_start < new_end <= text_len:
                    fixed_entity = {
                        "label": label,
                        "text": entity_text,
                        "start": new_start,
                        "end": new_end
                    }
                    fixed_entities.append(fixed_entity)
                    stats["fixed"] += 1
                    continue
        
        # Cannot find or fix, remove it
        stats["removed"] += 1
    
    # Fix spans (same logic)
    fixed_spans = []
    spans = sample.get("spans", [])
    
    for span in spans:
        if not isinstance(span, dict):
            stats["removed"] += 1
            continue
        
        span_text = span.get("text", "").strip()
        label = span.get("label", "")
        original_start = span.get("start", -1)
        original_end = span.get("end", -1)
        
        # Check if current position is valid
        if 0 <= original_start < original_end <= text_len:
            # Position is valid, keep it
            fixed_spans.append(span)
            continue
        
        # Try to find span in text
        if span_text:
            found_pos = find_entity_in_text(span_text, text, max(0, original_start - 10))
            
            if found_pos:
                new_start, new_end = found_pos
                # Verify it's within text bounds
                if 0 <= new_start < new_end <= text_len:
                    fixed_span = {
                        "label": label,
                        "text": span_text,
                        "start": new_start,
                        "end": new_end
                    }
                    fixed_spans.append(fixed_span)
                    continue
        
        # Cannot find or fix, remove it
        stats["removed"] += 1
    
    # Create fixed sample
    fixed_sample = sample.copy()
    fixed_sample["entities"] = fixed_entities
    fixed_sample["spans"] = fixed_spans
    
    return fixed_sample, stats


def fix_dataset_file(input_path: Path, output_path: Path, backup: bool = True) -> Dict:
    """
    Fix toàn bộ dataset file
    """
    print(f"\nProcessing {input_path.name}...")
    
    # Load dataset
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"success": False, "error": str(e), "stats": {}}
    
    if not isinstance(data, list):
        return {"success": False, "error": "Dataset must be a list", "stats": {}}
    
    # Backup if requested
    if backup:
        backup_path = input_path.with_suffix('.json.backup')
        print(f"  Creating backup: {backup_path.name}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Fix each sample
    fixed_data = []
    total_stats = {"removed": 0, "fixed": 0, "kept": 0, "total": len(data)}
    
    for idx, sample in enumerate(data):
        fixed_sample, sample_stats = fix_entity_spans(sample)
        fixed_data.append(fixed_sample)
        
        total_stats["removed"] += sample_stats["removed"]
        total_stats["fixed"] += sample_stats["fixed"]
        total_stats["kept"] += sample_stats["kept"]
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples...")
    
    # Save fixed dataset
    print(f"  Saving fixed dataset to {output_path.name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    return {
        "success": True,
        "stats": total_stats,
        "output_path": str(output_path)
    }


def main():
    """Main function"""
    print("=" * 70)
    print("FIX SPAN ALIGNMENT IN PROCESSED DATASET")
    print("=" * 70)
    
    # Dataset paths
    processed_dir = PROJECT_ROOT / "src" / "data" / "processed"
    train_path = processed_dir / "train.json"
    val_path = processed_dir / "val.json"
    test_path = processed_dir / "test.json"
    
    # Check if files exist
    for file_path in [train_path, val_path, test_path]:
        if not file_path.exists():
            print(f"\n[ERROR] File not found: {file_path}")
            return 1
    
    # Process each file
    results = {}
    for file_path in [train_path, val_path, test_path]:
        result = fix_dataset_file(file_path, file_path, backup=True)
        results[file_path.name] = result
        
        if result["success"]:
            stats = result["stats"]
            print(f"\n[OK] {file_path.name}:")
            print(f"  Total samples: {stats['total']}")
            print(f"  Kept (valid): {stats['kept']}")
            print(f"  Fixed: {stats['fixed']}")
            print(f"  Removed (invalid): {stats['removed']}")
            print(f"  Success rate: {(stats['kept'] + stats['fixed']) / stats['total'] * 100:.1f}%")
        else:
            print(f"\n[ERROR] {file_path.name}: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_success = all(r.get("success", False) for r in results.values())
    
    if all_success:
        print("\n[SUCCESS] All datasets fixed successfully!")
        print("\nNext steps:")
        print("1. Run validation script to verify fixes:")
        print("   python scripts/data/validate_processed_dataset.py")
        print("2. If validation passes, you can start training!")
        return 0
    else:
        print("\n[WARNING] Some datasets failed to fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

