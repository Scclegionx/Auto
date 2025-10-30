#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo lại processed data từ dataset đã cải thiện
"""

import json
import random
from pathlib import Path

def regenerate_processed_data():
    """Tạo lại processed data từ dataset đã cải thiện"""
    
    print("🔄 REGENERATING PROCESSED DATA")
    print("=" * 50)
    
    # Load improved raw dataset
    raw_dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    print(f"📖 Loading improved raw dataset from {raw_dataset_path}...")
    
    with open(raw_dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✅ Loaded {len(raw_data)} samples")
    
    # Shuffle data
    print("🔀 Shuffling data...")
    random.seed(42)  # For reproducibility
    random.shuffle(raw_data)
    
    # Split data (80% train, 10% val, 10% test)
    total_samples = len(raw_data)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    train_data = raw_data[:train_size]
    val_data = raw_data[train_size:train_size + val_size]
    test_data = raw_data[train_size + val_size:]
    
    print(f"✂️ Data split:")
    print(f"   Train: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
    print(f"   Val: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
    print(f"   Test: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    
    # Create processed directory
    processed_dir = "src/data/processed"
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print("\n💾 Saving fresh data splits...")
    
    # Save train
    train_file = Path(processed_dir) / "train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved train: {train_file}")
    
    # Save val
    val_file = Path(processed_dir) / "val.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved val: {val_file}")
    
    # Save test
    test_file = Path(processed_dir) / "test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved test: {test_file}")
    
    # Show sample from each split
    print("\n📋 Sample from each split:")
    print(f"Train sample: {train_data[0]['input'][:50]}...")
    print(f"Val sample: {val_data[0]['input'][:50]}...")
    print(f"Test sample: {test_data[0]['input'][:50]}...")
    
    # Show entity distribution
    print("\n📊 ENTITY DISTRIBUTION IN NEW DATA:")
    print("-" * 40)
    
    all_entities = {}
    for item in train_data + val_data + test_data:
        for entity in item.get('entities', []):
            entity_label = entity['label']
            all_entities[entity_label] = all_entities.get(entity_label, 0) + 1
    
    for entity, count in sorted(all_entities.items()):
        print(f"  {entity}: {count}")
    
    print("\n🎉 FRESH PROCESSED DATA GENERATED!")
    print("=" * 50)
    print("Dataset is now ready for training with improved data!")

if __name__ == "__main__":
    regenerate_processed_data()
