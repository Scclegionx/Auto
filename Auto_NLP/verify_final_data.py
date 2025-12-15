#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiểm tra data cuối cùng trước khi training
"""

import json
from collections import Counter

def verify_final_data():
    """Kiểm tra data cuối cùng"""
    
    print("🔍 VERIFYING FINAL DATA")
    print("=" * 50)
    
    # Load raw dataset
    with open("src/data/raw/elderly_commands_master.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Load processed datasets
    with open("src/data/processed/train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open("src/data/processed/val.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    with open("src/data/processed/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"📊 DATA SIZES:")
    print(f"   Raw: {len(raw_data)} samples")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    print(f"   Total processed: {len(train_data) + len(val_data) + len(test_data)} samples")
    
    # Check if total matches
    if len(raw_data) == len(train_data) + len(val_data) + len(test_data):
        print("✅ Total samples match!")
    else:
        print("❌ Total samples don't match!")
    
    # Check entity distribution in all processed data
    print("\n📊 ENTITY DISTRIBUTION IN ALL PROCESSED DATA:")
    print("-" * 50)
    
    all_processed_data = train_data + val_data + test_data
    processed_entities = Counter()
    
    for item in all_processed_data:
        for entity in item.get('entities', []):
            processed_entities[entity['label']] += 1
    
    for entity, count in processed_entities.most_common():
        print(f"  {entity}: {count}")
    
    # Check entity distribution in raw data
    print("\n📊 ENTITY DISTRIBUTION IN RAW DATA:")
    print("-" * 50)
    
    raw_entities = Counter()
    for item in raw_data:
        for entity in item.get('entities', []):
            raw_entities[entity['label']] += 1
    
    for entity, count in raw_entities.most_common():
        print(f"  {entity}: {count}")
    
    # Compare distributions
    print("\n🔍 DISTRIBUTION COMPARISON:")
    print("-" * 50)
    
    all_entities = set(raw_entities.keys()) | set(processed_entities.keys())
    match_count = 0
    
    for entity in sorted(all_entities):
        raw_count = raw_entities.get(entity, 0)
        processed_count = processed_entities.get(entity, 0)
        
        if raw_count == processed_count:
            print(f"✅ {entity}: {raw_count} = {processed_count}")
            match_count += 1
        else:
            print(f"❌ {entity}: {raw_count} != {processed_count}")
    
    print(f"\n📈 MATCH RATE: {match_count}/{len(all_entities)} entities match")
    
    # Check key improvements
    print("\n🎯 KEY IMPROVEMENTS CHECK:")
    print("-" * 50)
    
    # Check PLATFORM in call commands
    call_samples = [item for item in all_processed_data if item.get('command') == 'call']
    call_with_platform = sum(1 for item in call_samples if any(e['label'] == 'PLATFORM' for e in item.get('entities', [])))
    print(f"Call samples with PLATFORM: {call_with_platform}/{len(call_samples)}")
    
    # Check DATE in set-alarm commands
    alarm_samples = [item for item in all_processed_data if item.get('command') == 'set-alarm']
    alarm_with_date = sum(1 for item in alarm_samples if any(e['label'] == 'DATE' for e in item.get('entities', [])))
    print(f"Alarm samples with DATE: {alarm_with_date}/{len(alarm_samples)}")
    
    # Check sample quality
    print("\n📋 SAMPLE QUALITY CHECK:")
    print("-" * 50)
    
    # Show a few samples from each split
    print("Train samples:")
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        entities = [e['label'] for e in sample.get('entities', [])]
        print(f"  {i+1}. {sample['input'][:50]}... -> {entities}")
    
    print("\nVal samples:")
    for i in range(min(3, len(val_data))):
        sample = val_data[i]
        entities = [e['label'] for e in sample.get('entities', [])]
        print(f"  {i+1}. {sample['input'][:50]}... -> {entities}")
    
    print("\nTest samples:")
    for i in range(min(3, len(test_data))):
        sample = test_data[i]
        entities = [e['label'] for e in sample.get('entities', [])]
        print(f"  {i+1}. {sample['input'][:50]}... -> {entities}")
    
    # Final assessment
    print("\n💡 FINAL ASSESSMENT:")
    print("-" * 50)
    
    if len(raw_data) == len(all_processed_data) and match_count == len(all_entities):
        print("🎉 PERFECT! Data is ready for training!")
    elif len(raw_data) == len(all_processed_data):
        print("✅ GOOD! Data is ready for training (minor entity count differences due to sampling)")
    else:
        print("❌ ISSUE! Data needs to be regenerated")

if __name__ == "__main__":
    verify_final_data()
