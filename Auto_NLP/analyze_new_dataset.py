#!/usr/bin/env python3
"""
Phân tích dataset mới
"""

import json
from collections import Counter

def analyze_new_dataset():
    """Phân tích dataset mới"""
    print("=== PHÂN TÍCH DATASET MỚI ===")
    
    with open('nlp_command_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Tổng số mẫu: {len(data)}")
    
    # Phân tích commands
    commands = [item['command'] for item in data]
    command_counts = Counter(commands)
    
    print(f"\nSố loại command: {len(command_counts)}")
    print("Phân bố commands:")
    for cmd, count in sorted(command_counts.items()):
        print(f"  - {cmd}: {count} mẫu")
    
    # Phân tích entities
    entity_labels = set()
    value_labels = set()
    entity_texts = []
    value_texts = []
    
    for item in data:
        for entity in item.get('entities', []):
            if isinstance(entity, dict):
                entity_labels.add(entity.get('label', ''))
                entity_texts.append(entity.get('text', ''))
        
        for value in item.get('values', []):
            if isinstance(value, dict):
                value_labels.add(value.get('label', ''))
                value_texts.append(value.get('text', ''))
    
    print(f"\nEntity labels ({len(entity_labels)}): {sorted(entity_labels)}")
    print(f"Value labels ({len(value_labels)}): {sorted(value_labels)}")
    
    # Phân tích độ dài text
    text_lengths = [len(item['input']) for item in data]
    print(f"\nĐộ dài text - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Avg: {sum(text_lengths)/len(text_lengths):.1f}")
    
    # Phân tích unique entities và values
    unique_entities = set(entity_texts)
    unique_values = set(value_texts)
    print(f"\nUnique entities: {len(unique_entities)}")
    print(f"Unique values: {len(unique_values)}")
    
    # Phân tích một số mẫu
    print(f"\n=== MẪU DỮ LIỆU ===")
    for i, item in enumerate(data[:5]):
        print(f"\nMẫu {i+1}:")
        print(f"  Input: {item['input']}")
        print(f"  Command: {item['command']}")
        print(f"  Entities: {item.get('entities', [])}")
        print(f"  Values: {item.get('values', [])}")

if __name__ == "__main__":
    analyze_new_dataset() 