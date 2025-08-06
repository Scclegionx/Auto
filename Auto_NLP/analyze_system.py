#!/usr/bin/env python3
"""
Script phân tích hệ thống PhoBERT_SAM
"""

import json
import os
from collections import Counter

def analyze_dataset():
    """Phân tích dataset"""
    print("=== PHÂN TÍCH DATASET ===")
    
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
    
    for item in data:
        for entity in item.get('entities', []):
            if isinstance(entity, dict):
                entity_labels.add(entity.get('label', ''))
        
        for value in item.get('values', []):
            if isinstance(value, dict):
                value_labels.add(value.get('label', ''))
    
    print(f"\nEntity labels ({len(entity_labels)}): {sorted(entity_labels)}")
    print(f"Value labels ({len(value_labels)}): {sorted(value_labels)}")
    
    # Phân tích độ dài text
    text_lengths = [len(item['input']) for item in data]
    print(f"\nĐộ dài text - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Avg: {sum(text_lengths)/len(text_lengths):.1f}")

def analyze_models():
    """Phân tích models đã train"""
    print("\n=== PHÂN TÍCH MODELS ===")
    
    model_files = [
        'models/best_intent_model.pth',
        'models/best_entity_model.pth', 
        'models/best_command_model.pth',
        'models/best_unified_model.pth'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ {model_file}: {size_mb:.1f} MB")
        else:
            print(f"✗ {model_file}: Không tồn tại")

def analyze_config():
    """Phân tích config"""
    print("\n=== PHÂN TÍCH CONFIG ===")
    
    from config import model_config, intent_config, entity_config, command_config, training_config
    
    print(f"Model: {model_config.model_name}")
    print(f"Max length: {model_config.max_length}")
    print(f"Batch size: {model_config.batch_size}")
    print(f"Learning rate: {model_config.learning_rate}")
    print(f"Epochs: {model_config.num_epochs}")
    
    print(f"\nIntent classes: {intent_config.num_intents}")
    print(f"Entity classes: {entity_config.num_entities}")
    print(f"Command classes: {command_config.num_commands}")
    
    print(f"\nDevice: {training_config.device}")
    print(f"Output dir: {training_config.output_dir}")

def analyze_structure():
    """Phân tích cấu trúc hệ thống"""
    print("\n=== PHÂN TÍCH CẤU TRÚC ===")
    
    # Kiểm tra các file chính
    main_files = [
        'main.py', 'inference.py', 'config.py', 'utils.py',
        'data/data_processor.py', 'training/trainer.py',
        'models/__init__.py', 'models/unified_model.py',
        'requirements.txt', 'README.md'
    ]
    
    for file in main_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"✓ {file}: {size_kb:.1f} KB")
        else:
            print(f"✗ {file}: Không tồn tại")
    
    # Kiểm tra thư mục
    directories = ['data', 'models', 'training', 'logs']
    for dir_name in directories:
        if os.path.exists(dir_name):
            files = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print(f"✓ {dir_name}/: {files} files")
        else:
            print(f"✗ {dir_name}/: Không tồn tại")

if __name__ == "__main__":
    analyze_dataset()
    analyze_models()
    analyze_config()
    analyze_structure() 