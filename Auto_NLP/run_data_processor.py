#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để chạy data processor với dataset mới
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configs
from src.training.configs.config import ModelConfig, IntentConfig, EntityConfig, ValueConfig, CommandConfig, TrainingConfig

# Create config instances
model_config = ModelConfig()
intent_config = IntentConfig()
entity_config = EntityConfig()
value_config = ValueConfig()
command_config = CommandConfig()
training_config = TrainingConfig()

# Import data processor
from src.data.processed.data_processor import DataProcessor

def main():
    """Chạy data processor với dataset mới"""
    print("🔄 PROCESSING DATASET WITH NEW ENTITY MAPPING")
    print("=" * 60)
    
    # Paths
    raw_dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT_UPDATED.json"
    processed_dir = "src/data/processed"
    
    # Check if raw dataset exists
    if not os.path.exists(raw_dataset_path):
        print(f"❌ Raw dataset not found: {raw_dataset_path}")
        return False
    
    print(f"📁 Raw dataset: {raw_dataset_path}")
    print(f"📁 Processed dir: {processed_dir}")
    
    try:
        # Create data processor
        processor = DataProcessor()
        
        # Load raw dataset
        print("\n📖 Loading raw dataset...")
        raw_data = processor.load_dataset(raw_dataset_path)
        print(f"✅ Loaded {len(raw_data)} samples")
        
        # Process data
        print("\n🔄 Processing data...")
        processed_data = processor.prepare_multi_task_data(raw_data)
        print(f"✅ Processed {len(processed_data)} samples")
        
        # Split data
        print("\n✂️ Splitting data...")
        train_data, val_data = processor.split_dataset(processed_data, train_ratio=0.8)
        test_data = val_data  # Use val as test for now
        print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors_to_lists(data):
            """Convert PyTorch tensors to lists for JSON serialization"""
            converted_data = []
            for item in data:
                converted_item = {}
                for key, value in item.items():
                    if hasattr(value, 'tolist'):  # PyTorch tensor
                        converted_item[key] = value.tolist()
                    elif isinstance(value, list):
                        converted_item[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                    else:
                        converted_item[key] = value
                converted_data.append(converted_item)
            return converted_data
        
        print("\n🔄 Converting tensors to lists...")
        train_data_converted = convert_tensors_to_lists(train_data)
        val_data_converted = convert_tensors_to_lists(val_data)
        test_data_converted = convert_tensors_to_lists(test_data)
        
        # Save processed data
        print("\n💾 Saving processed data...")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save train data
        train_file = os.path.join(processed_dir, "train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(train_data_converted, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved train data: {train_file}")
        
        # Save val data
        val_file = os.path.join(processed_dir, "val.json")
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data_converted, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved val data: {val_file}")
        
        # Save test data
        test_file = os.path.join(processed_dir, "test.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data_converted, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved test data: {test_file}")
        
        print("\n🎉 DATA PROCESSING COMPLETED!")
        print("=" * 60)
        print("You can now run training with:")
        print("python src/training/scripts/train_gpu.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
