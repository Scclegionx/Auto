#!/usr/bin/env python3
"""
Script để fix file elderly_command_dataset_reduced.json
"""

import json
import os
from pathlib import Path
import shutil
from datetime import datetime

def fix_dataset_reduced():
    """Fix file elderly_command_dataset_reduced.json"""
    file_path = Path("src/data/raw/elderly_command_dataset_reduced.json")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    print(f"Processing: {file_path}")
    
    # Create backup
    backup_dir = Path("src/data/backup")
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"elderly_command_dataset_reduced_fix_backup_{timestamp}.json"
    shutil.copy2(file_path, backup_file)
    print(f"Backup created: {backup_file}")
    
    try:
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples")
        
        # Count changes
        make_call_count = 0
        send_message_count = 0
        
        # Process each sample
        for item in data:
            if item.get("command") == "make-call":
                item["command"] = "call"
                make_call_count += 1
            elif item.get("command") == "send-message":
                item["command"] = "send-mess"
                send_message_count += 1
        
        print(f"Changed {make_call_count} make-call to call")
        print(f"Changed {send_message_count} send-message to send-mess")
        
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Fixed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("FIXING DATASET REDUCED")
    print("=" * 60)
    
    if fix_dataset_reduced():
        print("Dataset reduced fixed successfully!")
    else:
        print("Failed to fix dataset reduced")
    
    print("\nFIXING COMPLETED!")

if __name__ == "__main__":
    main()

