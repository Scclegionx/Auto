#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script để sync dataset dễ dàng
"""

import sys
from pathlib import Path

def show_help():
    """Hiển thị hướng dẫn sử dụng"""
    print("🔄 DATASET SYNC HELPER")
    print("=" * 50)
    print("Usage:")
    print("  python sync_dataset.py                    # Merge tất cả groups")
    print("  python sync_dataset.py <group_name>       # Sync group cụ thể")
    print("  python sync_dataset.py --help             # Hiển thị help")
    print("  python sync_dataset.py --list              # Liệt kê groups")
    print()
    print("Available groups:")
    print("  - call_commands")
    print("  - message_commands") 
    print("  - media_commands")
    print("  - search_commands")
    print("  - reminder_commands")
    print("  - check_commands")
    print("  - content_commands")
    print("  - app_commands")
    print("  - control_commands")
    print("  - tutorial_commands")
    print("  - social_media_commands")
    print("  - conversation_commands")

def list_groups():
    """Liệt kê các group files có sẵn"""
    grouped_dir = Path("src/data/grouped")
    
    if not grouped_dir.exists():
        print("❌ No grouped directory found")
        return
    
    group_files = list(grouped_dir.glob("*_commands.json"))
    
    if not group_files:
        print("❌ No group files found")
        return
    
    print("📁 Available groups:")
    for group_file in sorted(group_files):
        group_name = group_file.stem
        try:
            import json
            with open(group_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  ✅ {group_name:25} | {len(data):3d} samples")
        except Exception as e:
            print(f"  ❌ {group_name:25} | Error: {e}")

def main():
    if len(sys.argv) == 1:
        # No arguments - run full merge
        from dataset_merger import DatasetMerger
        merger = DatasetMerger()
        merger.run_full_merge()
        
    elif sys.argv[1] == "--help":
        show_help()
        
    elif sys.argv[1] == "--list":
        list_groups()
        
    else:
        # Sync specific group
        group_name = sys.argv[1]
        from dataset_merger import DatasetMerger
        merger = DatasetMerger()
        merger.run_sync_group(group_name)

if __name__ == "__main__":
    main()
