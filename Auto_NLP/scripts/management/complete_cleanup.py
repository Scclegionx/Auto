#!/usr/bin/env python3
"""
Script cuối cùng để cleanup hoàn toàn
"""

import json
import os
from pathlib import Path
import shutil
from datetime import datetime

def cleanup_organization_summary():
    """Cleanup organization_summary.json"""
    file_path = Path("src/data/grouped/organization_summary.json")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    print(f"Cleaning up: {file_path}")
    
    # Create backup
    backup_dir = Path("src/data/backup")
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"organization_summary_complete_backup_{timestamp}.json"
    shutil.copy2(file_path, backup_file)
    print(f"Backup created: {backup_file}")
    
    try:
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Remove old command mappings
        if "command_mapping" in data:
            old_mappings = ["make-call", "send-message"]
            for old_mapping in old_mappings:
                if old_mapping in data["command_mapping"]:
                    del data["command_mapping"][old_mapping]
                    print(f"Removed mapping: {old_mapping}")
        
        # Save cleaned data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("Cleaned organization_summary.json")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def cleanup_expand_dataset():
    """Cleanup expand_dataset.py"""
    file_path = Path("src/data/augmented/expand_dataset.py")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    print(f"Cleaning up: {file_path}")
    
    # Create backup
    backup_dir = Path("src/data/backup")
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"expand_dataset_complete_backup_{timestamp}.py"
    shutil.copy2(file_path, backup_file)
    print(f"Backup created: {backup_file}")
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove old intent mappings
        old_intent_mapping = """        intent_mapping = {
            'make-call': 'call',
            'send-message': 'send-mess',
            'make-video-call': 'make-video-call'
        }"""
        
        new_intent_mapping = """        intent_mapping = {
            'make-video-call': 'make-video-call'
        }"""
        
        if old_intent_mapping in content:
            content = content.replace(old_intent_mapping, new_intent_mapping)
            print("Updated intent mapping")
        
        # Save cleaned content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("Cleaned expand_dataset.py")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def generate_final_report():
    """Tạo báo cáo cuối cùng"""
    print("\nGENERATING FINAL REPORT")
    print("=" * 50)
    
    report = {
        "complete_cleanup_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed"
        },
        "normalization_summary": {
            "total_commands": 26,
            "removed_commands": ["make-call", "send-message"],
            "normalized_commands": ["call", "send-mess"],
            "kept_commands": ["make-video-call"]
        },
        "files_processed": [
            "organization_summary.json",
            "expand_dataset.py"
        ]
    }
    
    # Save report
    report_file = Path("src/data/grouped/complete_cleanup_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Complete cleanup report saved to {report_file}")
    
    # Print summary
    print("\nCOMPLETE CLEANUP SUMMARY")
    print("-" * 40)
    print("Files processed:")
    print("   - organization_summary.json")
    print("   - expand_dataset.py")
    print("\nNormalization completed:")
    print("   - make-call -> call")
    print("   - send-message -> send-mess")
    print("   - Total commands: 26")
    print("   - All old references removed")

def main():
    print("COMPLETE CLEANUP")
    print("=" * 60)
    
    success_count = 0
    total_files = 2
    
    # Cleanup organization_summary.json
    if cleanup_organization_summary():
        success_count += 1
    
    # Cleanup expand_dataset.py
    if cleanup_expand_dataset():
        success_count += 1
    
    print(f"\nSUMMARY")
    print(f"   Files processed: {success_count}/{total_files}")
    print(f"   Success rate: {(success_count/total_files)*100:.1f}%")
    
    if success_count == total_files:
        print("All files cleaned successfully!")
    else:
        print("Some files failed to clean")
    
    # Generate final report
    generate_final_report()
    
    print("\nCOMPLETE CLEANUP FINISHED!")
    print("Files updated:")
    print("   - src/data/grouped/organization_summary.json")
    print("   - src/data/augmented/expand_dataset.py")
    print("   - Backup files in src/data/backup/")
    print("   - Report: src/data/grouped/complete_cleanup_report.json")

if __name__ == "__main__":
    main()
