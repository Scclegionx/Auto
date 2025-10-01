#!/usr/bin/env python3
"""
Script để fix các file còn lại
"""

import os
import re
from pathlib import Path
import shutil
from datetime import datetime

def create_backup(file_path):
    """Tạo backup cho file"""
    if file_path.exists():
        backup_dir = Path("src/data/backup")
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{file_path.stem}_fix_backup_{timestamp}.py"
        if file_path.suffix == '.json':
            backup_file = backup_dir / f"{file_path.stem}_fix_backup_{timestamp}.json"
        shutil.copy2(file_path, backup_file)
        print(f"Backup created: {backup_file}")
        return True
    return False

def fix_file(file_path):
    """Fix một file"""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    print(f"Processing: {file_path}")
    
    # Create backup
    create_backup(file_path)
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Apply normalizations
        content = content.replace('"make-call"', '"call"')
        content = content.replace("'make-call'", "'call'")
        content = content.replace('"send-message"', '"send-mess"')
        content = content.replace("'send-message'", "'send-mess'")
        
        # Replace in comments
        content = content.replace("# make-call", "# call")
        content = content.replace("# send-message", "# send-mess")
        
        # Replace in variable names
        content = re.sub(r'\bmake-call\b', 'call', content)
        content = re.sub(r'\bsend-message\b', 'send-mess', content)
        
        # Save if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("FIXING REMAINING FILES")
    print("=" * 60)
    
    # Files to fix
    files_to_fix = [
        "src/inference/engines/intent_predictor.py",
        "src/inference/engines/nlp_processor.py", 
        "src/inference/api/api_server.py",
        "src/inference/engines/value_generator.py",
        "src/inference/engines/communication_optimizer.py",
        "src/data/processed/data_processor.py"
    ]
    
    success_count = 0
    total_files = len(files_to_fix)
    
    for file_path_str in files_to_fix:
        file_path = Path(file_path_str)
        if fix_file(file_path):
            success_count += 1
    
    print(f"\nSUMMARY")
    print(f"   Files processed: {success_count}/{total_files}")
    print(f"   Success rate: {(success_count/total_files)*100:.1f}%")
    
    if success_count == total_files:
        print("All files fixed successfully!")
    else:
        print("Some files failed to fix")
    
    print("\nFIXING COMPLETED!")
    print("All files have been normalized to use consistent command labels.")

if __name__ == "__main__":
    main()
