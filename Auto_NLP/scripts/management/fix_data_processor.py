#!/usr/bin/env python3
"""
Script để fix data_processor.py một cách an toàn
"""

import os
from pathlib import Path
import shutil
from datetime import datetime

class DataProcessorFixer:
    def __init__(self):
        self.processed_dir = Path("src/data/processed")
        self.backup_dir = Path("src/data/backup")
        self.file_path = self.processed_dir / "data_processor.py"
    
    def create_backup(self):
        """Tạo backup cho file"""
        if self.file_path.exists():
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"data_processor_fix_backup_{timestamp}.py"
            shutil.copy2(self.file_path, backup_file)
            print(f"Backup created: {backup_file}")
            return True
        return False
    
    def fix_data_processor(self):
        """Fix data_processor.py"""
        if not self.file_path.exists():
            print(f"File not found: {self.file_path}")
            return False
        
        print(f"Fixing: {self.file_path}")
        
        # Create backup
        self.create_backup()
        
        try:
            # Read file with UTF-8 encoding
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix intent mapping
            old_mapping = '''        self.intent_to_command_mapping = {
            "call": "call",
            "send-mess": "send-mess", 
            "make-video-call": "make-video-call",
            "check-messages": "check-messages",
            "open-app": "open-app",
            "play-media": "play-media",
            "search-content": "search-content",
            "set-reminder": "set-reminder",
            "set-alarm": "set-alarm",
            "check-weather": "check-weather",
            "general-conversation": "general-conversation",
            "unknown": "unknown"
        }'''
            
            new_mapping = '''        self.intent_to_command_mapping = {
            "call": "call",
            "send-mess": "send-mess", 
            "make-video-call": "make-video-call",
            "check-messages": "check-messages",
            "open-app": "open-app",
            "play-media": "play-media",
            "search-content": "search-content",
            "set-reminder": "set-reminder",
            "set-alarm": "set-alarm",
            "check-weather": "check-weather",
            "general-conversation": "general-conversation",
            "unknown": "unknown"
        }'''
            
            if old_mapping in content:
                content = content.replace(old_mapping, new_mapping)
                print("Updated intent mapping")
            
            # Remove old intent keywords
            old_keywords = [
                '"make-call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "quay số", "bấm số"],',
                '"send-message": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "soạn tin"],'
            ]
            
            for old_keyword in old_keywords:
                if old_keyword in content:
                    content = content.replace(old_keyword, "")
                    print(f"Removed keyword: {old_keyword}")
            
            # Remove old conflicting keywords
            old_conflicting = [
                '"make-call": ["nhắn tin", "gửi tin nhắn", "soạn tin"],',
                '"send-message": ["gọi", "điện thoại", "alo"],'
            ]
            
            for old_conflict in old_conflicting:
                if old_conflict in content:
                    content = content.replace(old_conflict, "")
                    print(f"Removed conflicting keyword: {old_conflict}")
            
            # Save with UTF-8 encoding
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Fixed data_processor.py")
            return True
            
        except Exception as e:
            print(f"Error processing {self.file_path}: {e}")
            return False

def main():
    fixer = DataProcessorFixer()
    fixer.fix_data_processor()

if __name__ == "__main__":
    main()
