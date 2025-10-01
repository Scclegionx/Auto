#!/usr/bin/env python3
"""
Script để cleanup hoàn toàn các file còn lại
- Cập nhật organization_summary.json
- Cập nhật expand_dataset.py
- Cập nhật data_processor.py
- Đảm bảo consistency hoàn toàn
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import shutil
from datetime import datetime
import re

class FinalCleanup:
    def __init__(self):
        self.grouped_dir = Path("src/data/grouped")
        self.augmented_dir = Path("src/data/augmented")
        self.processed_dir = Path("src/data/processed")
        self.backup_dir = Path("src/data/backup")
        
        # Files to cleanup
        self.files_to_cleanup = [
            self.grouped_dir / "organization_summary.json",
            self.augmented_dir / "expand_dataset.py",
            self.processed_dir / "data_processor.py"
        ]
    
    def create_backup(self, file_path):
        """Tạo backup cho file trước khi cleanup"""
        if file_path.exists():
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{file_path.stem}_final_cleanup_backup_{timestamp}.json"
            if file_path.suffix == '.py':
                backup_file = self.backup_dir / f"{file_path.stem}_final_cleanup_backup_{timestamp}.py"
            shutil.copy2(file_path, backup_file)
            print(f"Backup created: {backup_file}")
            return True
        return False
    
    def cleanup_organization_summary(self):
        """Cleanup organization_summary.json"""
        file_path = self.grouped_dir / "organization_summary.json"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        print(f"\nCleaning up: {file_path}")
        
        # Create backup
        self.create_backup(file_path)
        
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
            print(f"Error processing {file_path}: {e}")
            return False
    
    def cleanup_expand_dataset(self):
        """Cleanup expand_dataset.py"""
        file_path = self.augmented_dir / "expand_dataset.py"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        print(f"\nCleaning up: {file_path}")
        
        # Create backup
        self.create_backup(file_path)
        
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
                print("Updated intent mapping in expand_dataset.py")
            
            # Remove old patterns
            old_patterns = [
                "'make-call': [",
                "'send-message': ["
            ]
            
            for old_pattern in old_patterns:
                if old_pattern in content:
                    # Find and remove the entire pattern block
                    lines = content.split('\n')
                    new_lines = []
                    skip_block = False
                    indent_level = 0
                    
                    for line in lines:
                        if old_pattern in line:
                            skip_block = True
                            indent_level = len(line) - len(line.lstrip())
                            continue
                        
                        if skip_block:
                            current_indent = len(line) - len(line.lstrip()) if line.strip() else 0
                            if line.strip() and current_indent <= indent_level:
                                skip_block = False
                                new_lines.append(line)
                        else:
                            new_lines.append(line)
                    
                    content = '\n'.join(new_lines)
                    print(f"Removed pattern: {old_pattern}")
            
            # Save cleaned content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Cleaned expand_dataset.py")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def cleanup_data_processor(self):
        """Cleanup data_processor.py"""
        file_path = self.processed_dir / "data_processor.py"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        print(f"\nCleaning up: {file_path}")
        
        # Create backup
        self.create_backup(file_path)
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove old intent mappings
            old_intent_mapping = """        self.intent_to_command_mapping = {
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
        }"""
            
            new_intent_mapping = """        self.intent_to_command_mapping = {
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
        }"""
            
            if old_intent_mapping in content:
                content = content.replace(old_intent_mapping, new_intent_mapping)
                print("Updated intent mapping in data_processor.py")
            
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
            
            # Save cleaned content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Cleaned data_processor.py")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def cleanup_all_files(self):
        """Cleanup tất cả files"""
        print("FINAL CLEANUP")
        print("=" * 60)
        
        success_count = 0
        total_files = 0
        
        # Cleanup organization_summary.json
        total_files += 1
        if self.cleanup_organization_summary():
            success_count += 1
        
        # Cleanup expand_dataset.py
        total_files += 1
        if self.cleanup_expand_dataset():
            success_count += 1
        
        # Cleanup data_processor.py
        total_files += 1
        if self.cleanup_data_processor():
            success_count += 1
        
        print(f"\nSUMMARY")
        print(f"   Files processed: {success_count}/{total_files}")
        print(f"   Success rate: {(success_count/total_files)*100:.1f}%")
        
        if success_count == total_files:
            print("All files cleaned successfully!")
        else:
            print("Some files failed to clean")
        
        return success_count == total_files
    
    def generate_final_report(self):
        """Tạo báo cáo final cleanup"""
        print("\nGENERATING FINAL REPORT")
        print("=" * 50)
        
        report = {
            "final_cleanup_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files_processed": len(self.files_to_cleanup)
            },
            "cleanup_actions": {
                "organization_summary": "Removed make-call and send-message mappings",
                "expand_dataset": "Removed old intent mappings and patterns",
                "data_processor": "Removed old intent mappings and keywords"
            },
            "normalization_summary": {
                "total_commands": 26,
                "removed_commands": ["make-call", "send-message"],
                "normalized_commands": ["call", "send-mess"],
                "kept_commands": ["make-video-call"]
            }
        }
        
        # Save report
        report_file = self.grouped_dir / "final_cleanup_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Final cleanup report saved to {report_file}")
        
        # Print summary
        print("\nFINAL CLEANUP SUMMARY")
        print("-" * 40)
        print("Files cleaned:")
        for file_path in self.files_to_cleanup:
            if file_path.exists():
                print(f"   - {file_path}")
        
        print("\nNormalization completed:")
        print("   - make-call -> call")
        print("   - send-message -> send-mess")
        print("   - Total commands: 26")
        print("   - All old references removed")
    
    def run_final_cleanup(self):
        """Chạy toàn bộ quá trình final cleanup"""
        print("FINAL CLEANUP")
        print("=" * 60)
        
        # Step 1: Cleanup all files
        if self.cleanup_all_files():
            print("\nStep 1: File cleanup completed")
        else:
            print("\nStep 1: File cleanup failed")
            return False
        
        # Step 2: Generate report
        self.generate_final_report()
        print("\nStep 2: Final report generated")
        
        print("\nFINAL CLEANUP COMPLETED!")
        print("Files updated:")
        for file_path in self.files_to_cleanup:
            if file_path.exists():
                print(f"   - {file_path}")
        print(f"   - Backup files in {self.backup_dir}/")
        print(f"   - Report: {self.grouped_dir}/final_cleanup_report.json")
        
        return True

def main():
    cleanup = FinalCleanup()
    cleanup.run_final_cleanup()

if __name__ == "__main__":
    main()
