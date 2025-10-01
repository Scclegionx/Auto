#!/usr/bin/env python3
"""
Script để normalize commands trong dataset
- Thay thế make-call -> call
- Thay thế send-message -> send-mess
- Loại bỏ trùng lặp và đảm bảo consistency
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import shutil
from datetime import datetime

class CommandNormalizer:
    def __init__(self):
        self.raw_dir = Path("src/data/raw")
        self.grouped_dir = Path("src/data/grouped")
        self.backup_dir = Path("src/data/backup")
        
        # Command normalization mapping
        self.normalization_map = {
            "make-call": "call",
            "send-message": "send-mess"
        }
        
        # Files to process
        self.files_to_process = [
            self.raw_dir / "elderly_command_dataset_expanded.json",
            self.grouped_dir / "call_commands.json",
            self.grouped_dir / "message_commands.json",
            self.grouped_dir / "media_commands.json",
            self.grouped_dir / "search_commands.json",
            self.grouped_dir / "reminder_commands.json",
            self.grouped_dir / "check_commands.json",
            self.grouped_dir / "content_commands.json",
            self.grouped_dir / "app_commands.json",
            self.grouped_dir / "control_commands.json",
            self.grouped_dir / "tutorial_commands.json",
            self.grouped_dir / "social_media_commands.json",
            self.grouped_dir / "conversation_commands.json"
        ]
    
    def create_backup(self, file_path):
        """Tạo backup cho file trước khi normalize"""
        if file_path.exists():
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{file_path.stem}_backup_{timestamp}.json"
            shutil.copy2(file_path, backup_file)
            print(f"Backup created: {backup_file}")
            return True
        return False
    
    def normalize_sample(self, sample):
        """Normalize một sample"""
        if 'command' in sample:
            original_command = sample['command']
            normalized_command = self.normalization_map.get(original_command, original_command)
            
            if original_command != normalized_command:
                sample['command'] = normalized_command
                sample['original_command'] = original_command  # Keep track of original
                return True
        return False
    
    def normalize_file(self, file_path):
        """Normalize một file JSON"""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        print(f"\nProcessing: {file_path}")
        
        # Create backup
        self.create_backup(file_path)
        
        try:
            # Load data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Invalid data format in {file_path}")
                return False
            
            # Normalize samples
            normalized_count = 0
            command_changes = Counter()
            
            for sample in data:
                if self.normalize_sample(sample):
                    normalized_count += 1
                    original = sample.get('original_command', '')
                    new = sample['command']
                    command_changes[f"{original} -> {new}"] += 1
            
            # Save normalized data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Normalized {normalized_count} samples")
            if command_changes:
                print("Command changes:")
                for change, count in command_changes.most_common():
                    print(f"   {change}: {count} samples")
            
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def normalize_all_files(self):
        """Normalize tất cả files"""
        print("COMMAND NORMALIZATION")
        print("=" * 60)
        print(f"Normalization mapping: {self.normalization_map}")
        print()
        
        success_count = 0
        total_files = 0
        
        for file_path in self.files_to_process:
            total_files += 1
            if self.normalize_file(file_path):
                success_count += 1
        
        print(f"\nSUMMARY")
        print(f"   Files processed: {success_count}/{total_files}")
        print(f"   Success rate: {(success_count/total_files)*100:.1f}%")
        
        if success_count == total_files:
            print("All files normalized successfully!")
        else:
            print("Some files failed to normalize")
        
        return success_count == total_files
    
    def update_management_scripts(self):
        """Cập nhật các management scripts để sử dụng normalized commands"""
        print("\nUPDATING MANAGEMENT SCRIPTS")
        print("=" * 50)
        
        # Update dataset_merger.py
        merger_file = Path("scripts/management/dataset_merger.py")
        if merger_file.exists():
            try:
                with open(merger_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update command groups mapping
                old_mapping = """        command_groups = {
            'call': 'call_commands',
            'make-call': 'call_commands',
            'make-video-call': 'call_commands',
            'send-mess': 'message_commands',
            'send-message': 'message_commands',"""
                
                new_mapping = """        command_groups = {
            'call': 'call_commands',
            'make-video-call': 'call_commands',
            'send-mess': 'message_commands',"""
                
                if old_mapping in content:
                    content = content.replace(old_mapping, new_mapping)
                    
                    with open(merger_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("Updated dataset_merger.py")
                else:
                    print("Could not find mapping in dataset_merger.py")
                    
            except Exception as e:
                print(f"Error updating dataset_merger.py: {e}")
        
        # Update organize_dataset.py
        organizer_file = Path("scripts/management/organize_dataset.py")
        if organizer_file.exists():
            try:
                with open(organizer_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update command groups mapping
                old_mapping = """        self.command_groups = {
            'call': 'CALL_COMMANDS',
            'make-call': 'CALL_COMMANDS', 
            'make-video-call': 'CALL_COMMANDS',
            'send-mess': 'MESSAGE_COMMANDS',
            'send-message': 'MESSAGE_COMMANDS',"""
                
                new_mapping = """        self.command_groups = {
            'call': 'CALL_COMMANDS',
            'make-video-call': 'CALL_COMMANDS',
            'send-mess': 'MESSAGE_COMMANDS',"""
                
                if old_mapping in content:
                    content = content.replace(old_mapping, new_mapping)
                    
                    with open(organizer_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("Updated organize_dataset.py")
                else:
                    print("Could not find mapping in organize_dataset.py")
                    
            except Exception as e:
                print(f"Error updating organize_dataset.py: {e}")
        
        # Update expand_dataset.py
        expander_file = Path("src/data/augmented/expand_dataset.py")
        if expander_file.exists():
            try:
                with open(expander_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update intent mapping
                old_mapping = """        intent_mapping = {
            'make-call': 'call',
            'send-message': 'send-mess',
            'make-video-call': 'make-video-call'
        }"""
                
                new_mapping = """        intent_mapping = {
            'make-video-call': 'make-video-call'
        }"""
                
                if old_mapping in content:
                    content = content.replace(old_mapping, new_mapping)
                    
                    with open(expander_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("Updated expand_dataset.py")
                else:
                    print("Could not find mapping in expand_dataset.py")
                    
            except Exception as e:
                print(f"Error updating expand_dataset.py: {e}")
    
    def generate_normalization_report(self):
        """Tạo báo cáo normalization"""
        print("\nGENERATING NORMALIZATION REPORT")
        print("=" * 50)
        
        report = {
            "normalization_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "normalization_mapping": self.normalization_map,
                "files_processed": len(self.files_to_process)
            },
            "command_statistics": {},
            "files_status": {}
        }
        
        # Analyze each file
        for file_path in self.files_to_process:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        command_counts = Counter(item.get('command', 'unknown') for item in data)
                        report["command_statistics"][str(file_path)] = dict(command_counts)
                        report["files_status"][str(file_path)] = "processed"
                    else:
                        report["files_status"][str(file_path)] = "invalid_format"
                        
                except Exception as e:
                    report["files_status"][str(file_path)] = f"error: {str(e)}"
            else:
                report["files_status"][str(file_path)] = "not_found"
        
        # Save report
        report_file = self.grouped_dir / "normalization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Normalization report saved to {report_file}")
        
        # Print summary
        print("\nCOMMAND DISTRIBUTION SUMMARY")
        print("-" * 40)
        
        all_commands = Counter()
        for file_stats in report["command_statistics"].values():
            for cmd, count in file_stats.items():
                all_commands[cmd] += count
        
        for cmd, count in all_commands.most_common():
            print(f"   {cmd}: {count} samples")
    
    def run_full_normalization(self):
        """Chạy toàn bộ quá trình normalization"""
        print("COMMAND NORMALIZER")
        print("=" * 60)
        
        # Step 1: Normalize all files
        if self.normalize_all_files():
            print("\nStep 1: File normalization completed")
        else:
            print("\nStep 1: File normalization failed")
            return False
        
        # Step 2: Update management scripts
        self.update_management_scripts()
        print("\nStep 2: Management scripts updated")
        
        # Step 3: Generate report
        self.generate_normalization_report()
        print("\nStep 3: Normalization report generated")
        
        print("\nCOMMAND NORMALIZATION COMPLETED!")
        print("Files updated:")
        for file_path in self.files_to_process:
            if file_path.exists():
                print(f"   - {file_path}")
        print(f"   - Backup files in {self.backup_dir}/")
        print(f"   - Report: {self.grouped_dir}/normalization_report.json")
        
        return True

def main():
    normalizer = CommandNormalizer()
    normalizer.run_full_normalization()

if __name__ == "__main__":
    main()
