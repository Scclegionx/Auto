#!/usr/bin/env python3
"""
Script để cleanup hoàn toàn các commands cũ
- Loại bỏ original_command fields
- Cập nhật summary files
- Đảm bảo consistency hoàn toàn
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import shutil
from datetime import datetime

class CommandCleanup:
    def __init__(self):
        self.raw_dir = Path("src/data/raw")
        self.grouped_dir = Path("src/data/grouped")
        self.backup_dir = Path("src/data/backup")
        
        # Files to cleanup
        self.files_to_cleanup = [
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
        
        # Summary files to update
        self.summary_files = [
            self.grouped_dir / "merge_summary.json",
            self.grouped_dir / "summary.json",
            self.grouped_dir / "organization_summary.json"
        ]
    
    def create_backup(self, file_path):
        """Tạo backup cho file trước khi cleanup"""
        if file_path.exists():
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{file_path.stem}_cleanup_backup_{timestamp}.json"
            shutil.copy2(file_path, backup_file)
            print(f"Backup created: {backup_file}")
            return True
        return False
    
    def cleanup_sample(self, sample):
        """Cleanup một sample - loại bỏ original_command"""
        cleaned = False
        
        if 'original_command' in sample:
            del sample['original_command']
            cleaned = True
        
        return cleaned
    
    def cleanup_file(self, file_path):
        """Cleanup một file JSON"""
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
            
            if not isinstance(data, list):
                print(f"Invalid data format in {file_path}")
                return False
            
            # Cleanup samples
            cleaned_count = 0
            
            for sample in data:
                if self.cleanup_sample(sample):
                    cleaned_count += 1
            
            # Save cleaned data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Cleaned {cleaned_count} samples")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def update_summary_files(self):
        """Cập nhật các summary files để phản ánh normalized commands"""
        print("\nUPDATING SUMMARY FILES")
        print("=" * 50)
        
        # Load current data to get accurate counts
        all_data = []
        for file_path in self.files_to_cleanup:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # Calculate new statistics
        command_counts = Counter(item.get('command', 'unknown') for item in all_data)
        
        # Update merge_summary.json
        merge_summary_file = self.grouped_dir / "merge_summary.json"
        if merge_summary_file.exists():
            try:
                with open(merge_summary_file, 'r', encoding='utf-8') as f:
                    merge_summary = json.load(f)
                
                # Update command distribution
                merge_summary["command_distribution"] = dict(command_counts)
                merge_summary["merge_info"]["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with open(merge_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(merge_summary, f, ensure_ascii=False, indent=2)
                
                print("Updated merge_summary.json")
                
            except Exception as e:
                print(f"Error updating merge_summary.json: {e}")
        
        # Update summary.json
        summary_file = self.grouped_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                # Update command distribution
                summary["command_distribution"] = dict(command_counts)
                summary["total_commands"] = len(command_counts)
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                
                print("Updated summary.json")
                
            except Exception as e:
                print(f"Error updating summary.json: {e}")
    
    def cleanup_all_files(self):
        """Cleanup tất cả files"""
        print("COMMAND CLEANUP")
        print("=" * 60)
        
        success_count = 0
        total_files = 0
        
        for file_path in self.files_to_cleanup:
            total_files += 1
            if self.cleanup_file(file_path):
                success_count += 1
        
        print(f"\nSUMMARY")
        print(f"   Files processed: {success_count}/{total_files}")
        print(f"   Success rate: {(success_count/total_files)*100:.1f}%")
        
        if success_count == total_files:
            print("All files cleaned successfully!")
        else:
            print("Some files failed to clean")
        
        return success_count == total_files
    
    def generate_cleanup_report(self):
        """Tạo báo cáo cleanup"""
        print("\nGENERATING CLEANUP REPORT")
        print("=" * 50)
        
        report = {
            "cleanup_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files_processed": len(self.files_to_cleanup)
            },
            "command_statistics": {},
            "files_status": {}
        }
        
        # Analyze each file
        for file_path in self.files_to_cleanup:
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
        report_file = self.grouped_dir / "cleanup_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Cleanup report saved to {report_file}")
        
        # Print summary
        print("\nCOMMAND DISTRIBUTION SUMMARY")
        print("-" * 40)
        
        all_commands = Counter()
        for file_stats in report["command_statistics"].values():
            for cmd, count in file_stats.items():
                all_commands[cmd] += count
        
        for cmd, count in all_commands.most_common():
            print(f"   {cmd}: {count} samples")
    
    def run_full_cleanup(self):
        """Chạy toàn bộ quá trình cleanup"""
        print("COMMAND CLEANUP")
        print("=" * 60)
        
        # Step 1: Cleanup all files
        if self.cleanup_all_files():
            print("\nStep 1: File cleanup completed")
        else:
            print("\nStep 1: File cleanup failed")
            return False
        
        # Step 2: Update summary files
        self.update_summary_files()
        print("\nStep 2: Summary files updated")
        
        # Step 3: Generate report
        self.generate_cleanup_report()
        print("\nStep 3: Cleanup report generated")
        
        print("\nCOMMAND CLEANUP COMPLETED!")
        print("Files updated:")
        for file_path in self.files_to_cleanup:
            if file_path.exists():
                print(f"   - {file_path}")
        print(f"   - Backup files in {self.backup_dir}/")
        print(f"   - Report: {self.grouped_dir}/cleanup_report.json")
        
        return True

def main():
    cleanup = CommandCleanup()
    cleanup.run_full_cleanup()

if __name__ == "__main__":
    main()
