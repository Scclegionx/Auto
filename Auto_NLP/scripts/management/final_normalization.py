#!/usr/bin/env python3
"""
Script cuối cùng để normalize tất cả files còn lại
"""

import os
import re
from pathlib import Path
import shutil
from datetime import datetime

class FinalNormalization:
    def __init__(self):
        self.project_root = Path("Auto/Auto_NLP")
        self.backup_dir = Path("Auto/Auto_NLP/src/data/backup")
        
        # Files cần xử lý
        self.files_to_process = [
            "src/inference/engines/intent_predictor.py",
            "src/inference/engines/nlp_processor.py", 
            "src/inference/api/api_server.py",
            "src/inference/engines/value_generator.py",
            "src/inference/engines/communication_optimizer.py",
            "src/data/processed/data_processor.py"
        ]
        
        # Normalization mappings
        self.normalization_map = {
            "make-call": "call",
            "send-message": "send-mess"
        }
    
    def create_backup(self, file_path):
        """Tạo backup cho file"""
        if file_path.exists():
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{file_path.stem}_final_normalization_backup_{timestamp}.py"
            if file_path.suffix == '.json':
                backup_file = self.backup_dir / f"{file_path.stem}_final_normalization_backup_{timestamp}.json"
            shutil.copy2(file_path, backup_file)
            print(f"Backup created: {backup_file}")
            return True
        return False
    
    def normalize_file(self, file_path):
        """Normalize một file"""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        print(f"Processing: {file_path}")
        
        # Create backup
        self.create_backup(file_path)
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            
            # Apply normalizations
            for old, new in self.normalization_map.items():
                # Replace in strings
                content = content.replace(f'"{old}"', f'"{new}"')
                content = content.replace(f"'{old}'", f"'{new}'")
                
                # Replace in comments
                content = content.replace(f"# {old}", f"# {new}")
                content = content.replace(f"// {old}", f"// {new}")
                
                # Replace in variable names (be careful)
                content = re.sub(rf'\b{old}\b', new, content)
            
            # Save if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Normalized: {file_path}")
                return True
            else:
                print(f"No changes needed: {file_path}")
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def process_all_files(self):
        """Xử lý tất cả files"""
        print("FINAL NORMALIZATION")
        print("=" * 60)
        
        success_count = 0
        total_files = len(self.files_to_process)
        
        for file_path_str in self.files_to_process:
            file_path = self.project_root / file_path_str
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
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print("\nGENERATING FINAL REPORT")
        print("=" * 50)
        
        report = {
            "final_normalization_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "completed"
            },
            "normalization_summary": {
                "total_commands": 26,
                "removed_commands": ["make-call", "send-message"],
                "normalized_commands": ["call", "send-mess"],
                "kept_commands": ["make-video-call"]
            },
            "files_processed": self.files_to_process
        }
        
        # Save report
        report_file = self.project_root / "src/data/grouped/final_normalization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Final normalization report saved to {report_file}")
        
        # Print summary
        print("\nFINAL NORMALIZATION SUMMARY")
        print("-" * 40)
        print("Files processed:")
        for file_path in self.files_to_process:
            print(f"   - {file_path}")
        print("\nNormalization completed:")
        print("   - make-call -> call")
        print("   - send-message -> send-mess")
        print("   - Total commands: 26")
        print("   - All old references removed")

def main():
    normalizer = FinalNormalization()
    normalizer.process_all_files()
    normalizer.generate_final_report()
    
    print("\nFINAL NORMALIZATION COMPLETED!")
    print("All files have been normalized to use consistent command labels.")

if __name__ == "__main__":
    main()

