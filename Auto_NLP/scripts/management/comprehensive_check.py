#!/usr/bin/env python3
"""
Script kiểm tra toàn diện sau khi normalize
"""

import json
import os
from pathlib import Path
import re
from collections import Counter

class ComprehensiveChecker:
    def __init__(self):
        self.project_root = Path(".")
        self.issues = []
        self.warnings = []
        self.success_count = 0
        
    def check_syntax_errors(self):
        """Kiểm tra lỗi syntax trong Python files"""
        print("🔍 Checking syntax errors...")
        
        python_files = [
            "src/inference/engines/intent_predictor.py",
            "src/inference/engines/nlp_processor.py",
            "src/inference/api/api_server.py",
            "src/inference/engines/value_generator.py",
            "src/inference/engines/communication_optimizer.py",
            "src/data/processed/data_processor.py"
        ]
        
        for file_path in python_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    compile(content, file_path, 'exec')
                    print(f"✅ {file_path}: Syntax OK")
                    self.success_count += 1
                except SyntaxError as e:
                    self.issues.append(f"❌ {file_path}: Syntax error - {e}")
                except Exception as e:
                    self.warnings.append(f"⚠️ {file_path}: {e}")
            else:
                self.warnings.append(f"⚠️ {file_path}: File not found")
    
    def check_json_validity(self):
        """Kiểm tra tính hợp lệ của JSON files"""
        print("\n🔍 Checking JSON validity...")
        
        json_files = [
            "src/data/grouped/merge_summary.json",
            "src/data/grouped/summary.json",
            "src/data/grouped/organization_summary.json",
            "src/data/raw/elderly_command_dataset_reduced.json"
        ]
        
        for file_path in json_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"✅ {file_path}: JSON valid")
                    self.success_count += 1
                except json.JSONDecodeError as e:
                    self.issues.append(f"❌ {file_path}: JSON error - {e}")
                except Exception as e:
                    self.warnings.append(f"⚠️ {file_path}: {e}")
            else:
                self.warnings.append(f"⚠️ {file_path}: File not found")
    
    def check_command_consistency(self):
        """Kiểm tra tính nhất quán của commands"""
        print("\n🔍 Checking command consistency...")
        
        # Load label maps
        try:
            with open("models/label_maps.json", 'r', encoding='utf-8') as f:
                label_maps = json.load(f)
            
            commands = set(label_maps.get("command", []))
            intents = set(label_maps.get("intent", []))
            
            print(f"✅ Label maps loaded: {len(commands)} commands, {len(intents)} intents")
            
            # Check for old commands
            old_commands = {"make-call", "send-message"}
            found_old = commands.intersection(old_commands)
            
            if found_old:
                self.issues.append(f"❌ Found old commands in label_maps.json: {found_old}")
            else:
                print("✅ No old commands found in label_maps.json")
                self.success_count += 1
            
            # Check for missing new commands
            new_commands = {"call", "send-mess"}
            missing_new = new_commands - commands
            
            if missing_new:
                self.issues.append(f"❌ Missing new commands in label_maps.json: {missing_new}")
            else:
                print("✅ All new commands found in label_maps.json")
                self.success_count += 1
                
        except Exception as e:
            self.issues.append(f"❌ Error loading label_maps.json: {e}")
    
    def check_dataset_consistency(self):
        """Kiểm tra tính nhất quán của dataset"""
        print("\n🔍 Checking dataset consistency...")
        
        dataset_files = [
            "src/data/raw/elderly_command_dataset_reduced.json",
            "src/data/grouped/call_commands.json",
            "src/data/grouped/message_commands.json"
        ]
        
        all_commands = set()
        
        for file_path in dataset_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    file_commands = set()
                    for item in data:
                        if isinstance(item, dict) and "command" in item:
                            file_commands.add(item["command"])
                    
                    all_commands.update(file_commands)
                    
                    # Check for old commands
                    old_commands = {"make-call", "send-message"}
                    found_old = file_commands.intersection(old_commands)
                    
                    if found_old:
                        self.issues.append(f"❌ {file_path}: Found old commands {found_old}")
                    else:
                        print(f"✅ {file_path}: No old commands found")
                        self.success_count += 1
                        
                except Exception as e:
                    self.issues.append(f"❌ {file_path}: Error - {e}")
            else:
                self.warnings.append(f"⚠️ {file_path}: File not found")
        
        # Check command distribution
        print(f"📊 Total unique commands found: {len(all_commands)}")
        print(f"📊 Commands: {sorted(all_commands)}")
        
        # Check for expected commands
        expected_commands = {"call", "send-mess", "make-video-call"}
        missing_expected = expected_commands - all_commands
        
        if missing_expected:
            self.warnings.append(f"⚠️ Missing expected commands: {missing_expected}")
        else:
            print("✅ All expected commands found")
            self.success_count += 1
    
    def check_import_consistency(self):
        """Kiểm tra tính nhất quán của imports"""
        print("\n🔍 Checking import consistency...")
        
        python_files = [
            "src/inference/engines/intent_predictor.py",
            "src/inference/engines/nlp_processor.py",
            "src/inference/api/api_server.py"
        ]
        
        for file_path in python_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for old command references in strings
                    old_patterns = [
                        r'"make-call"',
                        r"'make-call'",
                        r'"send-message"',
                        r"'send-message'"
                    ]
                    
                    found_old = []
                    for pattern in old_patterns:
                        if re.search(pattern, content):
                            found_old.append(pattern)
                    
                    if found_old:
                        self.issues.append(f"❌ {file_path}: Found old command references: {found_old}")
                    else:
                        print(f"✅ {file_path}: No old command references found")
                        self.success_count += 1
                        
                except Exception as e:
                    self.warnings.append(f"⚠️ {file_path}: {e}")
            else:
                self.warnings.append(f"⚠️ {file_path}: File not found")
    
    def check_api_endpoints(self):
        """Kiểm tra API endpoints"""
        print("\n🔍 Checking API endpoints...")
        
        api_file = "src/inference/api/api_server.py"
        if Path(api_file).exists():
            try:
                with open(api_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for old command references
                old_patterns = [
                    r'"make-call"',
                    r"'make-call'",
                    r'"send-message"',
                    r"'send-message'"
                ]
                
                found_old = []
                for pattern in old_patterns:
                    if re.search(pattern, content):
                        found_old.append(pattern)
                
                if found_old:
                    self.issues.append(f"❌ {api_file}: Found old command references: {found_old}")
                else:
                    print(f"✅ {api_file}: No old command references found")
                    self.success_count += 1
                    
            except Exception as e:
                self.issues.append(f"❌ {api_file}: Error - {e}")
        else:
            self.warnings.append(f"⚠️ {api_file}: File not found")
    
    def check_backup_integrity(self):
        """Kiểm tra tính toàn vẹn của backup files"""
        print("\n🔍 Checking backup integrity...")
        
        backup_dir = Path("src/data/backup")
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("*"))
            print(f"✅ Found {len(backup_files)} backup files")
            self.success_count += 1
        else:
            self.warnings.append("⚠️ Backup directory not found")
    
    def generate_report(self):
        """Tạo báo cáo kiểm tra"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE CHECK REPORT")
        print("="*60)
        
        print(f"\n✅ Successes: {self.success_count}")
        print(f"❌ Issues: {len(self.issues)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        
        if self.issues:
            print("\n❌ ISSUES FOUND:")
            for issue in self.issues:
                print(f"   {issue}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"   {warning}")
        
        if not self.issues and not self.warnings:
            print("\n🎉 ALL CHECKS PASSED! No issues or warnings found.")
        elif not self.issues:
            print("\n✅ NO CRITICAL ISSUES! Only warnings found.")
        else:
            print("\n❌ CRITICAL ISSUES FOUND! Please review and fix.")
        
        # Save report
        report = {
            "check_timestamp": "2025-01-01 16:20:00",
            "successes": self.success_count,
            "issues": self.issues,
            "warnings": self.warnings,
            "status": "passed" if not self.issues else "failed"
        }
        
        with open("src/data/grouped/comprehensive_check_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 Report saved to: src/data/grouped/comprehensive_check_report.json")
    
    def run_all_checks(self):
        """Chạy tất cả các kiểm tra"""
        print("🔍 COMPREHENSIVE CHECK AFTER NORMALIZATION")
        print("="*60)
        
        self.check_syntax_errors()
        self.check_json_validity()
        self.check_command_consistency()
        self.check_dataset_consistency()
        self.check_import_consistency()
        self.check_api_endpoints()
        self.check_backup_integrity()
        self.generate_report()

def main():
    checker = ComprehensiveChecker()
    checker.run_all_checks()

if __name__ == "__main__":
    main()
