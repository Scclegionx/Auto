import json
import os
from pathlib import Path
from collections import defaultdict
import shutil

class DatasetMerger:
    def __init__(self):
        self.grouped_dir = Path("src/data/grouped")
        self.raw_dir = Path("src/data/raw")
        self.main_file = self.raw_dir / "elderly_command_dataset_expanded.json"
        self.backup_dir = Path("src/data/backup")
        
    def create_backup(self):
        """Tạo backup file tổng trước khi merge"""
        if self.main_file.exists():
            self.backup_dir.mkdir(exist_ok=True)
            backup_file = self.backup_dir / f"elderly_command_dataset_expanded_backup_{self._get_timestamp()}.json"
            shutil.copy2(self.main_file, backup_file)
            print(f"✅ Backup created: {backup_file}")
            return True
        return False
    
    def _get_timestamp(self):
        """Lấy timestamp cho backup file"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_all_groups(self):
        """Load tất cả group files"""
        all_data = []
        group_stats = {}
        
        if not self.grouped_dir.exists():
            print(f"❌ Grouped directory not found: {self.grouped_dir}")
            return None, None
        
        # Load từng group file
        group_files = list(self.grouped_dir.glob("*_commands.json"))
        
        if not group_files:
            print(f"❌ No group files found in {self.grouped_dir}")
            return None, None
        
        print(f"📂 Found {len(group_files)} group files")
        
        for group_file in group_files:
            try:
                with open(group_file, 'r', encoding='utf-8') as f:
                    group_data = json.load(f)
                
                group_name = group_file.stem
                all_data.extend(group_data)
                group_stats[group_name] = len(group_data)
                
                print(f"✅ {group_name:25} | {len(group_data):3d} samples")
                
            except Exception as e:
                print(f"❌ Error loading {group_file}: {e}")
                continue
        
        return all_data, group_stats
    
    def validate_data(self, data):
        """Validate dữ liệu trước khi merge"""
        if not data:
            print("❌ No data to validate")
            return False
        
        print(f"\n🔍 VALIDATING DATA")
        print("=" * 50)
        
        # Kiểm tra cấu trúc
        required_fields = ['input', 'command']
        missing_fields = 0
        invalid_samples = 0
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"❌ Sample {i}: Not a dictionary")
                invalid_samples += 1
                continue
                
            for field in required_fields:
                if field not in item:
                    print(f"❌ Sample {i}: Missing field '{field}'")
                    missing_fields += 1
                    break
        
        print(f"📊 Validation Results:")
        print(f"   Total samples: {len(data)}")
        print(f"   Missing fields: {missing_fields}")
        print(f"   Invalid samples: {invalid_samples}")
        
        if missing_fields > 0 or invalid_samples > 0:
            print("❌ Validation failed!")
            return False
        
        print("✅ Validation passed!")
        return True
    
    def merge_groups(self):
        """Merge tất cả group files thành file tổng"""
        print("🔄 MERGING GROUP FILES TO MAIN DATASET")
        print("=" * 60)
        
        # Tạo backup
        self.create_backup()
        
        # Load tất cả group data
        all_data, group_stats = self.load_all_groups()
        
        if all_data is None:
            print("❌ Failed to load group data")
            return False
        
        # Validate data
        if not self.validate_data(all_data):
            print("❌ Data validation failed")
            return False
        
        # Shuffle data để tránh bias
        import random
        random.shuffle(all_data)
        
        # Lưu file tổng
        try:
            with open(self.main_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Merged dataset saved to {self.main_file}")
            print(f"📊 Total samples: {len(all_data)}")
            
        except Exception as e:
            print(f"❌ Error saving merged dataset: {e}")
            return False
        
        # Tạo summary
        self.create_merge_summary(all_data, group_stats)
        
        return True
    
    def create_merge_summary(self, data, group_stats):
        """Tạo summary file cho merge operation"""
        from collections import Counter
        
        # Thống kê command
        command_counts = Counter(item['command'] for item in data)
        
        # Thống kê entity
        entity_counts = Counter()
        samples_with_entities = 0
        
        for item in data:
            if 'entities' in item and item['entities']:
                samples_with_entities += 1
                for entity in item['entities']:
                    label = entity.get('label', 'UNKNOWN')
                    entity_counts[label] += 1
        
        # Thống kê value
        value_counts = Counter()
        samples_with_values = 0
        
        for item in data:
            if 'values' in item and item['values']:
                samples_with_values += 1
                for value in item['values']:
                    label = value.get('label', 'UNKNOWN')
                    value_counts[label] += 1
        
        summary = {
            "merge_info": {
                "timestamp": self._get_timestamp(),
                "total_samples": len(data),
                "total_groups": len(group_stats),
                "group_files": group_stats
            },
            "command_distribution": dict(command_counts),
            "entity_stats": {
                "samples_with_entities": samples_with_entities,
                "total_entities": sum(entity_counts.values()),
                "entity_types": len(entity_counts),
                "top_entities": dict(entity_counts.most_common(10))
            },
            "value_stats": {
                "samples_with_values": samples_with_values,
                "total_values": sum(value_counts.values()),
                "value_types": len(value_counts),
                "top_values": dict(value_counts.most_common(10))
            }
        }
        
        summary_file = self.grouped_dir / "merge_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Merge summary saved to {summary_file}")
    
    def sync_single_group(self, group_name):
        """Sync một group cụ thể với file tổng"""
        print(f"🔄 SYNCING GROUP: {group_name}")
        print("=" * 50)
        
        group_file = self.grouped_dir / f"{group_name}.json"
        
        if not group_file.exists():
            print(f"❌ Group file not found: {group_file}")
            return False
        
        # Load group data
        try:
            with open(group_file, 'r', encoding='utf-8') as f:
                group_data = json.load(f)
            print(f"✅ Loaded {len(group_data)} samples from {group_name}")
        except Exception as e:
            print(f"❌ Error loading group file: {e}")
            return False
        
        # Load main dataset
        main_data = []
        if self.main_file.exists():
            try:
                with open(self.main_file, 'r', encoding='utf-8') as f:
                    main_data = json.load(f)
            except Exception as e:
                print(f"❌ Error loading main dataset: {e}")
                return False
        
        # Remove old samples of this group
        main_data = [item for item in main_data if not self._is_group_sample(item, group_name)]
        
        # Add new group data
        main_data.extend(group_data)
        
        # Shuffle và save
        import random
        random.shuffle(main_data)
        
        try:
            with open(self.main_file, 'w', encoding='utf-8') as f:
                json.dump(main_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Synced {group_name} to main dataset")
            print(f"📊 Total samples: {len(main_data)}")
            return True
        except Exception as e:
            print(f"❌ Error saving synced dataset: {e}")
            return False
    
    def _is_group_sample(self, item, group_name):
        """Kiểm tra xem sample có thuộc group không"""
        command = item.get('command', '')
        
        # Mapping command to group
        command_groups = {
            'call': 'call_commands',
            'make-video-call': 'call_commands',
            'send-mess': 'message_commands',
            'play-audio': 'media_commands',
            'play-content': 'media_commands',
            'play-media': 'media_commands',
            'search-content': 'search_commands',
            'search-internet': 'search_commands',
            'set-reminder': 'reminder_commands',
            'set-alarm': 'reminder_commands',
            'check-weather': 'check_commands',
            'check-messages': 'check_commands',
            'check-device-status': 'check_commands',
            'check-health-status': 'check_commands',
            'open-app': 'app_commands',
            'open-app-action': 'app_commands',
            'read-news': 'content_commands',
            'read-content': 'content_commands',
            'view-content': 'content_commands',
            'control-device': 'control_commands',
            'adjust-settings': 'control_commands',
            'app-tutorial': 'tutorial_commands',
            'provide-instructions': 'tutorial_commands',
            'navigation-help': 'tutorial_commands',
            'browse-social-media': 'social_media_commands',
            'general-conversation': 'conversation_commands'
        }
        
        expected_group = command_groups.get(command, '')
        return expected_group == group_name
    
    def run_full_merge(self):
        """Chạy merge đầy đủ"""
        print("🚀 DATASET MERGER")
        print("=" * 60)
        
        if self.merge_groups():
            print("\n🎉 Merge completed successfully!")
            print("📁 Files updated:")
            print(f"   - {self.main_file}")
            print(f"   - {self.grouped_dir}/merge_summary.json")
            print(f"   - Backup in {self.backup_dir}/")
        else:
            print("\n❌ Merge failed!")
    
    def run_sync_group(self, group_name):
        """Sync một group cụ thể"""
        if self.sync_single_group(group_name):
            print(f"\n🎉 Group {group_name} synced successfully!")
        else:
            print(f"\n❌ Failed to sync group {group_name}")

def main():
    merger = DatasetMerger()
    
    import sys
    if len(sys.argv) > 1:
        # Sync specific group
        group_name = sys.argv[1]
        merger.run_sync_group(group_name)
    else:
        # Full merge
        merger.run_full_merge()

if __name__ == "__main__":
    main()
