import json
from pathlib import Path
from collections import defaultdict

class DatasetOrganizer:
    def __init__(self):
        self.raw_dir = Path("src/data/raw")
        self.main_file = self.raw_dir / "elderly_command_dataset_expanded.json"
        self.backup_dir = Path("src/data/backup")
        
        # Command groups mapping
        self.command_groups = {
            'call': 'CALL_COMMANDS',
            'make-call': 'CALL_COMMANDS', 
            'make-video-call': 'CALL_COMMANDS',
            'send-mess': 'MESSAGE_COMMANDS',
            'send-message': 'MESSAGE_COMMANDS',
            'play-audio': 'MEDIA_COMMANDS',
            'play-content': 'MEDIA_COMMANDS',
            'play-media': 'MEDIA_COMMANDS',
            'search-content': 'SEARCH_COMMANDS',
            'search-internet': 'SEARCH_COMMANDS',
            'set-reminder': 'REMINDER_COMMANDS',
            'set-alarm': 'REMINDER_COMMANDS',
            'check-weather': 'CHECK_COMMANDS',
            'check-messages': 'CHECK_COMMANDS',
            'check-device-status': 'CHECK_COMMANDS',
            'check-health-status': 'CHECK_COMMANDS',
            'open-app': 'APP_COMMANDS',
            'open-app-action': 'APP_COMMANDS',
            'read-news': 'CONTENT_COMMANDS',
            'read-content': 'CONTENT_COMMANDS',
            'view-content': 'CONTENT_COMMANDS',
            'control-device': 'CONTROL_COMMANDS',
            'adjust-settings': 'CONTROL_COMMANDS',
            'app-tutorial': 'TUTORIAL_COMMANDS',
            'provide-instructions': 'TUTORIAL_COMMANDS',
            'navigation-help': 'TUTORIAL_COMMANDS',
            'browse-social-media': 'SOCIAL_MEDIA_COMMANDS',
            'general-conversation': 'CONVERSATION_COMMANDS'
        }
        
        # Group order for organization
        self.group_order = [
            'CALL_COMMANDS',
            'MESSAGE_COMMANDS', 
            'MEDIA_COMMANDS',
            'SEARCH_COMMANDS',
            'REMINDER_COMMANDS',
            'CHECK_COMMANDS',
            'CONTENT_COMMANDS',
            'APP_COMMANDS',
            'CONTROL_COMMANDS',
            'TUTORIAL_COMMANDS',
            'SOCIAL_MEDIA_COMMANDS',
            'CONVERSATION_COMMANDS'
        ]
    
    def create_backup(self):
        """Tạo backup trước khi sắp xếp"""
        if self.main_file.exists():
            self.backup_dir.mkdir(exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"elderly_command_dataset_expanded_backup_{timestamp}.json"
            
            import shutil
            shutil.copy2(self.main_file, backup_file)
            print(f"✅ Backup created: {backup_file}")
            return True
        return False
    
    def load_dataset(self):
        """Load dataset từ file tổng"""
        try:
            with open(self.main_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} samples from {self.main_file}")
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
    
    def organize_by_groups(self, data):
        """Sắp xếp data theo nhóm command"""
        print("\n🔄 ORGANIZING DATASET BY COMMAND GROUPS")
        print("=" * 60)
        
        # Group data by command groups
        grouped_data = defaultdict(list)
        ungrouped_data = []
        
        for item in data:
            command = item.get('command', '')
            group = self.command_groups.get(command, 'UNKNOWN_GROUP')
            
            if group == 'UNKNOWN_GROUP':
                ungrouped_data.append(item)
                print(f"⚠️  Unknown command: {command}")
            else:
                grouped_data[group].append(item)
        
        # Organize by group order
        organized_data = []
        group_stats = {}
        
        for group in self.group_order:
            if group in grouped_data:
                group_items = grouped_data[group]
                organized_data.extend(group_items)
                group_stats[group] = len(group_items)
                print(f"✅ {group:25} | {len(group_items):3d} samples")
        
        # Add ungrouped items at the end
        if ungrouped_data:
            organized_data.extend(ungrouped_data)
            group_stats['UNKNOWN_GROUP'] = len(ungrouped_data)
            print(f"⚠️  UNKNOWN_GROUP        | {len(ungrouped_data):3d} samples")
        
        return organized_data, group_stats
    
    def save_organized_dataset(self, data, group_stats):
        """Lưu dataset đã sắp xếp"""
        try:
            with open(self.main_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Organized dataset saved to {self.main_file}")
            print(f"📊 Total samples: {len(data)}")
            
            # Save organization summary
            summary = {
                "organization_info": {
                    "total_samples": len(data),
                    "total_groups": len(group_stats),
                    "group_distribution": group_stats
                },
                "group_order": self.group_order,
                "command_mapping": self.command_groups
            }
            
            summary_file = Path("src/data/grouped/organization_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Organization summary saved to {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving organized dataset: {e}")
            return False
        
        return True
    
    def organize_dataset(self):
        """Sắp xếp toàn bộ dataset"""
        print("🔄 DATASET ORGANIZER")
        print("=" * 60)
        
        # Tạo backup
        self.create_backup()
        
        # Load dataset
        data = self.load_dataset()
        if not data:
            print("❌ No data to organize")
            return False
        
        # Organize by groups
        organized_data, group_stats = self.organize_by_groups(data)
        
        # Save organized dataset
        if self.save_organized_dataset(organized_data, group_stats):
            print("\n🎉 Dataset organization completed!")
            print("📁 Files updated:")
            print(f"   - {self.main_file}")
            print(f"   - src/data/grouped/organization_summary.json")
            print(f"   - Backup in {self.backup_dir}/")
            return True
        else:
            print("\n❌ Dataset organization failed!")
            return False

def main():
    organizer = DatasetOrganizer()
    organizer.organize_dataset()

if __name__ == "__main__":
    main()
