import json
import random
import re
from typing import List, Dict
import numpy as np

class DatasetExpander:
    def __init__(self, original_file: str = "elderly_command_dataset_reduced.json"):
        self.original_file = original_file
        self.expanded_file = "elderly_command_dataset_expanded.json"
        
    def load_original_data(self) -> List[Dict]:
        with open(self.original_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📂 Loaded {len(data)} samples from {self.original_file}")
        return data
    
    def expand_dataset(self, target_size: int = 1000) -> List[Dict]:
        original_data = self.load_original_data()
        current_size = len(original_data)
        
        if current_size >= target_size:
            print(f"✅ Dataset already has {current_size} samples, no expansion needed")
            return original_data
        
        print(f"🚀 Expanding dataset from {current_size} to {target_size} samples")

        command_counts = {}
        for item in original_data:
            command = item['command']
            command_counts[command] = command_counts.get(command, 0) + 1
        
        print(f"📊 Current command distribution:")
        for command, count in sorted(command_counts.items()):
            print(f"   {command}: {count} samples")

        total_needed = target_size - current_size
        expanded_data = original_data.copy()
        
        # Strategy 1: Augment existing samples
        print("🔄 Strategy 1: Augmenting existing samples...")
        augmented_samples = self.augment_existing_samples(original_data, total_needed // 2)
        expanded_data.extend(augmented_samples)
        
        # Strategy 2: Generate new samples
        print("🔄 Strategy 2: Generating new samples...")
        remaining_needed = target_size - len(expanded_data)
        if remaining_needed > 0:
            new_samples = self.generate_new_samples(original_data, remaining_needed)
            expanded_data.extend(new_samples)
        
        # Shuffle data
        random.shuffle(expanded_data)
        
        print(f"✅ Expanded dataset to {len(expanded_data)} samples")
        return expanded_data
    
    def augment_existing_samples(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Augment existing samples bằng cách thay đổi từ ngữ"""
        augmented = []
        
        augmentation_templates = {
            'send-mess': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}',
                'Viết tin nhắn cho {person} về {message}',
                'Gửi cho {person} tin nhắn {message}'
            ],
            'make-call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại',
                'Gọi {person} ngay bây giờ',
                'Thực hiện cuộc gọi cho {person}'
            ],
            'search-content': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}'
            ],
            'play-content': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}'
            ],
            'set-reminder': [
                'Đặt nhắc nhở {reminder}',
                'Tạo lời nhắc {reminder}',
                'Đặt lịch nhắc {reminder}',
                'Tạo nhắc nhở cho {reminder}',
                'Đặt báo thức cho {reminder}'
            ]
        }
        
        # Common entities
        persons = ['cháu Vương', 'chị Hương', 'anh Nam', 'bà nội', 'ông nội', 'mẹ', 'bố', 'em gái', 'anh trai']
        messages = ['chiều này đón bà tại công viên Thống nhất lúc 16h chiều', 'sáng mai có hẹn bác sĩ', 'tối nay về muộn', 'đã nhận được tin nhắn']
        queries = ['cách nấu phở', 'thời tiết hôm nay', 'tin tức mới nhất', 'công thức làm bánh', 'địa chỉ bệnh viện']
        contents = ['nhạc trữ tình', 'phim hài', 'video hướng dẫn', 'bài hát mới', 'tin tức thời sự']
        reminders = ['uống thuốc lúc 8h sáng', 'họp gia đình tối nay', 'đi khám bệnh ngày mai', 'gọi điện cho con']
        
        for _ in range(num_samples):
            # Chọn random sample
            sample = random.choice(data)
            command = sample['command']
            original_text = sample['input']
            
            # Tạo augmented version
            if command in augmentation_templates:
                template = random.choice(augmentation_templates[command])
                
                # Thay thế placeholders
                if '{person}' in template:
                    template = template.replace('{person}', random.choice(persons))
                if '{message}' in template:
                    template = template.replace('{message}', random.choice(messages))
                if '{query}' in template:
                    template = template.replace('{query}', random.choice(queries))
                if '{content}' in template:
                    template = template.replace('{content}', random.choice(contents))
                if '{reminder}' in template:
                    template = template.replace('{reminder}', random.choice(reminders))
                
                augmented_sample = {
                    'input': template,
                    'command': command
                }
                augmented.append(augmented_sample)
            else:
                # Nếu không có template, sử dụng synonym replacement
                augmented_text = self.replace_synonyms(original_text)
                augmented_sample = {
                    'input': augmented_text,
                    'command': command
                }
                augmented.append(augmented_sample)
        
        return augmented
    
    def replace_synonyms(self, text: str) -> str:
        """Thay thế từ đồng nghĩa"""
        synonyms = {
            'nhắn tin': ['gửi tin nhắn', 'soạn tin nhắn', 'viết tin nhắn'],
            'gọi điện': ['gọi', 'thực hiện cuộc gọi', 'liên lạc'],
            'tìm kiếm': ['tìm', 'tra cứu', 'tìm hiểu'],
            'phát': ['bật', 'mở', 'chạy', 'xem'],
            'đặt nhắc nhở': ['tạo lời nhắc', 'đặt lịch nhắc', 'tạo nhắc nhở'],
            'kiểm tra': ['xem', 'kiểm tra', 'xem xét'],
            'mở': ['bật', 'khởi động', 'chạy'],
            'tắt': ['đóng', 'dừng', 'ngắt']
        }
        
        for word, syns in synonyms.items():
            if word in text:
                text = text.replace(word, random.choice(syns))
        
        return text
    
    def generate_new_samples(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Tạo samples mới dựa trên patterns"""
        new_samples = []
        
        # Patterns cho từng command
        patterns = {
            'send-mess': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}'
            ],
            'make-call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại'
            ],
            'search-content': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}'
            ],
            'play-content': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Xem {content}'
            ],
            'set-reminder': [
                'Đặt nhắc nhở {reminder}',
                'Tạo lời nhắc {reminder}',
                'Đặt lịch nhắc {reminder}'
            ],
            'check-weather': [
                'Kiểm tra thời tiết hôm nay',
                'Xem dự báo thời tiết',
                'Thời tiết như thế nào',
                'Nhiệt độ hôm nay bao nhiêu'
            ],
            'check-messages': [
                'Kiểm tra tin nhắn từ {person}',
                'Xem tin nhắn mới',
                'Đọc tin nhắn chưa đọc',
                'Kiểm tra hộp thư'
            ]
        }
        
        # Entities
        persons = ['cháu Vương', 'chị Hương', 'anh Nam', 'bà nội', 'ông nội', 'mẹ', 'bố', 'em gái', 'anh trai', 'con trai', 'con gái']
        messages = [
            'chiều này đón bà tại công viên Thống nhất lúc 16h chiều',
            'sáng mai có hẹn bác sĩ',
            'tối nay về muộn',
            'đã nhận được tin nhắn',
            'nhớ uống thuốc đúng giờ',
            'có việc gấp cần liên lạc'
        ]
        queries = [
            'cách nấu phở',
            'thời tiết hôm nay',
            'tin tức mới nhất',
            'công thức làm bánh',
            'địa chỉ bệnh viện',
            'cách sử dụng điện thoại',
            'thông tin về bệnh tiểu đường',
            'địa điểm du lịch gần đây'
        ]
        contents = [
            'nhạc trữ tình',
            'phim hài',
            'video hướng dẫn',
            'bài hát mới',
            'tin tức thời sự',
            'nhạc vàng',
            'phim hành động',
            'bài hát cũ'
        ]
        reminders = [
            'uống thuốc lúc 8h sáng',
            'họp gia đình tối nay',
            'đi khám bệnh ngày mai',
            'gọi điện cho con',
            'đi chợ sáng mai',
            'hẹn bác sĩ tuần sau'
        ]
        
        commands = list(patterns.keys())
        
        for _ in range(num_samples):
            command = random.choice(commands)
            if command in patterns:
                pattern = random.choice(patterns[command])
                
                # Thay thế placeholders
                if '{person}' in pattern:
                    pattern = pattern.replace('{person}', random.choice(persons))
                if '{message}' in pattern:
                    pattern = pattern.replace('{message}', random.choice(messages))
                if '{query}' in pattern:
                    pattern = pattern.replace('{query}', random.choice(queries))
                if '{content}' in pattern:
                    pattern = pattern.replace('{content}', random.choice(contents))
                if '{reminder}' in pattern:
                    pattern = pattern.replace('{reminder}', random.choice(reminders))
                
                new_sample = {
                    'input': pattern,
                    'command': command
                }
                new_samples.append(new_sample)
        
        return new_samples
    
    def save_expanded_data(self, data: List[Dict]):
        """Lưu dataset đã mở rộng"""
        with open(self.expanded_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved expanded dataset to {self.expanded_file}")
        print(f"📊 File size: {len(json.dumps(data, ensure_ascii=False).encode()) / 1024:.2f} KB")
        
        # Phân tích distribution
        command_counts = {}
        for item in data:
            command = item['command']
            command_counts[command] = command_counts.get(command, 0) + 1
        
        print(f"📈 Final command distribution:")
        for command, count in sorted(command_counts.items()):
            percentage = (count / len(data)) * 100
            print(f"   {command}: {count} samples ({percentage:.1f}%)")

def main():
    """Main function"""
    print("🚀 Dataset Expansion for PhoBERT-Large")
    print("=" * 50)
    
    # Tạo expander
    expander = DatasetExpander()
    
    # Mở rộng dataset đến 1000 samples (khuyến nghị cho large model)
    expanded_data = expander.expand_dataset(target_size=1000)
    
    # Lưu dataset đã mở rộng
    expander.save_expanded_data(expanded_data)
    
    print("\n🎉 Dataset expansion completed!")
    print("📋 Next steps:")
    print("   1. Use the expanded dataset for training")
    print("   2. Run: python train_gpu.py")
    print("   3. The large model will have more data to learn from")

if __name__ == "__main__":
    main()
