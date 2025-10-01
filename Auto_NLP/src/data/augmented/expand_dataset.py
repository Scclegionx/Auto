import json
import random
import re
from typing import List, Dict
import os
from pathlib import Path

class DatasetExpander:
    def __init__(self, original_file: str = "elderly_command_dataset_reduced.json"):
        # Đảm bảo path đúng
        data_dir = Path(__file__).parent.parent / "raw"
        self.original_file = data_dir / original_file
        self.expanded_file = data_dir / "elderly_command_dataset_expanded.json"
        
    def load_original_data(self) -> List[Dict]:
        try:
            with open(self.original_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📂 Loaded {len(data)} samples from {self.original_file}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {self.original_file}")
            print("🔧 Please check the file path and try again")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []
    
    def expand_dataset(self, target_size: int = 1000) -> List[Dict]:
        original_data = self.load_original_data()
        current_size = len(original_data)
        
        if current_size >= target_size:
            print(f"✅ Dataset already has {current_size} samples, no expansion needed")
            return original_data
        
        print(f"🚀 Expanding dataset from {current_size} to {target_size} samples")
        print(f"📊 Need to generate {target_size - current_size} additional samples")

        command_counts = {}
        for item in original_data:
            command = item['command']
            command_counts[command] = command_counts.get(command, 0) + 1
        
        for command, count in sorted(command_counts.items()):
            print(f"   {command}: {count} samples")

        total_needed = target_size - current_size
        expanded_data = original_data.copy()
        
        augmented_samples = self.augment_existing_samples(original_data, total_needed // 2)
        expanded_data.extend(augmented_samples)
        
        remaining_needed = target_size - len(expanded_data)
        if remaining_needed > 0:
            new_samples = self.generate_new_samples(original_data, remaining_needed)
            expanded_data.extend(new_samples)
        
        random.shuffle(expanded_data)
        
        print(f"✅ Expanded dataset to {len(expanded_data)} samples")
        return expanded_data
    
    def augment_existing_samples(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Augment existing samples bằng cách thay đổi từ ngữ"""
        augmented = []
        
        # ========================================
        # AUGMENTATION TEMPLATES BY COMMAND TYPE
        # ========================================
        
        # Intent mapping để tránh lặp lại
        intent_mapping = {
            'make-call': 'call',
            'send-message': 'send-mess',
            'make-video-call': 'make-video-call'
        }
        
        # Function để xử lý intent mapping
        def process_intent_mapping(intent):
            # Kiểm tra mapping trước
            if intent in intent_mapping:
                return intent_mapping[intent]
            return intent
        
        augmentation_templates = {
            # ===== CALL COMMANDS =====
            'call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại',
                'Gọi {person} ngay bây giờ',
                'Thực hiện cuộc gọi cho {person}',
                'Gọi cho {person} lúc {time}',
                'Gọi điện thoại cho {person}',
                'Gọi {person} khẩn cấp'
            ],
            'make-video-call': [
                'Gọi video cho {person}',
                'Thực hiện cuộc gọi video đến {person}',
                'Gọi FaceTime với {person}',
                'Gọi video {person} ngay bây giờ',
                'Thực hiện cuộc gọi video cho {person}',
                'Gọi video cho {person} lúc {time}',
                'Gọi video {person} khẩn cấp'
            ],
            
            # ===== MESSAGE COMMANDS =====
            'send-mess': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}',
                'Viết tin nhắn cho {person} về {message}',
                'Gửi cho {person} tin nhắn {message}',
                'Nhắn tin cho {person} là {message}',
                'Gửi tin nhắn cho {person} rằng {message}',
                'Soạn tin cho {person} nội dung {message}'
            ],
            
            # ===== SEARCH COMMANDS =====
            'search-content': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng'
            ],
            'search-internet': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng'
            ],
            
            # ===== MEDIA COMMANDS =====
            'play-content': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để xem'
            ],
            'play-media': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để xem'
            ],
            'play-audio': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Nghe {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để nghe'
            ],
            
            # ===== REMINDER COMMANDS =====
            'set-reminder': [
                'Đặt nhắc nhở {reminder}',
                'Tạo lời nhắc {reminder}',
                'Đặt lịch nhắc {reminder}',
                'Tạo nhắc nhở cho {reminder}',
                'Đặt báo thức cho {reminder}',
                'Nhắc nhở {reminder}',
                'Đặt nhắc nhở {reminder} lúc {time}',
                'Tạo lời nhắc {reminder} cho tôi'
            ],
            'set-alarm': [
                'Đặt báo thức {time}',
                'Tạo báo thức {time}',
                'Đặt chuông báo thức {time}',
                'Báo thức {time}',
                'Đặt báo thức cho {time}',
                'Tạo báo thức cho {time}',
                'Đặt chuông {time}',
                'Báo thức lúc {time}'
            ],
            
            # ===== CHECK COMMANDS =====
            'check-weather': [
                'Kiểm tra thời tiết hôm nay',
                'Xem dự báo thời tiết',
                'Thời tiết như thế nào',
                'Nhiệt độ hôm nay bao nhiêu',
                'Kiểm tra thời tiết',
                'Xem thời tiết hôm nay',
                'Dự báo thời tiết',
                'Thời tiết hôm nay ra sao'
            ],
            'check-messages': [
                'Kiểm tra tin nhắn từ {person}',
                'Xem tin nhắn mới',
                'Đọc tin nhắn chưa đọc',
                'Kiểm tra hộp thư',
                'Xem tin nhắn',
                'Kiểm tra tin nhắn',
                'Đọc tin nhắn',
                'Xem tin nhắn mới nhất'
            ],
            'check-device-status': [
                'Kiểm tra trạng thái thiết bị',
                'Xem trạng thái điện thoại',
                'Kiểm tra pin điện thoại',
                'Xem dung lượng bộ nhớ',
                'Kiểm tra kết nối mạng',
                'Xem trạng thái thiết bị',
                'Kiểm tra thiết bị',
                'Xem thông tin thiết bị'
            ],
            'check-health-status': [
                'Kiểm tra sức khỏe',
                'Xem tình trạng sức khỏe',
                'Kiểm tra nhịp tim',
                'Xem chỉ số sức khỏe',
                'Kiểm tra huyết áp',
                'Xem tình trạng sức khỏe',
                'Kiểm tra sức khỏe hôm nay',
                'Xem báo cáo sức khỏe'
            ]
        }
        
        # ===== DATA TEMPLATES =====
        persons = [
            'cháu Vương', 'chị Hương', 'anh Nam', 'bà nội', 'ông nội', 'mẹ', 'bố', 
            'em gái', 'anh trai', 'con trai', 'con gái', 'cháu trai', 'cháu gái',
            'bà ngoại', 'ông ngoại', 'chú', 'bác', 'cô', 'dì', 'dượng', 'mợ',
            'anh rể', 'chị dâu', 'em rể', 'em dâu', 'cháu nội', 'cháu ngoại'
        ]
        
        messages = [
            'chiều này đón bà tại công viên Thống nhất lúc 16h chiều',
            'sáng mai có hẹn bác sĩ',
            'tối nay về muộn',
            'đã nhận được tin nhắn',
            'nhớ uống thuốc đúng giờ',
            'có việc gấp cần liên lạc',
            'tối nay sẽ về sớm',
            'sáng mai đi chợ',
            'chiều nay có hẹn bạn',
            'tối nay ăn cơm ở nhà',
            'sáng mai đi khám bệnh',
            'chiều nay đón cháu ở trường',
            'tối nay xem phim cùng nhau',
            'sáng mai đi chùa',
            'chiều nay đi dạo công viên'
        ]
        
        queries = [
            'cách nấu phở', 'thời tiết hôm nay', 'tin tức mới nhất', 'công thức làm bánh',
            'địa chỉ bệnh viện', 'cách sử dụng điện thoại', 'thông tin về bệnh tiểu đường',
            'địa điểm du lịch gần đây', 'cách nấu canh chua', 'thời tiết ngày mai',
            'tin tức thể thao', 'công thức làm bánh mì', 'địa chỉ nhà thuốc',
            'cách sử dụng ứng dụng', 'thông tin về bệnh tim', 'địa điểm mua sắm gần đây'
        ]
        
        contents = [
            'nhạc trữ tình', 'phim hài', 'video hướng dẫn', 'bài hát mới', 'tin tức thời sự',
            'nhạc vàng', 'phim hành động', 'bài hát cũ', 'nhạc bolero', 'phim tình cảm',
            'video ca nhạc', 'tin tức thể thao', 'nhạc dân ca', 'phim cổ trang',
            'bài hát thiếu nhi', 'video hài kịch'
        ]
        
        reminders = [
            'uống thuốc lúc 8h sáng', 'họp gia đình tối nay', 'đi khám bệnh ngày mai',
            'gọi điện cho con', 'đi chợ sáng mai', 'hẹn bác sĩ tuần sau',
            'uống thuốc lúc 2h chiều', 'đi chùa sáng mai', 'họp lớp tối nay',
            'đi khám răng tuần sau', 'gọi điện cho cháu', 'đi dạo công viên chiều nay',
            'uống thuốc lúc 9h tối', 'đi chợ chiều nay', 'hẹn bạn cũ tối mai'
        ]
        
        times = [
            '8h sáng', '9h sáng', '10h sáng', '2h chiều', '3h chiều', '4h chiều',
            '7h tối', '8h tối', '9h tối', 'sáng mai', 'chiều nay', 'tối nay',
            'ngày mai', 'tuần sau', 'tháng sau', 'bây giờ', 'ngay bây giờ'
        ]
        
        # Intent mapping để tránh lặp lại
        intent_mapping = {
            'make-call': 'call',
            'send-message': 'send-mess',
            'make-video-call': 'make-video-call'
        }
        
        # Function để xử lý intent mapping
        def process_intent_mapping(intent):
            # Kiểm tra mapping trước
            if intent in intent_mapping:
                return intent_mapping[intent]
            return intent
        
        for _ in range(num_samples):
            sample = random.choice(data)
            command = sample['command']
            original_text = sample['input']
            
            # Sử dụng mapping để xử lý intent
            mapped_command = process_intent_mapping(command)
            
            if mapped_command in augmentation_templates:
                template = random.choice(augmentation_templates[mapped_command])
                
                # Replace placeholders with random data
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
                if '{time}' in template:
                    template = template.replace('{time}', random.choice(times))
                
                augmented_sample = {
                    'input': template,
                    'command': mapped_command  # Sử dụng mapped command
                }
                augmented.append(augmented_sample)
            else:
                # Fallback: use synonym replacement
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
        """Tạo samples mới dựa trên patterns - Cải thiện với nhiều patterns hơn"""
        new_samples = []
        
        # ===== NEW SAMPLE PATTERNS BY COMMAND =====
        patterns = {
            # ===== CALL PATTERNS =====
            'call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại',
                'Gọi {person} ngay bây giờ',
                'Gọi cho {person} lúc {time}',
                'Gọi điện thoại cho {person}',
                'Gọi {person} khẩn cấp'
            ],
            'make-call': [
                'Gọi điện cho {person}',
                'Thực hiện cuộc gọi đến {person}',
                'Liên lạc với {person} qua điện thoại',
                'Gọi {person} ngay bây giờ',
                'Gọi cho {person} lúc {time}',
                'Gọi điện thoại cho {person}',
                'Gọi {person} khẩn cấp'
            ],
            'make-video-call': [
                'Gọi video cho {person}',
                'Thực hiện cuộc gọi video đến {person}',
                'Gọi FaceTime với {person}',
                'Gọi video {person} ngay bây giờ',
                'Gọi video cho {person} lúc {time}',
                'Gọi video {person} khẩn cấp'
            ],
            
            # ===== MESSAGE PATTERNS =====
            'send-mess': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}',
                'Viết tin nhắn cho {person} về {message}',
                'Gửi cho {person} tin nhắn {message}',
                'Nhắn tin cho {person} là {message}',
                'Gửi tin nhắn cho {person} rằng {message}',
                'Soạn tin cho {person} nội dung {message}'
            ],
            'send-message': [
                'Nhắn tin cho {person} rằng {message}',
                'Gửi tin nhắn cho {person} nội dung {message}',
                'Soạn tin nhắn gửi {person} với nội dung {message}',
                'Viết tin nhắn cho {person} về {message}',
                'Gửi cho {person} tin nhắn {message}',
                'Nhắn tin cho {person} là {message}',
                'Gửi tin nhắn cho {person} rằng {message}',
                'Soạn tin cho {person} nội dung {message}'
            ],
            
            # ===== SEARCH PATTERNS =====
            'search-content': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng'
            ],
            'search-internet': [
                'Tìm kiếm {query}',
                'Tìm thông tin về {query}',
                'Tra cứu {query}',
                'Tìm hiểu về {query}',
                'Tìm kiếm thông tin {query}',
                'Tìm {query} trên internet',
                'Tra cứu thông tin {query}',
                'Tìm kiếm {query} trên mạng'
            ],
            
            # ===== MEDIA PATTERNS =====
            'play-content': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để xem'
            ],
            'play-media': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Xem {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để xem'
            ],
            'play-audio': [
                'Phát {content}',
                'Bật {content}',
                'Mở {content}',
                'Chạy {content}',
                'Nghe {content}',
                'Phát {content} ngay bây giờ',
                'Bật {content} cho tôi',
                'Mở {content} để nghe'
            ],
            
            # ===== REMINDER PATTERNS =====
            'set-reminder': [
                'Đặt nhắc nhở {reminder}',
                'Tạo lời nhắc {reminder}',
                'Đặt lịch nhắc {reminder}',
                'Tạo nhắc nhở cho {reminder}',
                'Đặt báo thức cho {reminder}',
                'Nhắc nhở {reminder}',
                'Đặt nhắc nhở {reminder} lúc {time}',
                'Tạo lời nhắc {reminder} cho tôi'
            ],
            'set-alarm': [
                'Đặt báo thức {time}',
                'Tạo báo thức {time}',
                'Đặt chuông báo thức {time}',
                'Báo thức {time}',
                'Đặt báo thức cho {time}',
                'Tạo báo thức cho {time}',
                'Đặt chuông {time}',
                'Báo thức lúc {time}'
            ],
            
            # ===== CHECK PATTERNS =====
            'check-weather': [
                'Kiểm tra thời tiết hôm nay',
                'Xem dự báo thời tiết',
                'Thời tiết như thế nào',
                'Nhiệt độ hôm nay bao nhiêu',
                'Kiểm tra thời tiết',
                'Xem thời tiết hôm nay',
                'Dự báo thời tiết',
                'Thời tiết hôm nay ra sao'
            ],
            'check-messages': [
                'Kiểm tra tin nhắn từ {person}',
                'Xem tin nhắn mới',
                'Đọc tin nhắn chưa đọc',
                'Kiểm tra hộp thư',
                'Xem tin nhắn',
                'Kiểm tra tin nhắn',
                'Đọc tin nhắn',
                'Xem tin nhắn mới nhất'
            ],
            'check-device-status': [
                'Kiểm tra trạng thái thiết bị',
                'Xem trạng thái điện thoại',
                'Kiểm tra pin điện thoại',
                'Xem dung lượng bộ nhớ',
                'Kiểm tra kết nối mạng',
                'Xem trạng thái thiết bị',
                'Kiểm tra thiết bị',
                'Xem thông tin thiết bị'
            ],
            'check-health-status': [
                'Kiểm tra sức khỏe',
                'Xem tình trạng sức khỏe',
                'Kiểm tra nhịp tim',
                'Xem chỉ số sức khỏe',
                'Kiểm tra huyết áp',
                'Xem tình trạng sức khỏe',
                'Kiểm tra sức khỏe hôm nay',
                'Xem báo cáo sức khỏe'
            ]
        }
        
        # ===== DATA TEMPLATES FOR NEW SAMPLES =====
        persons = [
            'cháu Vương', 'chị Hương', 'anh Nam', 'bà nội', 'ông nội', 'mẹ', 'bố', 
            'em gái', 'anh trai', 'con trai', 'con gái', 'cháu trai', 'cháu gái',
            'bà ngoại', 'ông ngoại', 'chú', 'bác', 'cô', 'dì', 'dượng', 'mợ',
            'anh rể', 'chị dâu', 'em rể', 'em dâu', 'cháu nội', 'cháu ngoại'
        ]
        
        messages = [
            'chiều này đón bà tại công viên Thống nhất lúc 16h chiều',
            'sáng mai có hẹn bác sĩ',
            'tối nay về muộn',
            'đã nhận được tin nhắn',
            'nhớ uống thuốc đúng giờ',
            'có việc gấp cần liên lạc',
            'tối nay sẽ về sớm',
            'sáng mai đi chợ',
            'chiều nay có hẹn bạn',
            'tối nay ăn cơm ở nhà',
            'sáng mai đi khám bệnh',
            'chiều nay đón cháu ở trường',
            'tối nay xem phim cùng nhau',
            'sáng mai đi chùa',
            'chiều nay đi dạo công viên'
        ]
        
        queries = [
            'cách nấu phở', 'thời tiết hôm nay', 'tin tức mới nhất', 'công thức làm bánh',
            'địa chỉ bệnh viện', 'cách sử dụng điện thoại', 'thông tin về bệnh tiểu đường',
            'địa điểm du lịch gần đây', 'cách nấu canh chua', 'thời tiết ngày mai',
            'tin tức thể thao', 'công thức làm bánh mì', 'địa chỉ nhà thuốc',
            'cách sử dụng ứng dụng', 'thông tin về bệnh tim', 'địa điểm mua sắm gần đây'
        ]
        
        contents = [
            'nhạc trữ tình', 'phim hài', 'video hướng dẫn', 'bài hát mới', 'tin tức thời sự',
            'nhạc vàng', 'phim hành động', 'bài hát cũ', 'nhạc bolero', 'phim tình cảm',
            'video ca nhạc', 'tin tức thể thao', 'nhạc dân ca', 'phim cổ trang',
            'bài hát thiếu nhi', 'video hài kịch'
        ]
        
        reminders = [
            'uống thuốc lúc 8h sáng', 'họp gia đình tối nay', 'đi khám bệnh ngày mai',
            'gọi điện cho con', 'đi chợ sáng mai', 'hẹn bác sĩ tuần sau',
            'uống thuốc lúc 2h chiều', 'đi chùa sáng mai', 'họp lớp tối nay',
            'đi khám răng tuần sau', 'gọi điện cho cháu', 'đi dạo công viên chiều nay',
            'uống thuốc lúc 9h tối', 'đi chợ chiều nay', 'hẹn bạn cũ tối mai'
        ]
        
        times = [
            '8h sáng', '9h sáng', '10h sáng', '2h chiều', '3h chiều', '4h chiều',
            '7h tối', '8h tối', '9h tối', 'sáng mai', 'chiều nay', 'tối nay',
            'ngày mai', 'tuần sau', 'tháng sau', 'bây giờ', 'ngay bây giờ'
        ]
        
        commands = list(patterns.keys())
        
        # Intent mapping để tránh lặp lại
        intent_mapping = {
            'make-call': 'call',
            'send-message': 'send-mess',
            'make-video-call': 'make-video-call'
        }
        
        # Function để xử lý intent mapping
        def process_intent_mapping(intent):
            # Kiểm tra mapping trước
            if intent in intent_mapping:
                return intent_mapping[intent]
            return intent
        
        for _ in range(num_samples):
            command = random.choice(commands)
            
            # Sử dụng mapping để xử lý intent
            mapped_command = process_intent_mapping(command)
            
            if mapped_command in patterns:
                pattern = random.choice(patterns[mapped_command])
                
                # Replace placeholders with random data
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
                if '{time}' in pattern:
                    pattern = pattern.replace('{time}', random.choice(times))
                
                new_sample = {
                    'input': pattern,
                    'command': mapped_command  # Sử dụng mapped command
                }
                new_samples.append(new_sample)
        
        return new_samples
    
    def save_expanded_data(self, data: List[Dict]):
        """Lưu dataset đã mở rộng với error handling"""
        try:
            # Tạo thư mục nếu chưa tồn tại
            self.expanded_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.expanded_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved expanded dataset to {self.expanded_file}")
            
            # Thống kê command distribution
            command_counts = {}
            for item in data:
                command = item['command']
                command_counts[command] = command_counts.get(command, 0) + 1
            
            print(f"📈 Final command distribution:")
            for command, count in sorted(command_counts.items()):
                percentage = (count / len(data)) * 100
                print(f"   {command}: {count} samples ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            print("🔧 Please check file permissions and try again")

def main():
    """Main function với error handling"""
    print("🚀 Dataset Expansion for PhoBERT-Large")
    print("=" * 50)
    
    try:
        expander = DatasetExpander()
        
        # Kiểm tra file gốc có tồn tại không
        if not expander.original_file.exists():
            print(f"❌ Original file not found: {expander.original_file}")
            print("🔧 Please check the file path and try again")
            return
        
        expanded_data = expander.expand_dataset(target_size=1000)
        
        if not expanded_data:
            print("❌ No data to save")
            return
        
        expander.save_expanded_data(expanded_data)
        
        print("\n🎉 Dataset expansion completed!")
        print("📋 Next steps:")
        print("   1. Use the expanded dataset for training")
        print("   2. Run: python run_training.py")
        print("   3. The large model will have more data to learn from")
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        print("🔧 Please check the code and try again")

if __name__ == "__main__":
    main()
