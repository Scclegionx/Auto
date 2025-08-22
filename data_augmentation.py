#!/usr/bin/env python3
"""
Script tăng cường dữ liệu cho dataset
Tạo thêm samples để cải thiện training
"""

import json
import random
import re
from typing import List, Dict
from copy import deepcopy

class DataAugmentor:
    """Data augmentation cho dataset tiếng Việt"""
    
    def __init__(self):
        # Từ đồng nghĩa cho các intent chính
        self.synonyms = {
            "call": ["gọi", "điện thoại", "alo", "kết nối", "liên lạc", "gọi điện"],
            "send-mess": ["gửi tin nhắn", "nhắn tin", "text", "sms", "thông báo", "gửi tin"],
            "set-alarm": ["đặt báo thức", "nhắc nhở", "hẹn giờ", "đánh thức", "chuông báo", "đặt nhắc"],
            "check-weather": ["thời tiết", "nhiệt độ", "mưa", "nắng", "khí hậu", "dự báo thời tiết"],
            "play-media": ["phát nhạc", "bật nhạc", "nghe nhạc", "video", "xem video", "phát video"],
            "read-news": ["đọc tin tức", "tin tức", "báo", "thời sự", "đọc báo", "tin mới"],
            "check-health-status": ["sức khỏe", "tình trạng sức khỏe", "kiểm tra sức khỏe", "bệnh tình"],
            "set-reminder": ["nhắc nhở thuốc", "đặt lịch", "lịch trình", "nhắc việc", "đặt nhắc nhở"],
            "general-conversation": ["trò chuyện", "nói chuyện", "tâm sự", "giao tiếp", "hội thoại"]
        }
        
        # Các từ lịch sự
        self.polite_words = ["xin", "vui lòng", "làm ơn", "có thể", "nhờ", "giúp"]
        
        # Các từ cảm xúc
        self.emotion_words = ["tôi muốn", "tôi cần", "tôi mong", "tôi hy vọng", "tôi thích"]
        
        # Các từ thời gian
        self.time_words = ["bây giờ", "ngay bây giờ", "lúc này", "hiện tại", "ngay lập tức"]
        
        # Các từ địa điểm
        self.location_words = ["ở đây", "tại đây", "nơi này", "chỗ này"]
    
    def augment_with_synonyms(self, text: str, intent: str) -> List[str]:
        """Tạo variants bằng cách thay thế từ đồng nghĩa"""
        augmented_texts = []
        
        if intent in self.synonyms:
            synonyms = self.synonyms[intent]
            for synonym in synonyms:
                # Tìm từ khóa gốc trong text
                for original_word in self.synonyms[intent][:3]:  # Chỉ dùng 3 từ đầu
                    if original_word in text.lower():
                        new_text = re.sub(
                            rf'\b{re.escape(original_word)}\b', 
                            synonym, 
                            text, 
                            flags=re.IGNORECASE
                        )
                        if new_text != text:
                            augmented_texts.append(new_text)
                        break
        
        return augmented_texts
    
    def augment_with_polite_words(self, text: str) -> List[str]:
        """Thêm từ lịch sự"""
        augmented_texts = []
        
        for polite_word in self.polite_words:
            # Thêm vào đầu câu
            new_text = f"{polite_word} {text}"
            augmented_texts.append(new_text)
            
            # Thêm vào giữa câu (nếu có thể)
            if "tôi" in text.lower():
                new_text = text.replace("tôi", f"{polite_word} tôi")
                if new_text != text:
                    augmented_texts.append(new_text)
        
        return augmented_texts
    
    def augment_with_emotion_words(self, text: str) -> List[str]:
        """Thêm từ cảm xúc"""
        augmented_texts = []
        
        for emotion_word in self.emotion_words:
            # Thêm vào đầu câu
            new_text = f"{emotion_word} {text}"
            augmented_texts.append(new_text)
        
        return augmented_texts
    
    def augment_with_time_words(self, text: str) -> List[str]:
        """Thêm từ thời gian"""
        augmented_texts = []
        
        for time_word in self.time_words:
            # Thêm vào cuối câu
            new_text = f"{text} {time_word}"
            augmented_texts.append(new_text)
            
            # Thêm vào đầu câu
            new_text = f"{time_word} {text}"
            augmented_texts.append(new_text)
        
        return augmented_texts
    
    def augment_with_location_words(self, text: str) -> List[str]:
        """Thêm từ địa điểm"""
        augmented_texts = []
        
        for location_word in self.location_words:
            # Thêm vào cuối câu
            new_text = f"{text} {location_word}"
            augmented_texts.append(new_text)
        
        return augmented_texts
    
    def augment_sentence_structure(self, text: str) -> List[str]:
        """Thay đổi cấu trúc câu"""
        augmented_texts = []
        
        # Thêm "ạ" vào cuối (lịch sự)
        if not text.endswith("ạ"):
            augmented_texts.append(f"{text} ạ")
        
        # Thêm "nhé" vào cuối
        if not text.endswith("nhé"):
            augmented_texts.append(f"{text} nhé")
        
        # Thêm "đấy" vào cuối
        if not text.endswith("đấy"):
            augmented_texts.append(f"{text} đấy")
        
        # Thay đổi "tôi" thành "em" hoặc "cháu"
        if "tôi" in text.lower():
            augmented_texts.append(text.replace("tôi", "em"))
            augmented_texts.append(text.replace("tôi", "cháu"))
        
        return augmented_texts
    
    def augment_dataset(self, dataset: List[Dict], factor: float = 2.0) -> List[Dict]:
        """Tăng cường toàn bộ dataset"""
        print(f"Bắt đầu augmentation với factor: {factor}")
        
        augmented_dataset = []
        total_original = len(dataset)
        target_size = int(total_original * factor)
        
        # Thêm dữ liệu gốc
        augmented_dataset.extend(dataset)
        
        # Tạo thêm dữ liệu
        while len(augmented_dataset) < target_size:
            # Chọn ngẫu nhiên một sample
            sample = random.choice(dataset)
            text = sample["input"]
            intent = sample["command"]
            
            # Tạo variants
            new_texts = []
            
            # Augmentation với synonyms
            new_texts.extend(self.augment_with_synonyms(text, intent))
            
            # Augmentation với polite words
            new_texts.extend(self.augment_with_polite_words(text))
            
            # Augmentation với emotion words
            new_texts.extend(self.augment_with_emotion_words(text))
            
            # Augmentation với time words
            new_texts.extend(self.augment_with_time_words(text))
            
            # Augmentation với location words
            new_texts.extend(self.augment_with_location_words(text))
            
            # Augmentation với sentence structure
            new_texts.extend(self.augment_sentence_structure(text))
            
            # Tạo samples mới
            for new_text in new_texts:
                if len(augmented_dataset) >= target_size:
                    break
                
                # Tạo sample mới
                new_sample = deepcopy(sample)
                new_sample["input"] = new_text
                new_sample["augmented"] = True  # Đánh dấu là augmented
                
                augmented_dataset.append(new_sample)
        
        print(f"Augmentation hoàn thành:")
        print(f"  - Dữ liệu gốc: {total_original} samples")
        print(f"  - Dữ liệu sau augmentation: {len(augmented_dataset)} samples")
        print(f"  - Tăng trưởng: {len(augmented_dataset) / total_original:.1f}x")
        
        return augmented_dataset
    
    def balance_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Cân bằng dataset"""
        print("Cân bằng dataset...")
        
        # Đếm số lượng mỗi intent
        intent_counts = {}
        for item in dataset:
            intent = item["command"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("Phân bố intent trước khi cân bằng:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  {intent}: {count}")
        
        # Tìm số lượng mẫu mục tiêu (trung bình)
        target_count = int(sum(intent_counts.values()) / len(intent_counts))
        
        balanced_dataset = []
        
        for intent, count in intent_counts.items():
            intent_samples = [item for item in dataset if item["command"] == intent]
            
            if count < target_count:
                # Oversampling: lặp lại samples
                while len(intent_samples) < target_count:
                    intent_samples.extend(random.sample(intent_samples, min(len(intent_samples), target_count - len(intent_samples))))
            elif count > target_count:
                # Undersampling: lấy ngẫu nhiên
                intent_samples = random.sample(intent_samples, target_count)
            
            balanced_dataset.extend(intent_samples)
        
        # Shuffle dataset
        random.shuffle(balanced_dataset)
        
        # Đếm lại
        new_intent_counts = {}
        for item in balanced_dataset:
            intent = item["command"]
            new_intent_counts[intent] = new_intent_counts.get(intent, 0) + 1
        
        print("Phân bố intent sau khi cân bằng:")
        for intent, count in sorted(new_intent_counts.items()):
            print(f"  {intent}: {count}")
        
        return balanced_dataset

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data augmentation cho PhoBERT_SAM")
    parser.add_argument("--input", type=str, default="elderly_command_dataset_reduced.json",
                       help="Input dataset file")
    parser.add_argument("--output", type=str, default="elderly_command_dataset_augmented.json",
                       help="Output dataset file")
    parser.add_argument("--factor", type=float, default=3.0,
                       help="Augmentation factor (số lần tăng)")
    parser.add_argument("--balance", action="store_true",
                       help="Cân bằng dataset sau augmentation")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Augmentation
    augmentor = DataAugmentor()
    augmented_dataset = augmentor.augment_dataset(dataset, args.factor)
    
    # Balance nếu cần
    if args.balance:
        augmented_dataset = augmentor.balance_dataset(augmented_dataset)
    
    # Save dataset
    print(f"Saving augmented dataset: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=2)
    
    print("✅ Augmentation hoàn thành!")

if __name__ == "__main__":
    main()
