"""
Entity Extractor Module cho hệ thống gọi điện/nhắn tin
Tập trung vào RECEIVER, TIME, MESSAGE, PLATFORM extraction
"""

import re
from typing import Dict, List, Optional, Tuple

class EntityExtractor:
    """Entity extractor chuyên biệt cho hệ thống gọi điện/nhắn tin"""
    
    def __init__(self):
        self.receiver_patterns = self._build_receiver_patterns()
        self.time_patterns = self._build_time_patterns()
        self.message_patterns = self._build_message_patterns()
        self.platform_patterns = self._build_platform_patterns()
        
    def _build_receiver_patterns(self) -> List[Tuple[str, str]]:
        """Xây dựng patterns cho RECEIVER extraction - Tối ưu cho người già"""
        return [
            # Pattern 1: Gọi trực tiếp (ưu tiên cao)
            (r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"alo\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+điện\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+thoại\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            
            # Pattern 2: Nhắn tin (ưu tiên cao)
            (r"nhắn\s+(?:tin|tin nhắn)?\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            (r"gửi\s+(?:tin|tin nhắn)?\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            (r"soạn\s+tin\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            
            # Pattern 3: Với platform (cải thiện để extract chính xác)
            (r"nhắn\s+tin\s+qua\s+[\w\s]+\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))", "nhắn"),
            (r"gửi\s+tin\s+qua\s+[\w\s]+\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))", "nhắn"),
            
            # Pattern 4: Video call
            (r"gọi\s+video\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"facetime\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            (r"video\s+call\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "video"),
            
            # Pattern 5: Khẩn cấp
            (r"gọi\s+ngay\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+khẩn\s+cấp\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá))?(?:$|[\.,])", "gọi"),
            
            # Pattern 6: Nhiều người (tối ưu cho gia đình)
            (r"gọi\s+cho\s+(?:cả\s+nhà|tất\s+cả|mọi\s+người|con\s+cháu|gia\s+đình)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+(?:cả\s+nhà|tất\s+cả|mọi\s+người|con\s+cháu|gia\s+đình)", "nhắn"),
            
            # Pattern 7: Quan hệ phức tạp (tối ưu cho người già)
            (r"gọi\s+cho\s+([\w\s]+?)\s+(?:của|ở|tại)\s+[\w\s]+", "gọi"),
            (r"nhắn\s+tin\s+cho\s+([\w\s]+?)\s+(?:của|ở|tại)\s+[\w\s]+", "nhắn"),
            
            # Pattern 8: Quan hệ gia đình (thêm mới)
            (r"gọi\s+cho\s+(?:bố|mẹ|ông|bà|anh|chị|em|con|cháu|chú|bác|cô|dì|dượng|mợ)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+(?:bố|mẹ|ông|bà|anh|chị|em|con|cháu|chú|bác|cô|dì|dượng|mợ)", "nhắn"),
            
            # Pattern 9: Tên riêng (thêm mới)
            (r"gọi\s+cho\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "gọi"),
            (r"nhắn\s+tin\s+cho\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "nhắn"),
            
            # Pattern 10: Fallback patterns
            (r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"(?:cuộc gọi|gọi điện|gọi thoại)\s+(?:cho|tới|đến)\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá|ngay bây giờ|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
        ]
    
    def _build_time_patterns(self) -> List[str]:
        """Xây dựng patterns cho TIME extraction - Tối ưu cho người già"""
        return [
            # Thời gian cụ thể (ưu tiên cao)
            r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(?:phút)?\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*rưỡi\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*giờ\s*rưỡi\s*(?:sáng|trưa|chiều|tối|đêm)?",
            r"(\d{1,2})\s*giờ\s*(?:kém|thiếu)\s*(\d{1,2})",
            
            # Thời gian tương đối (tối ưu cho người già)
            r"(sáng|trưa|chiều|tối|đêm)\s*(?:nay|mai|kia)?",
            r"(hôm\s+nay|ngày\s+mai|tuần\s+sau|tháng\s+sau)",
            r"(sau\s+(?:khi\s+)?ăn|sau\s+bữa\s+(?:sáng|trưa|tối))",
            r"(trước\s+(?:khi\s+)?ăn|trước\s+bữa\s+(?:sáng|trưa|tối))",
            
            # Thời gian khẩn cấp
            r"(ngay|ngay\s+bây\s+giờ|bây\s+giờ|lập\s+tức)",
            r"(khi\s+nào|khi\s+đó|lúc\s+đó)",
            
            # Thời gian định kỳ (thêm mới)
            r"(hàng\s+ngày|hàng\s+tuần|hàng\s+tháng)",
            r"(thứ\s+\d+\s+hàng\s+tuần)",
            r"(ngày\s+\d+\s+hàng\s+tháng)",
            
            # Thời gian theo bữa ăn (tối ưu cho người già)
            r"(sau\s+bữa\s+sáng|sau\s+bữa\s+trưa|sau\s+bữa\s+tối)",
            r"(trước\s+bữa\s+sáng|trước\s+bữa\s+trưa|trước\s+bữa\s+tối)",
            
            # Thời gian theo hoạt động (thêm mới)
            r"(sau\s+khi\s+ngủ|trước\s+khi\s+ngủ)",
            r"(sau\s+khi\s+đi\s+chợ|trước\s+khi\s+đi\s+chợ)",
            r"(sau\s+khi\s+đi\s+bệnh\s+viện|trước\s+khi\s+đi\s+bệnh\s+viện)",
        ]
    
    def _build_message_patterns(self) -> List[str]:
        """Xây dựng patterns cho MESSAGE extraction - Tối ưu cho người già"""
        return [
            # Pattern 1: Rằng là (ưu tiên cao)
            r"rằng\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 2: Là
            r"là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 3: Nói
            r"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nói\s+rõ\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 4: Nhắn/Gửi
            r"nhắn\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 5: Với nội dung trong ngoặc
            r"[\"'](.+?)[\"']",
            
            # Pattern 6: Sau từ khóa
            r"(?:nội\s+dung|tin\s+nhắn)\s+(?:là|rằng)?\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 7: Tin nhắn dài (thêm mới)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 8: Tin nhắn với thời gian (thêm mới)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)\s+lúc\s+[\w\s]+",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)\s+lúc\s+[\w\s]+",
        ]
    
    def _build_platform_patterns(self) -> List[str]:
        """Xây dựng patterns cho PLATFORM extraction - Tối ưu cho người già"""
        return [
            # Pattern 1: Qua/Bằng/Trên (ưu tiên cao)
            r"qua\s+(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            r"bằng\s+(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            r"trên\s+(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            r"sử\s+dụng\s+(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            r"dùng\s+(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            
            # Pattern 2: Trực tiếp (thêm mới)
            r"(zalo|messenger|youtube|facebook|sms|tin\s+nhắn)",
            
            # Pattern 3: Tên gọi khác (thêm mới)
            r"(zalo|messenger|youtube|facebook|sms|tin\s+nhắn|facebook|youtube)",
            
            # Pattern 4: Tên gọi thân thiện (thêm mới)
            r"(zalo|messenger|youtube|facebook|sms|tin\s+nhắn|facebook|youtube|google)",
        ]
    
    def extract_receiver(self, text: str) -> Optional[Dict[str, str]]:
        """Extract RECEIVER entity với độ chính xác cao"""
        text_lower = text.lower()
        
        for pattern, action_type in self.receiver_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                receiver = match.group(1).strip()
                
                # Làm sạch receiver
                receiver = self._clean_receiver(receiver)
                
                if receiver and len(receiver) > 1:
                    return {
                        "RECEIVER": receiver,
                        "ACTION_TYPE": action_type
                    }
        
        return None
    
    def extract_time(self, text: str) -> Optional[str]:
        """Extract TIME entity"""
        text_lower = text.lower()
        
        for pattern in self.time_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if match.groups():
                    time_value = " ".join([g for g in match.groups() if g])
                    if time_value:
                        return time_value.strip()
                else:
                    return match.group(0).strip()
        
        return None
    
    def extract_message(self, text: str, receiver: str = None) -> Optional[str]:
        """Extract MESSAGE entity"""
        text_lower = text.lower()
        
        for pattern in self.message_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                message = match.group(1).strip()
                
                # Làm sạch message
                message = self._clean_message(message)
                
                if message and len(message) > 3:
                    return message
        
        return None
    
    def extract_platform(self, text: str) -> str:
        """Extract PLATFORM entity với logic thông minh"""
        text_lower = text.lower()
        
        for pattern in self.platform_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                platform = match.group(1).lower()
                return platform
        
        # Logic thông minh dựa trên context
        if any(word in text_lower for word in ["gọi", "alo", "gọi điện", "gọi thoại"]):
            return "phone"
        elif any(word in text_lower for word in ["nhắn", "gửi", "tin nhắn", "sms"]):
            return "sms"
        elif any(word in text_lower for word in ["tìm", "tìm kiếm", "search", "youtube", "facebook"]):
            return "youtube" if "youtube" in text_lower else "facebook" if "facebook" in text_lower else "google"
        else:
            return "sms"
    
    def _clean_receiver(self, receiver: str) -> str:
        """Làm sạch receiver entity - Tối ưu cho người già"""
        unwanted_words = [
            "rằng", "là", "nói", "nhắn", "gửi", "lúc", "vào", "nhé", "nha", "ạ", "nhá", 
            "ngay", "bây giờ", "qua", "messenger", "zalo", "facebook", "telegram", 
            "instagram", "tiktok", "sms", "tin", "nhắn", "gửi", "cho", "tới", "đến",
            "chiều", "sáng", "trưa", "tối", "đêm", "nay", "mai", "hôm", "ngày", "tuần", "tháng",
            "của", "ở", "tại", "với", "và", "hoặc", "hay", "nếu", "khi", "sau", "trước",
            "điện", "khẩn cấp", "video", "con", "sẽ", "đã", "có", "vì", "bị", "đau", "bụng"
        ]
        
        words = receiver.split()
        cleaned_words = []
        
        for word in words:
            if word.lower() not in unwanted_words:
                cleaned_words.append(word)
        
        # Giới hạn 2-3 từ để tránh extract quá dài
        if len(cleaned_words) > 3:
            cleaned_words = cleaned_words[:3]
        
        return " ".join(cleaned_words).strip()
    
    def _clean_message(self, message: str) -> str:
        """Làm sạch message entity"""
        unwanted_prefixes = ["rằng", "là", "nói", "nhắn", "gửi"]
        
        for prefix in unwanted_prefixes:
            if message.lower().startswith(prefix + " "):
                message = message[len(prefix):].strip()
        
        return message.strip()
    
    def extract_all_entities(self, text: str) -> Dict[str, str]:
        """Extract tất cả entities cho hệ thống gọi điện/nhắn tin"""
        entities = {}
        
        receiver_result = self.extract_receiver(text)
        if receiver_result:
            entities.update(receiver_result)
        
        time_result = self.extract_time(text)
        if time_result:
            entities["TIME"] = time_result
        
        message_result = self.extract_message(text, entities.get("RECEIVER"))
        if message_result:
            entities["MESSAGE"] = message_result
        
        platform_result = self.extract_platform(text)
        if platform_result:
            entities["PLATFORM"] = platform_result
        
        return entities

# Test function
def test_entity_extraction():
    """Test các trường hợp thực tế"""
    extractor = EntityExtractor()
    
    test_cases = [
        "gọi cho bố",
        "alo cho mẹ",
        "nhắn tin cho bà ngoại rằng tối con sẽ về",
        "gửi tin nhắn qua Zalo cho chị Hương",
        "gọi video cho con gái",
        "nhắn tin qua Messenger tới Bà Sam rằng chiều này sẽ qua nhà bà Hà ăn rằm lúc tám giờ tối",
        "gọi ngay cho bác sĩ",
        "nhắn tin cho cả nhà rằng tối nay ăn cơm",
        "gọi cho bà ngoại của con",
        "nếu bố gọi thì nhắn tin cho mẹ",
    ]
    
    print("🧪 TESTING ENTITY EXTRACTION")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{text}'")
        entities = extractor.extract_all_entities(text)
        
        if entities:
            for key, value in entities.items():
                print(f"   {key}: {value}")
        else:
            print("   ❌ Không extract được entities")

if __name__ == "__main__":
    test_entity_extraction()
