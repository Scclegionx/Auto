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
            # Pattern 1: Gọi trực tiếp (ưu tiên cao) - Cải thiện cho "Bố Dũng"
            (r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"alo\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+điện\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"gọi\s+thoại\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            
            # Pattern 1.1: Nói chuyện điện thoại (thêm mới cho trường hợp "Tôi muốn nói chuyện điện thoại với Bố Dũng")
            (r"nói\s+chuyện\s+điện\s+thoại\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"nói\s+chuyện\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"trò\s+chuyện\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            (r"liên\s+lạc\s+(?:với|cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:vì|lúc|vào|nhé|nha|ạ|nhá|ngay|bây giờ))?(?:$|[\.,])", "gọi"),
            
            # Pattern 2: Nhắn tin (ưu tiên cao) - Cải thiện boundary
            (r"nhắn\s+(?:tin|tin nhắn)?\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            (r"gửi\s+(?:tin|tin nhắn)?\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            (r"soạn\s+tin\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:\s+(?:rằng|là|nói|nhé|nha|ạ|nhá))?(?:$|[\.,])", "nhắn"),
            
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
            
            # Pattern 9: Nhắn tin cho [người] rằng [nội dung] (cải thiện)
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 10: Với nội dung là [nội dung] (thêm mới)
            r"với\s+nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
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
        """Extract RECEIVER entity với độ chính xác cao - Cải thiện boundary detection"""
        text_lower = text.lower()
        
        # Kiểm tra xem có phải nhắn tin với số điện thoại không
        if any(word in text_lower for word in ["nhắn tin", "gửi tin", "soạn tin"]) and \
           any(word in text_lower for word in ["số", "điện thoại", "qua", "gửi", "nhắn", "cho"]):
            # Extract số điện thoại từ chữ
            phone_number = self._extract_phone_number_from_text(text)
            if phone_number:
                return {
                    "RECEIVER": phone_number,
                    "ACTION_TYPE": "nhắn"
                }
        
        # Kiểm tra xem có phải gọi điện với số điện thoại không
        if any(word in text_lower for word in ["gọi điện", "gọi", "alo"]) and \
           any(word in text_lower for word in ["số", "điện thoại"]):
            # Nếu có số điện thoại nhưng không có tên người, trả về None
            # Để tránh extract sai thông tin
            return None
        
        for pattern, action_type in self.receiver_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                receiver = match.group(1).strip()
                
                # Cải thiện boundary detection
                receiver = self._improve_receiver_boundary(receiver, text_lower)
                
                # Làm sạch receiver
                receiver = self._clean_receiver(receiver)
                
                if receiver and len(receiver) > 1:
                    return {
                        "RECEIVER": receiver,
                        "ACTION_TYPE": action_type
                    }
        
        return None
    
    def _improve_receiver_boundary(self, receiver: str, full_text: str) -> str:
        """Cải thiện boundary detection cho receiver - Tối ưu cho "Bố Dũng" """
        words = receiver.split()
        if not words:
            return receiver
            
        # Tìm vị trí của receiver trong full text
        receiver_start = full_text.find(receiver.lower())
        if receiver_start == -1:
            return receiver
            
        # Tìm từ đầu tiên sau receiver trong full text
        after_receiver = full_text[receiver_start + len(receiver):].strip()
        if not after_receiver:
            return receiver
            
        # Tách từ đầu tiên sau receiver
        first_word_after = after_receiver.split()[0] if after_receiver.split() else ""
        
        # Mở rộng danh sách stop words để xử lý tốt hơn
        stop_words = ["là", "rằng", "nói", "sẽ", "đã", "có", "vì", "bị", "đau", "bụng", 
                      "đón", "ở", "tại", "với", "và", "hoặc", "hay", "nếu", "khi", "sau", "trước",
                      "tối", "nay", "chiều", "sáng", "trưa", "đêm", "mai", "hôm", "ngày",
                      "nhớ", "thương", "yêu", "quý", "mến", "kính", "trọng", "quý", "mến",
                      "điện", "thoại", "gọi", "nhắn", "tin", "nhắn", "gửi", "soạn", "viết"]
        
        if first_word_after.lower() in stop_words:
            # Tìm vị trí của từ stop trong receiver
            for i, word in enumerate(words):
                if word.lower() in stop_words:
                    return " ".join(words[:i])
        
        # Xử lý đặc biệt cho trường hợp "Bố Dũng" - giữ nguyên nếu là tên riêng
        if len(words) == 2 and words[0].lower() in ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu"]:
            return receiver
            
        return receiver
    
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
        
        # Kiểm tra pattern "với nội dung là"
        if "với nội dung là" in text_lower:
            start_pos = text_lower.find("với nội dung là")
            if start_pos != -1:
                message = text[start_pos + len("với nội dung là"):].strip()
                if message:
                    return message
        
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
            # Mặc định là phone nếu có số điện thoại
            if any(word in text_lower for word in ["số", "điện thoại"]):
                return "phone"
            return "sms"
    
    def _clean_receiver(self, receiver: str) -> str:
        """Làm sạch receiver entity - Tối ưu cho người già và "Bố Dũng" """
        # Danh sách từ cần loại bỏ (mở rộng)
        unwanted_words = [
            "rằng", "là", "nói", "nhắn", "gửi", "lúc", "vào", "nhé", "nha", "ạ", "nhá", 
            "ngay", "bây giờ", "qua", "messenger", "zalo", "facebook", "telegram", 
            "instagram", "tiktok", "sms", "tin", "nhắn", "gửi", "cho", "tới", "đến",
            "chiều", "sáng", "trưa", "tối", "đêm", "nay", "mai", "hôm", "ngày", "tuần", "tháng",
            "của", "ở", "tại", "với", "và", "hoặc", "hay", "nếu", "khi", "sau", "trước",
            "điện", "khẩn cấp", "video", "con", "sẽ", "đã", "có", "vì", "bị", "đau", "bụng",
            "sẽ", "đón", "bà", "ở", "bệnh", "viện", "tối", "nay", "chiều", "sáng", "trưa",
            "nhớ", "thương", "yêu", "quý", "mến", "kính", "trọng", "quý", "mến"
        ]
        
        words = receiver.split()
        cleaned_words = []
        
        # Logic cải thiện: dừng khi gặp từ chỉ thời gian hoặc động từ
        stop_words = ["là", "rằng", "nói", "sẽ", "đã", "có", "vì", "bị", "đau", "bụng", 
                      "đón", "ở", "tại", "với", "và", "hoặc", "hay", "nếu", "khi", "sau", "trước",
                      "nhớ", "thương", "yêu", "quý", "mến", "kính", "trọng", "quý", "mến"]
        
        for word in words:
            word_lower = word.lower()
            
            # Dừng khi gặp từ chỉ thời gian hoặc động từ
            if word_lower in stop_words:
                break
                
            # Chỉ thêm từ không có trong danh sách unwanted
            if word_lower not in unwanted_words:
                cleaned_words.append(word)
        
        # Xử lý đặc biệt cho trường hợp "Bố Dũng" - giữ nguyên nếu là tên riêng
        if len(cleaned_words) == 2 and cleaned_words[0].lower() in ["bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu"]:
            return " ".join(cleaned_words)
        
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
    
    def extract_phone_number(self, text: str) -> Optional[str]:
        """Extract phone number từ text - Hỗ trợ cả số và chữ"""
        # Pattern cho số điện thoại Việt Nam (dạng số)
        phone_patterns = [
            r"(\d{10,11})",  # 10-11 chữ số
            r"(\d{3,4}\s*\d{3,4}\s*\d{3,4})",  # Có khoảng trắng
            r"(\d{3,4}-\d{3,4}-\d{3,4})",  # Có dấu gạch ngang
            r"(\d{3,4}\.\d{3,4}\.\d{3,4})",  # Có dấu chấm
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(1)
                # Làm sạch số điện thoại
                phone = re.sub(r'[^\d]', '', phone)
                if len(phone) >= 10:
                    return phone
        
        # Xử lý số điện thoại bằng chữ
        phone_text = self._extract_phone_number_from_text(text)
        if phone_text:
            return phone_text
        
        return None
    
    def _extract_phone_number_from_text(self, text: str) -> Optional[str]:
        """Chuyển đổi số điện thoại từ chữ sang số"""
        # Mapping từ chữ sang số
        number_mapping = {
            "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
            "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9"
        }
        
        text_lower = text.lower()
        
        # Tìm pattern "số" + các từ số
        if "số" in text_lower:
            # Tìm vị trí của "số"
            start_pos = text_lower.find("số")
            if start_pos != -1:
                # Lấy phần sau "số"
                after_số = text_lower[start_pos + 2:].strip()
                words = after_số.split()
                
                # Tìm chuỗi số điện thoại liên tiếp (10-11 từ)
                for i in range(len(words) - 9):  # Cần ít nhất 10 từ
                    phone_digits = []
                    j = i
                    
                    # Lấy 10-11 từ liên tiếp
                    while j < len(words) and len(phone_digits) < 11:
                        if words[j] in number_mapping:
                            phone_digits.append(number_mapping[words[j]])
                            j += 1
                        else:
                            break
                    
                    # Kiểm tra có đủ 10-11 chữ số không
                    if 10 <= len(phone_digits) <= 11:
                        phone_number = ''.join(phone_digits)
                        # Kiểm tra số điện thoại Việt Nam hợp lệ
                        if phone_number.startswith(('03', '05', '07', '08', '09')):
                            return phone_number
        
        # Tìm pattern 10-11 từ liên tiếp (không có "số")
        words = text_lower.split()
        if 10 <= len(words) <= 11:
            phone_digits = []
            for word in words:
                if word in number_mapping:
                    phone_digits.append(number_mapping[word])
                else:
                    # Nếu có từ không phải số, dừng lại
                    break
            
            # Kiểm tra có đủ 10-11 chữ số không
            if len(phone_digits) >= 10:
                phone_number = ''.join(phone_digits)
                # Kiểm tra số điện thoại Việt Nam hợp lệ
                if phone_number.startswith(('03', '05', '07', '08', '09')):
                    return phone_number
        
        return None
    
    def extract_all_entities(self, text: str) -> Dict[str, str]:
        """Extract tất cả entities cho hệ thống gọi điện/nhắn tin"""
        entities = {}
        
        # Extract phone number trước để kiểm tra
        phone_result = self.extract_phone_number(text)
        if phone_result:
            entities["PHONE_NUMBER"] = phone_result
        
        # Extract receiver (chỉ khi không có số điện thoại hoặc có tên người cụ thể)
        receiver_result = self.extract_receiver(text)
        if receiver_result:
            # Chỉ lấy RECEIVER, loại bỏ ACTION_TYPE
            if "RECEIVER" in receiver_result:
                entities["RECEIVER"] = receiver_result["RECEIVER"]
        
        # Loại bỏ TIME entity
        # time_result = self.extract_time(text)
        # if time_result:
        #     entities["TIME"] = time_result
        
        # Extract MESSAGE entity khi có
        message_result = self.extract_message(text, entities.get("RECEIVER"))
        if message_result:
            entities["MESSAGE"] = message_result
        
        platform_result = self.extract_platform(text)
        if platform_result:
            entities["PLATFORM"] = platform_result
        
        return entities
