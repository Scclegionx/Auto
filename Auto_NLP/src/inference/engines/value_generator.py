"""
Value Generator Module
Tạo giá trị output từ intent và entities
"""

import re
from typing import Dict, List, Optional

class ValueGenerator:
    """Module tạo giá trị output từ intent và entities"""
    
    def __init__(self):
        self.intent_templates = self._build_intent_templates()
        
    def _build_intent_templates(self) -> Dict[str, str]:
        """Xây dựng templates cho các intent - 13 unified commands"""
        return {
            # Communication
            "call": "Gọi cho {receiver} qua {platform}",
            "make-video-call": "Gọi video cho {receiver} qua {platform}",
            "send-mess": "Nhắn tin cho {receiver} qua {platform}: {message}",
            "add-contacts": "Thêm liên lạc {receiver}",
            
            # Media & Content
            "play-media": "Phát {content}",
            "view-content": "Xem {content}",
            
            # Search
            "search-internet": "Tìm kiếm: {query}",
            "search-youtube": "Tìm kiếm trên YouTube: {query}",
            
            # Information
            "get-info": "Thông tin {topic} {when} {location}",
            
            # Reminders & Alarms
            "set-alarm": "Báo thức lúc {time}",
            "set-event-calendar": "Nhắc nhở: {message} lúc {time}",
            
            # Device Control
            "open-cam": "Mở camera",
            "control-device": "Điều khiển {device}",
            
            # Fallback
            "unknown": "Không thể xác định"
        }
    
    def generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        """Tạo giá trị output từ intent và entities - Extract nội dung chính từ text"""
        if intent == "unknown" or intent == "error":
            return "Không thể xác định"
        
        # Extract main content from original text based on intent
        main_content = self._extract_main_content(intent, original_text, entities)
        if main_content:
            return main_content
    
    def _extract_main_content(self, intent: str, original_text: str, entities: Dict[str, str]) -> Optional[str]:
        """Extract nội dung chính từ text dựa trên intent"""
        text_lower = original_text.lower()
        # print(f"Extracting main content for intent: {intent}, text: {original_text}")
        
        if intent == "send-mess":
            # Extract message content after "là" or "rằng"
            message_patterns = [
                r"là\s+(.+?)(?:$|\.)",
                r"rằng\s+(.+?)(?:$|\.)",
                r"nói\s+(.+?)(?:$|\.)",
                r"nhắn\s+(.+?)(?:$|\.)",
                r"là\s+(.+?)$",  # Match to end of string
                r"rằng\s+(.+?)$"  # Match to end of string
            ]
            
            for pattern in message_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:  # Avoid very short content
                        return content
            
            # Fallback: use MESSAGE entity if available
            if entities.get("MESSAGE"):
                return entities.get("MESSAGE")
        
        elif intent in ["search-internet", "search-youtube"]:
            # Extract search query after "là" or "tìm kiếm"
            query_patterns = [
                r"là\s+(.+?)(?:$|\.)",
                r"tìm\s+kiếm\s+(.+?)(?:$|\.)",
                r"search\s+(.+?)(?:$|\.)",
                r"tra\s+cứu\s+(.+?)(?:$|\.)"
            ]
            
            for pattern in query_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:
                        return content
            
            # Fallback: use QUERY entity if available
            if entities.get("QUERY"):
                return entities.get("QUERY")
        
        elif intent == "play-media":
            # Extract media content after "phát" or "nghe"
            media_patterns = [
                r"phát\s+(.+?)(?:$|\.)",
                r"nghe\s+(.+?)(?:$|\.)",
                r"chơi\s+(.+?)(?:$|\.)",
                r"mở\s+(.+?)(?:$|\.)"
            ]
            
            for pattern in media_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:
                        return content
            
            # Fallback: use CONTENT entity if available
            if entities.get("CONTENT"):
                return entities.get("CONTENT")
        
        elif intent == "view-content":
            # Extract content to view after "xem" or "đọc"
            view_patterns = [
                r"xem\s+(.+?)(?:$|\.)",
                r"đọc\s+(.+?)(?:$|\.)",
                r"mở\s+(.+?)(?:$|\.)"
            ]
            
            for pattern in view_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:
                        return content
            
            # Fallback: use CONTENT entity if available
            if entities.get("CONTENT"):
                return entities.get("CONTENT")
        
        elif intent == "get-info":
            # Extract info query after "thông tin" or "kiểm tra"
            info_patterns = [
                r"thông\s+tin\s+(.+?)(?:$|\.)",
                r"kiểm\s+tra\s+(.+?)(?:$|\.)",
                r"đọc\s+(.+?)(?:$|\.)"
            ]
            
            for pattern in info_patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:
                        return content
            
            # Fallback: use TOPIC entity if available
            if entities.get("TOPIC"):
                return entities.get("TOPIC")
        
        # Fallback: Nếu intent không được nhận diện đúng, extract từ text
        # Kiểm tra các từ khóa tìm kiếm trong text
        search_keywords = ["tìm kiếm", "search", "tìm", "tra cứu", "kiểm tra", "thông tin"]
        for keyword in search_keywords:
            if keyword in text_lower:
                # Extract phần sau từ khóa
                pattern = rf"{keyword}\s+(.+?)(?:$|\.)"
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) > 3:
                        return content
                else:
                    # Nếu không match pattern, return toàn bộ text
                    return original_text
        
        # Nếu không có keyword nào match, return None để fallback
        return None
        
        # Các intent khác
        if intent == "set-alarm":
            # Extract alarm details
            if entities.get("TIME"):
                time_info = entities.get("TIME")
                label_info = entities.get("LABEL", "")
                if label_info:
                    return f"Báo thức lúc {time_info} - {label_info}"
                else:
                    return f"Báo thức lúc {time_info}"
        
        elif intent == "set-event-calendar":
            # Extract event details
            if entities.get("TITLE"):
                title = entities.get("TITLE")
                time_info = entities.get("TIME", "")
                if time_info:
                    return f"{title} lúc {time_info}"
                else:
                    return title
        
        return None
        
        # Fallback: use template with entities
        if not isinstance(entities, dict):
            entities = {}
        
        # Extract entities safely
        receiver = entities.get("RECEIVER", "")
        message = entities.get("MESSAGE", "")
        time = entities.get("TIME", "")
        query = entities.get("QUERY", "")
        app = entities.get("APP", "")
        content = entities.get("CONTENT", "")
        platform = entities.get("PLATFORM", "")
        topic = entities.get("TOPIC", "")
        when = entities.get("WHEN", "")
        location = entities.get("LOCATION", "")
        
        # Generate value based on intent using templates
        if intent in self.intent_templates:
            template = self.intent_templates[intent]
            
            # Prepare template variables
            template_vars = {
                "receiver": receiver or "người nhận",
                "message": message or "tin nhắn",
                "time": time or "thời gian",
                "query": query or "từ khóa",
                "content": content or "nội dung",
                "platform": platform or "web",
                "device": entities.get("DEVICE", "thiết bị"),
                "topic": topic or "",
                "when": when or "",
                "location": location or ""
            }
            
            # Fill template
            try:
                return template.format(**template_vars)
            except KeyError as e:
                # Fallback nếu template có key không tồn tại
                return f"Thực hiện: {intent}"
        
        # Fallback cho intent không có template
        return f"Thực hiện: {intent}"
    
    def _extract_receiver_fallback(self, text: str) -> str:
        """Fallback method để extract receiver"""
        patterns = [
            r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nhắn\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower(), re.IGNORECASE)
            if match and match.group(1):
                receiver = match.group(1).strip()
                unwanted_words = ["rằng", "là", "nói", "nhắn", "gửi", "lúc", "vào", "nhé", "nha", "ạ", "nhá"]
                words = receiver.split()
                cleaned_words = [word for word in words if word.lower() not in unwanted_words]
                if cleaned_words:
                    return " ".join(cleaned_words[:2])
        
        return ""
    
    def _extract_message_fallback(self, text: str) -> str:
        """Fallback method để extract message content - Cải thiện cho tin nhắn dài"""
        text_lower = text.lower()
        
        # Patterns để extract message content - Cải thiện cho tin nhắn dài
        patterns = [
            # Pattern 1: Rằng (ưu tiên cao)
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nói\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 2: Nhắn tin cho [người] rằng [nội dung]
            r"nhắn\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 3: Nhắn tin với nội dung là [nội dung]
            r"nhắn\s+tin\s+.*?nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+tin\s+.*?nội\s+dung\s+là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 4: Với nội dung [nội dung]
            r"với\s+nội\s+dung\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            
            # Pattern 5: Fallback - lấy phần sau "cho" và "rằng"
            r"cho\s+[\w\s]+\s+rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                message = match.group(1).strip()
                if message and len(message) > 3:
                    return message
        
        # Fallback cuối cùng: trả về toàn bộ text nếu không tìm thấy pattern
        return text
    
    def format_entities_for_display(self, entities: Dict[str, str]) -> str:
        """Format entities để hiển thị"""
        if not entities:
            return "Không có entities"
        
        formatted = []
        for key, value in entities.items():
            formatted.append(f"{key}: {value}")
        
        return " | ".join(formatted)
    
    def validate_entities(self, intent: str, entities: Dict[str, str]) -> Dict[str, str]:
        """Validate và làm sạch entities"""
        validated = {}
        
        for key, value in entities.items():
            if value and isinstance(value, str) and len(value.strip()) > 0:
                validated[key] = value.strip()
        
        # Intent-specific validation
        if intent in ["call", "call", "make-video-call", "send-mess", "send-mess"]:
            if "RECEIVER" not in validated:
                validated["RECEIVER"] = "người nhận"
        
        if intent in ["send-mess", "send-mess"]:
            if "PLATFORM" not in validated:
                validated["PLATFORM"] = "sms"
        
        return validated
    
    def _convert_phone_numbers_in_text(self, text: str) -> str:
        """Chuyển đổi số điện thoại từ chữ sang số trong text"""
        # Mapping từ chữ sang số
        number_mapping = {
            "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
            "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9"
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        result_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Kiểm tra xem có phải bắt đầu của số điện thoại không
            if word in number_mapping:
                # Tìm chuỗi số điện thoại liên tiếp (10-11 từ)
                phone_digits = []
                j = i
                
                while j < len(words) and len(phone_digits) < 11:
                    if words[j] in number_mapping:
                        phone_digits.append(number_mapping[words[j]])
                        j += 1
                    else:
                        break
                
                # Nếu tìm thấy 10-11 chữ số liên tiếp
                if 10 <= len(phone_digits) <= 11:
                    phone_number = ''.join(phone_digits)
                    # Kiểm tra số điện thoại Việt Nam hợp lệ
                    if phone_number.startswith(('03', '05', '07', '08', '09')):
                        result_words.append(phone_number)
                        i = j  # Bỏ qua các từ đã xử lý
                        continue
                    else:
                        # Nếu không hợp lệ, thêm từ gốc
                        result_words.append(word)
                else:
                    # Nếu không đủ 10-11 chữ số, thêm từ gốc
                    result_words.append(word)
            else:
                # Thêm từ gốc nếu không phải số
                result_words.append(word)
            
            i += 1
        
        # Khôi phục case gốc cho các từ không phải số
        original_words = text.split()
        final_words = []
        
        for i, word in enumerate(result_words):
            if word.isdigit():
                final_words.append(word)
            else:
                # Khôi phục case gốc
                if i < len(original_words):
                    original_word = original_words[i]
                    # Giữ nguyên case của từ gốc
                    if original_word.isupper():
                        final_words.append(word.upper())
                    elif original_word.istitle():
                        final_words.append(word.title())
                    else:
                        final_words.append(word)
                else:
                    final_words.append(word)
        
        return ' '.join(final_words)
