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
        """Xây dựng templates cho các intent"""
        return {
            "call": "Gọi cho {receiver}",
            "make-call": "Gọi cho {receiver}",
            "make-video-call": "Gọi video cho {receiver}",
            "send-mess": "Nhắn tin cho {receiver}: {message}",
            "send-message": "Nhắn tin cho {receiver}: {message}",
            "set-reminder": "Nhắc nhở: {message} lúc {time}",
            "set-alarm": "Báo thức lúc {time}",
            "search": "Tìm kiếm: {query}",
            "open-app": "Mở ứng dụng {app}",
            "play-media": "Phát {content}",
            "check-weather": "Kiểm tra thời tiết",
            "check-messages": "Kiểm tra tin nhắn",
            "help": "Trợ giúp",
            "unknown": "Không thể xác định"
        }
    
    def generate_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        """Tạo giá trị output từ intent và entities"""
        if intent == "unknown" or intent == "error":
            return "Không thể xác định"
        
        # Ensure entities is a dict
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
        
        # Generate value based on intent
        if intent in ["call", "make-call", "make-video-call"]:
            # Ưu tiên trả về số điện thoại nếu có
            phone_number = entities.get("PHONE_NUMBER", "")
            if phone_number:
                return phone_number
            
            # Nếu không có số điện thoại, trả về receiver
            if not receiver:
                receiver = self._extract_receiver_fallback(original_text)
            if not receiver:
                receiver = "người nhận"
            
            # Trả về receiver thay vì câu lệnh mô tả
            return receiver
        
        elif intent in ["send-mess", "send-message", "MESSAGE"]:
            # For message intents, extract the full message content
            if message:
                # Chuyển đổi số điện thoại từ chữ sang số trong message
                processed_message = self._convert_phone_numbers_in_text(message)
                return processed_message
            else:
                # Fallback: extract message from original text
                message_fallback = self._extract_message_fallback(original_text)
                if message_fallback:
                    # Chuyển đổi số điện thoại từ chữ sang số trong message
                    processed_message = self._convert_phone_numbers_in_text(message_fallback)
                    return processed_message
                else:
                    # Chuyển đổi số điện thoại từ chữ sang số trong original text
                    processed_text = self._convert_phone_numbers_in_text(original_text)
                    return processed_text
        
        elif intent == "set-reminder":
            if message and time:
                return f"{message} lúc {time}"
            elif message:
                return message
            elif time:
                return f"Nhắc nhở lúc {time}"
            else:
                return "Nhắc nhở"
        
        elif intent == "set-alarm":
            if time:
                return time
            else:
                return "Báo thức"
        
        elif intent == "search":
            if query:
                return query
            else:
                return "Tìm kiếm"
        
        elif intent == "open-app":
            if app:
                return f"Mở ứng dụng {app}"
            else:
                return "Mở ứng dụng"
        
        elif intent == "play-media":
            if content:
                return content
            else:
                return "Media"
        
        else:
            # Fallback: return original text if no specific template found
            return original_text
    
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
        if intent in ["call", "make-call", "make-video-call", "send-mess", "send-message"]:
            if "RECEIVER" not in validated:
                validated["RECEIVER"] = "người nhận"
        
        if intent in ["send-mess", "send-message"]:
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
