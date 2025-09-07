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
        
        # Get template for intent
        template = self.intent_templates.get(intent, "Không thể xác định")
        
        # Extract entities
        receiver = entities.get("RECEIVER", "")
        message = entities.get("MESSAGE", "")
        time = entities.get("TIME", "")
        query = entities.get("QUERY", "")
        app = entities.get("APP", "")
        content = entities.get("CONTENT", "")
        platform = entities.get("PLATFORM", "")
        
        # Generate value based on intent
        if intent in ["call", "make-call", "make-video-call"]:
            if not receiver:
                receiver = self._extract_receiver_fallback(original_text)
            if not receiver:
                receiver = "người nhận"
            
            platform = entities.get("PLATFORM", "phone")
            if intent == "make-video-call":
                return f"Gọi video qua {platform} cho {receiver}"
            else:
                return f"Gọi qua {platform} cho {receiver}"
        
        elif intent in ["send-mess", "send-message"]:
            if message:
                return message
            else:
                return "Tin nhắn"
        
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
            return template
    
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
