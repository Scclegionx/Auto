"""
Communication Optimizer Module
Tối ưu hóa hệ thống cho gọi điện và nhắn tin
"""

import re
from typing import Dict, List, Optional, Tuple
from .entity_extractor import EntityExtractor
from .value_generator import ValueGenerator

class CommunicationOptimizer:
    """Module tối ưu hóa cho hệ thống gọi điện/nhắn tin"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.value_generator = ValueGenerator()
        
        # Từ khóa ưu tiên cho gọi điện/nhắn tin
        self.call_keywords = [
            "gọi", "alo", "gọi điện", "gọi thoại", "gọi ngay", "gọi khẩn cấp"
        ]
        
        self.video_call_keywords = [
            "gọi video", "facetime", "video call", "video"
        ]
        
        self.message_keywords = [
            "nhắn", "gửi", "tin nhắn", "sms", "soạn tin", "liên lạc"
        ]
        
        # Quan hệ gia đình ưu tiên
        self.family_relations = [
            "bố", "mẹ", "ông", "bà", "anh", "chị", "em", "con", "cháu", 
            "chú", "bác", "cô", "dì", "dượng", "mợ", "cả nhà", "gia đình"
        ]
        
        # Platforms ưu tiên
        self.priority_platforms = {
            "call": ["phone", "zalo", "messenger"],
            "message": ["sms", "zalo", "messenger"],
            "video": ["zalo", "messenger"]
        }
    
    def optimize_for_communication(self, text: str) -> Dict[str, any]:
        """Tối ưu hóa text cho hệ thống gọi điện/nhắn tin"""
        text_lower = text.lower()
        
        # Xác định loại giao tiếp
        communication_type = self._detect_communication_type(text_lower)
        
        # Extract entities với tối ưu hóa
        entities = self._extract_optimized_entities(text, communication_type)
        
        # Tối ưu hóa platform
        entities = self._optimize_platform(entities, communication_type, text_lower)
        
        # Tối ưu hóa receiver
        entities = self._optimize_receiver(entities, text_lower)
        
        # Tối ưu hóa message
        entities = self._optimize_message(entities, text_lower)
        
        return {
            "communication_type": communication_type,
            "entities": entities,
            "confidence": self._calculate_confidence(entities, communication_type)
        }
    
    def _detect_communication_type(self, text_lower: str) -> str:
        """Xác định loại giao tiếp"""
        call_score = sum(1 for keyword in self.call_keywords if keyword in text_lower)
        video_call_score = sum(1 for keyword in self.video_call_keywords if keyword in text_lower)
        message_score = sum(1 for keyword in self.message_keywords if keyword in text_lower)
        
        # Ưu tiên message nếu có từ khóa "nhắn" hoặc "gửi"
        if "nhắn" in text_lower or "gửi" in text_lower or "tin" in text_lower:
            message_score += 2
        
        if video_call_score > 0:
            return "video_call"
        elif message_score > call_score:
            return "message"
        elif call_score > 0:
            return "call"
        else:
            return "unknown"
    
    def _extract_optimized_entities(self, text: str, communication_type: str) -> Dict[str, str]:
        """Extract entities với tối ưu hóa"""
        try:
            entities = self.entity_extractor.extract_all_entities(text)
            
            # Ensure entities is a dict
            if not isinstance(entities, dict):
                entities = {}
            
            # Thêm ACTION_TYPE nếu có
            if "ACTION_TYPE" in entities:
                action_type = entities.pop("ACTION_TYPE")
                entities["ACTION_TYPE"] = action_type
            
            # Đặc biệt xử lý cho message type
            if communication_type == "message":
                # Extract message content nếu chưa có
                if "MESSAGE" not in entities or not entities["MESSAGE"]:
                    message = self._extract_message_fallback(text.lower())
                    if message:
                        entities["MESSAGE"] = message
            
            return entities
        except Exception as e:
            print(f"⚠️ Error in entity extraction: {e}")
            return {}
    
    def _optimize_platform(self, entities: Dict[str, str], communication_type: str, text_lower: str) -> Dict[str, str]:
        """Tối ưu hóa platform dựa trên context"""
        current_platform = entities.get("PLATFORM", "")
        
        if communication_type == "call":
            if not current_platform or current_platform == "sms":
                # Ưu tiên phone cho gọi điện
                entities["PLATFORM"] = "phone"
        elif communication_type == "message":
            if not current_platform:
                # Ưu tiên sms cho nhắn tin
                entities["PLATFORM"] = "sms"
        elif communication_type == "video_call":
            if not current_platform or current_platform == "phone":
                # Ưu tiên zalo/messenger cho video call
                entities["PLATFORM"] = "zalo"
        
        return entities
    
    def _optimize_receiver(self, entities: Dict[str, str], text_lower: str) -> Dict[str, str]:
        """Tối ưu hóa receiver extraction"""
        receiver = entities.get("RECEIVER", "")
        
        if not receiver:
            # Fallback extraction
            receiver = self._extract_receiver_fallback(text_lower)
            if receiver:
                entities["RECEIVER"] = receiver
        
        # Làm sạch receiver
        if receiver:
            entities["RECEIVER"] = self._clean_receiver_optimized(receiver)
        
        return entities
    
    def _optimize_message(self, entities: Dict[str, str], text_lower: str) -> Dict[str, str]:
        """Tối ưu hóa message extraction"""
        message = entities.get("MESSAGE", "")
        
        if not message and ("rằng" in text_lower or "nhắn" in text_lower or "gửi" in text_lower):
            # Fallback extraction cho message
            message = self._extract_message_fallback(text_lower)
            if message:
                entities["MESSAGE"] = message
        
        return entities
    
    def _extract_receiver_fallback(self, text_lower: str) -> str:
        """Fallback extraction cho receiver"""
        patterns = [
            r"cho\s+([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gọi\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nhắn\s+(?:cho|tới|đến)?\s*([\w\s]+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                receiver = match.group(1).strip()
                if receiver:
                    return receiver
        
        return ""
    
    def _extract_message_fallback(self, text_lower: str) -> str:
        """Fallback extraction cho message"""
        patterns = [
            r"rằng\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"là\s+(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"nhắn\s+(?:cho\s+[\w\s]+?\s+)?(?:rằng\s+)?(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
            r"gửi\s+(?:cho\s+[\w\s]+?\s+)?(?:rằng\s+)?(.+?)(?:\s+(?:nhé|nha|ạ|nhá))?(?:$|[\.,])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match and match.group(1):
                message = match.group(1).strip()
                if message and len(message) > 3:
                    return message
        
        return ""
    
    def _clean_receiver_optimized(self, receiver: str) -> str:
        """Làm sạch receiver với tối ưu hóa"""
        unwanted_words = [
            "rằng", "là", "nói", "nhắn", "gửi", "lúc", "vào", "nhé", "nha", "ạ", "nhá", 
            "ngay", "bây giờ", "qua", "messenger", "zalo", "facebook", "telegram", 
            "instagram", "tiktok", "sms", "tin", "nhắn", "gửi", "cho", "tới", "đến",
            "chiều", "sáng", "trưa", "tối", "đêm", "nay", "mai", "hôm", "ngày", "tuần", "tháng",
            "của", "ở", "tại", "với", "và", "hoặc", "hay", "nếu", "khi", "sau", "trước"
        ]
        
        words = receiver.split()
        cleaned_words = []
        
        for word in words:
            if word.lower() not in unwanted_words:
                cleaned_words.append(word)
        
        # Giới hạn 2-3 từ
        if len(cleaned_words) > 3:
            cleaned_words = cleaned_words[:3]
        
        return " ".join(cleaned_words).strip()
    
    def _calculate_confidence(self, entities: Dict[str, str], communication_type: str) -> float:
        """Tính toán confidence score"""
        base_confidence = 0.5
        
        # Tăng confidence nếu có receiver
        if entities.get("RECEIVER"):
            base_confidence += 0.3
        
        # Tăng confidence nếu có platform phù hợp
        platform = entities.get("PLATFORM", "")
        if communication_type == "call" and platform in ["phone", "zalo", "messenger"]:
            base_confidence += 0.1
        elif communication_type == "message" and platform in ["sms", "zalo", "messenger"]:
            base_confidence += 0.1
        elif communication_type == "video_call" and platform in ["zalo", "messenger"]:
            base_confidence += 0.1
        
        # Tăng confidence nếu có message (cho nhắn tin)
        if communication_type == "message" and entities.get("MESSAGE"):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def get_optimized_value(self, intent: str, entities: Dict[str, str], original_text: str) -> str:
        """Tạo value tối ưu hóa"""
        try:
            return self.value_generator.generate_value(intent, entities, original_text)
        except Exception as e:
            print(f"⚠️ Error in value generator: {e}")
            # Fallback to simple value generation
            if intent in ["send-mess", "send-message"]:
                message = entities.get("MESSAGE", "")
                if message:
                    return message
                else:
                    return "Tin nhắn"
            elif intent in ["call", "make-call"]:
                receiver = entities.get("RECEIVER", "người nhận")
                return f"Gọi cho {receiver}"
            else:
                return f"Thực hiện: {intent}"
    
    def validate_communication_entities(self, entities: Dict[str, str], communication_type: str) -> Dict[str, str]:
        """Validate entities cho giao tiếp"""
        validated = {}
        
        for key, value in entities.items():
            if value and isinstance(value, str) and len(value.strip()) > 0:
                validated[key] = value.strip()
        
        # Validation theo loại giao tiếp
        if communication_type in ["call", "video_call"]:
            if "RECEIVER" not in validated:
                validated["RECEIVER"] = "người nhận"
            if "PLATFORM" not in validated:
                validated["PLATFORM"] = "phone" if communication_type == "call" else "zalo"
        
        elif communication_type == "message":
            if "RECEIVER" not in validated:
                validated["RECEIVER"] = "người nhận"
            if "PLATFORM" not in validated:
                validated["PLATFORM"] = "sms"
            if "MESSAGE" not in validated:
                validated["MESSAGE"] = "Tin nhắn"
        
        return validated
