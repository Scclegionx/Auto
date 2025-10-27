from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from .intent_predictor import IntentPredictor
from .entity_extractor import EntityExtractor
from .value_generator import ValueGenerator
from .communication_optimizer import CommunicationOptimizer

try:
    from .reasoning_engine import ReasoningEngine
except ImportError:
    ReasoningEngine = None

class NLPProcessor:
    """Main NLP processor kết hợp tất cả engines"""
    
    def __init__(self, device=None):
        self.device = device
        
        # Initialize engines
        self.intent_predictor = IntentPredictor(device)
        self.entity_extractor = EntityExtractor()
        self.value_generator = ValueGenerator()
        self.communication_optimizer = CommunicationOptimizer()
        
        # Initialize reasoning engine if available
        # Initialize reasoning engine
        self.reasoning_engine = None
        if ReasoningEngine:
            try:
                self.reasoning_engine = ReasoningEngine()
                print("INFO: Reasoning engine loaded successfully")
            except Exception as e:
                print(f"WARNING: Reasoning engine not available: {e}")
        
        # 1) 13 command chuẩn + display map tách riêng
        self.valid_commands = [
            "call","make-video-call","send-mess","add-contacts",
            "play-media","view-content",
            "search-internet","search-youtube",
            "get-info",
            "set-alarm","set-event-calendar",
            "open-cam","control-device"
        ]
        self.display_map = {
            "call":"Gọi điện","make-video-call":"Gọi video","send-mess":"Gửi tin nhắn","add-contacts":"Thêm liên lạc",
            "play-media":"Phát media","view-content":"Xem nội dung",
            "search-internet":"Tìm kiếm internet","search-youtube":"Tìm trên YouTube",
            "get-info":"Lấy thông tin",
            "set-alarm":"Đặt báo thức","set-event-calendar":"Tạo/nhắc lịch",
            "open-cam":"Mở camera","control-device":"Điều khiển thiết bị"
        }
        
        # 2) Chuẩn hóa intent cũ → mới
        self.command_normalization = {
            # giữ nguyên
            "call":"call","send-mess":"send-mess","make-video-call":"make-video-call","play-media":"play-media",
            "view-content":"view-content","search-internet":"search-internet","search-youtube":"search-youtube",
            "get-info":"get-info","set-alarm":"set-alarm","set-event-calendar":"set-event-calendar",
            "open-cam":"open-cam","control-device":"control-device",
            # map cũ → mới
            "play-content":"play-media","play-audio":"play-media",
            "read-news":"get-info","check-weather":"get-info","check-date":"get-info",
            "search-content":"search-internet",
            "set-reminder":"set-event-calendar",
            # các intent không dùng: để "unknown"
            "check-messages":"unknown","open-app":"unknown","open-app-action":"unknown",
            "check-health-status":"unknown","check-device-status":"unknown","read-content":"unknown",
            "adjust-settings":"unknown","browse-social-media":"unknown","general-conversation":"unknown",
            "help":"unknown","navigation-help":"unknown","provide-instructions":"unknown"
        }
    
    def load_model(self, model_path: str, tokenizer_name: str = "vinai/phobert-base") -> bool:
        """Load trained model"""
        return self.intent_predictor.load_model(model_path, tokenizer_name)
    
    def process_text(self, text: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Xử lý text và trả về kết quả đầy đủ - Tối ưu cho gọi điện/nhắn tin"""
        start_time = datetime.now()
        
        # Step 1: Predict intent
        intent_result = self.intent_predictor.predict_intent(text, confidence_threshold)
        
        # Step 2: Optimize for communication
        communication_result = self.communication_optimizer.optimize_for_communication(text)
        
        # Step 3: Generate command with normalization
        intent = intent_result["intent"]
        
        # Step 4: Extract entities with intent-specific extraction
        # Đảm bảo platform extraction hoạt động ngay cả khi intent = unknown
        if intent == "unknown":
            # Fallback: extract tất cả entities (legacy behavior)
            entities = self.entity_extractor._extract_all_legacy_entities(text)
            # Đảm bảo platform extraction hoạt động
            platform_result = self.entity_extractor.extract_platform(text)
            if platform_result:
                entities["PLATFORM"] = platform_result
        else:
            entities = self.entity_extractor.extract_all_entities(text, intent)
        
        # Step 4.5: Normalize entities theo command để tránh lộn xộn
        entities = self._normalize_entities_by_command(intent, entities)
        
        normalized_intent = self.command_normalization.get(intent, intent)
        command = normalized_intent if normalized_intent in self.valid_commands else "unknown"
        command_display = self.display_map.get(command, command)
        
        # Step 5: Generate optimized value with error handling
        try:
            # Use communication optimizer for communication intents (except send-mess for better value extraction)
            if intent in ["call", "make-video-call", "add-contacts"]:
                value = self.communication_optimizer.get_optimized_value(
                    intent_result["intent"], 
                    entities, 
                    text
                )
            else:
                # Use value generator for non-communication intents
                value = self.value_generator.generate_value(
                    intent_result["intent"], 
                    entities, 
                    text
                )
        except Exception as e:
            print(f"⚠️ Error in value generation: {e}")
            # Fallback to value generator
            value = self.value_generator.generate_value(
                intent_result["intent"], 
                entities, 
                text
            )
        
        # Step 6: Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "input_text": text,
            "intent": intent_result["intent"],
            "confidence": max(intent_result["confidence"], communication_result["confidence"]),
            "command": command,
            "entities": entities,
            "value": value,
            "method": intent_result.get("method", "trained_model"),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "communication_type": communication_result["communication_type"]
        }
    
    def _normalize_entities_by_command(self, intent: str, entities: Dict[str, str]) -> Dict[str, str]:
        """Normalize entities theo command để tránh lộn xộn trong JSON output"""
        
        # Command-specific entity mapping
        command_entity_mapping = {
            # Communication Commands
            "call": {
                "required": ["RECEIVER", "PLATFORM"],
                "optional": ["PHONE_NUMBER", "TIME"],
                "remove": ["MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE"]
            },
            "make-video-call": {
                "required": ["RECEIVER", "PLATFORM"],
                "optional": ["TIME"],
                "remove": ["MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            "send-mess": {
                "required": ["RECEIVER", "PLATFORM"],
                "optional": ["MESSAGE", "TIME"],
                "remove": ["QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            
            # Media Commands
            "play-media": {
                "required": [],
                "optional": ["FILE_PATH", "ARTIST", "PLAYLIST", "GENRE", "PODCAST", "RADIO"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "PHONE_NUMBER"]
            },
            "view-content": {
                "required": [],
                "optional": ["CONTENT_TYPE", "URL", "TITLE", "SOURCE"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE"]
            },
            
            # Search Commands
            "search-internet": {
                "required": ["QUERY"],
                "optional": ["PLATFORM", "LANGUAGE", "PREFERENCE"],
                "remove": ["RECEIVER", "MESSAGE", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            "search-youtube": {
                "required": ["QUERY"],
                "optional": ["PLATFORM", "CONTENT_TYPE", "DURATION"],
                "remove": ["RECEIVER", "MESSAGE", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            
            # Information Commands
            "get-info": {
                "required": [],
                "optional": ["INFO_TYPE", "TOPIC", "LOCATION", "TIME", "SOURCE"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            
            # Time-based Commands
            "set-alarm": {
                "required": ["TIME"],
                "optional": ["LABEL", "DURATION", "VOLUME", "REPEAT"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            "set-event-calendar": {
                "required": ["TITLE", "TIME"],
                "optional": ["LOCATION", "DURATION", "REPEAT", "ATTENDEES"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            
            # Device Commands
            "control-device": {
                "required": ["DEVICE_TYPE", "ACTION"],
                "optional": ["LEVEL", "MODE", "SETTINGS"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            "open-cam": {
                "required": ["CAMERA_TYPE"],
                "optional": ["MODE", "SETTINGS"],
                "remove": ["RECEIVER", "MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE", "PHONE_NUMBER"]
            },
            
            # Contact Commands
            "add-contacts": {
                "required": ["CONTACT_NAME"],
                "optional": ["PHONE_NUMBER", "EMAIL", "RELATIONSHIP"],
                "remove": ["MESSAGE", "QUERY", "FILE_PATH", "ARTIST", "GENRE"]
            }
        }
        
        # Get mapping for current intent
        mapping = command_entity_mapping.get(intent, {
            "required": [],
            "optional": [],
            "remove": []
        })
        
        # Clean up entities
        cleaned_entities = {}
        
        # Keep required and optional entities
        for key, value in entities.items():
            if key in mapping["required"] or key in mapping["optional"]:
                cleaned_entities[key] = value
        
        # Remove unwanted entities
        for key in mapping["remove"]:
            if key in cleaned_entities:
                del cleaned_entities[key]
        
        # Add default values for required entities if missing
        for required_entity in mapping["required"]:
            if required_entity not in cleaned_entities:
                if required_entity == "PLATFORM":
                    cleaned_entities[required_entity] = "phone" if intent in ["call", "make-video-call"] else "sms"
                elif required_entity == "RECEIVER":
                    cleaned_entities[required_entity] = "unknown"
                elif required_entity == "TIME":
                    cleaned_entities[required_entity] = "now"
                elif required_entity == "QUERY":
                    cleaned_entities[required_entity] = "search"
                elif required_entity == "TITLE":
                    cleaned_entities[required_entity] = "event"
                elif required_entity == "DEVICE_TYPE":
                    cleaned_entities[required_entity] = "device"
                elif required_entity == "ACTION":
                    cleaned_entities[required_entity] = "control"
                elif required_entity == "CAMERA_TYPE":
                    cleaned_entities[required_entity] = "camera"
                elif required_entity == "CONTACT_NAME":
                    cleaned_entities[required_entity] = "contact"
        
        return cleaned_entities
    
    async def process_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Xử lý text với reasoning engine"""
        start_time = datetime.now()
        
        try:
            if not self.reasoning_engine:
                # Fallback to simple processing
                return self.process_text(text)
            
            # Use reasoning engine (not async)
            reasoning_result = self.reasoning_engine.reasoning_predict(text)
            
            # Kiểm tra reasoning_result là dict, không phải string
            if not isinstance(reasoning_result, dict):
                print(f"⚠️ reasoning_result is not a dict: {type(reasoning_result)} - {reasoning_result}")
                # Fallback to simple processing
                return self.process_text(text)
            
            # Extract entities using our improved extractor with intent-specific extraction
            intent = reasoning_result.get("intent", "unknown")
            entities = self.entity_extractor.extract_all_entities(text, intent)
            
            # Generate command and value with normalization
            intent = reasoning_result.get("intent", "unknown")
            normalized_intent = self.command_normalization.get(intent, intent)
            command = normalized_intent if normalized_intent in self.valid_commands else "unknown"
            command_display = self.display_map.get(command, command)
            value = self.value_generator.generate_value(
                reasoning_result.get("intent", "unknown"), 
                entities, 
                text
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "input_text": text,
                "intent": reasoning_result.get("intent", "unknown"),
                "confidence": reasoning_result.get("confidence", 0.0),
                "command": command,
                "command_display": command_display,
                "entities": entities,
                "value": value,
                "method": "reasoning_engine",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "reasoning_details": reasoning_result.get("reasoning_details", {})
            }
            
        except Exception as e:
            print(f"ERROR Error in reasoning processing: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple processing
            result = self.process_text(text)
            result["method"] = "reasoning_engine_fallback"
            result["error"] = str(e)
            return result
    
    async def batch_process(self, texts: List[str], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Xử lý nhiều texts cùng lúc"""
        results = []
        
        for text in texts:
            try:
                result = self.process_text(text, confidence_threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    "input_text": text,
                    "intent": "error",
                    "confidence": 0.0,
                    "command": "error",
                    "entities": {},
                    "value": f"Lỗi xử lý: {str(e)}",
                    "method": "error",
                    "processing_time": 0.0,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    async def batch_process_with_reasoning(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Xử lý nhiều texts với reasoning engine"""
        results = []
        
        for text in texts:
            try:
                result = await self.process_with_reasoning(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "input_text": text,
                    "intent": "error",
                    "confidence": 0.0,
                    "command": "error",
                    "entities": {},
                    "value": f"Lỗi xử lý: {str(e)}",
                    "method": "error",
                    "processing_time": 0.0,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_available_intents(self) -> List[str]:
        """Lấy danh sách intents có sẵn"""
        if self.intent_predictor.id_to_intent:
            return list(self.intent_predictor.id_to_intent.values())
        return self.valid_commands
    
    def get_intent_to_command_mapping(self) -> Dict[str, str]:
        """Lấy mapping intent to command"""
        return self.command_normalization.copy()
    
    def get_valid_commands(self) -> List[str]:
        """Lấy danh sách 13 command chuẩn"""
        return self.valid_commands.copy()
    
    def get_display_map(self) -> Dict[str, str]:
        """Lấy mapping command to display name"""
        return self.display_map.copy()
    
    def is_model_loaded(self) -> bool:
        """Kiểm tra model đã được load chưa"""
        return self.intent_predictor.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model"""
        return {
            "model_loaded": self.is_model_loaded(),
            "device": str(self.device),
            "available_intents": len(self.get_available_intents()),
            "reasoning_engine_available": self.reasoning_engine is not None
        }
