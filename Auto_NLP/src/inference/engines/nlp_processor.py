"""
NLP Processor Module - Main orchestrator
Kết hợp tất cả các engines: Intent, Entity, Value Generation
"""

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
        self.reasoning_engine = None
        if ReasoningEngine:
            try:
                self.reasoning_engine = ReasoningEngine()
            except Exception as e:
                print(f"⚠️ Reasoning engine not available: {e}")
        
        # Intent to command mapping
        self.intent_to_command = {
            "adjust-settings": "adjust_settings",
            "app-tutorial": "app_tutorial", 
            "browse-social-media": "browse_social_media",
            "call": "call",
            "check-device-status": "check_device_status",
            "check-health-status": "check_health_status",
            "check-messages": "check_messages",
            "check-weather": "check_weather",
            "control-device": "control_device",
            "general-conversation": "general_conversation",
            "help": "help",
            "make-call": "make_call",
            "make-video-call": "make_video_call",
            "navigation-help": "navigation_help",
            "open-app": "open_app",
            "open-app-action": "open_app_action",
            "play-audio": "play_audio",
            "play-content": "play_content",
            "play-media": "play_media",
            "provide-instructions": "provide_instructions",
            "read-content": "read_content",
            "read-news": "read_news",
            "search-content": "search_content",
            "search-internet": "search_internet",
            "send-message": "send_message",
            "send-mess": "send_mess",
            "set-alarm": "set_alarm",
            "set-reminder": "set_reminder"
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
        
        # Step 3: Use optimized entities
        entities = communication_result["entities"]
        
        # Step 4: Generate command
        command = self.intent_to_command.get(intent_result["intent"], "unknown")
        
        # Step 5: Generate optimized value
        value = self.communication_optimizer.get_optimized_value(
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
    
    async def process_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Xử lý text với reasoning engine"""
        start_time = datetime.now()
        
        try:
            if not self.reasoning_engine:
                # Fallback to simple processing
                return self.process_text(text)
            
            # Use reasoning engine
            reasoning_result = await self.reasoning_engine.reasoning_predict(text)
            
            # Extract entities using our improved extractor
            entities = self.entity_extractor.extract_all_entities(text)
            
            # Generate command and value
            command = self.intent_to_command.get(reasoning_result["intent"], "unknown")
            value = self.value_generator.generate_value(
                reasoning_result["intent"], 
                entities, 
                text
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "input_text": text,
                "intent": reasoning_result["intent"],
                "confidence": reasoning_result["confidence"],
                "command": command,
                "entities": entities,
                "value": value,
                "method": "reasoning_engine",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "reasoning_details": reasoning_result.get("reasoning_details", {})
            }
            
        except Exception as e:
            print(f"❌ Error in reasoning processing: {e}")
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
        return list(self.intent_to_command.keys())
    
    def get_intent_to_command_mapping(self) -> Dict[str, str]:
        """Lấy mapping intent to command"""
        return self.intent_to_command.copy()
    
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
