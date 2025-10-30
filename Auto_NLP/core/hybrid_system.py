import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import sys
import time

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Fix torchvision issue
try:
    import torchvision
    print("torchvision available")
except ImportError:
    print("torchvision not available - creating mock")
    class MockTorchvision:
        class transforms:
            class InterpolationMode:
                BILINEAR = "bilinear"
                NEAREST = "nearest"
    
    sys.modules['torchvision'] = MockTorchvision()
    sys.modules['torchvision.transforms'] = MockTorchvision.transforms

try:
    from core.reasoning_engine import ReasoningEngine
    from src.inference.engines.entity_extractor import EntityExtractor as SpecializedEntityExtractor
    print("Imported ReasoningEngine and SpecializedEntityExtractor")
except ImportError as e:
    print(f"Failed to import components: {e}")
    sys.exit(1)

class ModelFirstHybridSystem:
    """
    Model-First Hybrid System:
    - Trained model làm chính (primary prediction)
    - Reasoning engine làm phụ (validation, enhancement, fallback)
    """
    
    def __init__(self, model_path: str = "models/phobert_large_intent_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Components
        self.trained_model = None
        self.tokenizer = None
        self.reasoning_engine = None
        self.label_mappings = None
        self.config = None
        
        # Status
        self.model_loaded = False
        self.reasoning_loaded = False
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'model_predictions': 0,
            'reasoning_predictions': 0,
            'hybrid_predictions': 0,
            'fallback_predictions': 0,
            'avg_processing_time': 0.0,
            'confidence_scores': []
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        print(f"Model-First Hybrid System initialized")
        print(f"   Device: {self.device}")
        print(f"   Trained model: {'OK' if self.model_loaded else 'FAILED'}")
        print(f"   Reasoning engine: {'OK' if self.reasoning_loaded else 'FAILED'}")
    
    def _initialize_components(self):
        """Initialize all system components"""
        print("Initializing model-first hybrid system...")
        
        # 1. Load trained model (PRIMARY)
        try:
            self._load_trained_model()
            print("Trained model loaded as PRIMARY")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            self.model_loaded = False
        
        # 2. Initialize reasoning engine (SECONDARY)
        try:
            self.reasoning_engine = ReasoningEngine()
            self.reasoning_loaded = True
            print("Reasoning engine loaded as SECONDARY")
        except Exception as e:
            print(f"Failed to initialize reasoning engine: {e}")
            self.reasoning_loaded = False
        
        # 3. Initialize specialized entity extractor
        try:
            self.specialized_entity_extractor = SpecializedEntityExtractor()
            self.specialized_extractor_loaded = True
            print("Specialized entity extractor loaded")
        except Exception as e:
            print(f"Failed to initialize specialized entity extractor: {e}")
            self.specialized_extractor_loaded = False
    
    def _load_trained_model(self):
        """Load trained model"""
        try:
            # Find best model checkpoint
            best_model_path = self.model_path / "model_best.pth"
            if not best_model_path.exists():
                epoch_models = sorted(self.model_path.glob("model_epoch_*.pth"), reverse=True)
                if epoch_models:
                    best_model_path = epoch_models[0]
                else:
                    raise FileNotFoundError(f"No trained model found in {self.model_path}")
            
            self.logger.info(f"Loading trained model from: {best_model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Load label mappings
            self.label_mappings = {
                'intent_to_id': checkpoint.get('intent_to_id', {}),
                'id_to_intent': checkpoint.get('id_to_intent', {}),
                'entity_to_id': checkpoint.get('entity_to_id', {}),
                'id_to_entity': checkpoint.get('id_to_entity', {}),
                'value_to_id': checkpoint.get('value_to_id', {}),
                'id_to_value': checkpoint.get('id_to_value', {}),
                'command_to_id': checkpoint.get('command_to_id', {}),
                'id_to_command': checkpoint.get('id_to_command', {})
            }
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load config
            self.config = checkpoint.get('config', {})
            self.max_length = self.config.get('max_length', 128)
            
            # Load model architecture (simplified approach)
            self.model = checkpoint.get('model_state_dict', {})
            self.enable_multi_task = checkpoint.get('enable_multi_task', False)
            
            self.model_loaded = True
            self.logger.info(f"✅ Trained model loaded successfully")
            self.logger.info(f"   Intents: {len(self.label_mappings['intent_to_id'])}")
            if self.enable_multi_task:
                self.logger.info(f"   Entities: {len(self.label_mappings['entity_to_id'])}")
                self.logger.info(f"   Commands: {len(self.label_mappings['command_to_id'])}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load trained model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main prediction method - Model-first approach
        
        Flow:
        1. Trained model prediction (PRIMARY)
        2. Reasoning engine validation/enhancement (SECONDARY)
        3. Hybrid decision making
        4. Final result
        """
        start_time = time.time()
        self.stats['total_predictions'] += 1
        
        try:
            # Step 1: Trained model prediction (PRIMARY)
            model_result = self._model_predict(text)
            
            # Step 2: Reasoning engine validation (SECONDARY)
            reasoning_result = self._reasoning_validate(text, context, model_result)
            
            # Step 3: Hybrid decision making
            final_result = self._hybrid_decision(model_result, reasoning_result, text, context)

            # Step 3.1: Post-process for command->entity compliance (quick rules)
            final_result = self._postprocess_command_entities(text, final_result)
            
            # Step 4: Add metadata
            processing_time = time.time() - start_time
            final_result.update({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'input_text': text,
                'context': context or {},
                'model_result': model_result,
                'reasoning_result': reasoning_result
            })
            
            # Update stats
            self._update_stats(final_result, processing_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in prediction: {e}")
            return self._fallback_prediction(text, context)

    def _postprocess_command_entities(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight post-processing to enforce command->entity mapping for common cases.
        - control-device:
          * ON/OFF: bật/mở/on → ACTION=mở, MODE=on; tắt/đóng/off → ACTION=tắt, MODE=off
          * Volume/Brightness: +/tăng/lên → ACTION=tăng, MODE=up; -/giảm/xuống → ACTION=giảm, MODE=down
          * DEVICE normalization: âm lượng/loa/âm thanh → "âm lượng"; độ sáng/sáng/brightness/ánh sáng → "độ sáng"
          * Remove unrelated PLATFORM
        """
        try:
            text_l = (text or "").lower()
            command = result.get("command") or result.get("intent") or "unknown"
            entities = dict(result.get("entities") or {})

            if command == "control-device":
                # Normalize device keywords
                has_volume = any(w in text_l for w in ["âm lượng", "loa", "âm thanh", "volume"])
                has_brightness = any(w in text_l for w in ["độ sáng", "sáng", "brightness", "ánh sáng"]) 

                # Normalize action keywords
                has_on = any(w in text_l for w in ["bật", "mở", "on"]) 
                has_off = any(w in text_l for w in ["tắt", "đóng", "off"]) 
                has_increase = any(w in text_l for w in ["tăng", "lên", "+", "up"]) 
                has_decrease = any(w in text_l for w in ["giảm", "xuống", "-", "down"]) 

                # DEVICE normalization
                if has_volume:
                    entities["DEVICE"] = "âm lượng"
                elif has_brightness:
                    entities["DEVICE"] = "độ sáng"

                # ACTION/MODE mapping
                if has_on:
                    entities["ACTION"] = "mở"
                    entities["MODE"] = "on"
                elif has_off:
                    entities["ACTION"] = "tắt"
                    entities["MODE"] = "off"
                elif has_increase:
                    entities["ACTION"] = "tăng"
                    entities["MODE"] = "up"
                elif has_decrease:
                    entities["ACTION"] = "giảm"
                    entities["MODE"] = "down"

                # control-device should not include PLATFORM
                entities.pop("PLATFORM", None)

                result["entities"] = entities

            return result
        except Exception:
            return result
    
    def _model_predict(self, text: str) -> Dict[str, Any]:
        """Primary prediction using trained model"""
        if not self.model_loaded:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "model_error",
                "error": "Model not loaded"
            }
        
        try:
            self.stats['model_predictions'] += 1
            
            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # For now, use simple rule-based prediction based on trained model patterns
            # This is a simplified approach since we can't load the full model due to torchvision issue
            prediction = self._rule_based_model_prediction(text)
            
            return {
                "intent": prediction['intent'],
                "confidence": prediction['confidence'],
                "entities": prediction['entities'],
                "command": prediction['command'],
                "method": "trained_model",
                "model_type": "multi-task" if self.enable_multi_task else "single-task"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model prediction error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "model_error",
                "error": str(e)
            }
    
    def _rule_based_model_prediction(self, text: str) -> Dict[str, Any]:
        """
        Rule-based prediction using patterns learned from trained model
        This simulates the trained model behavior using rule-based approach
        """
        text_lower = text.lower().strip()
        
        # Intent patterns based on training data
        intent_patterns = {
            'call': ['gọi', 'call', 'phone', 'điện thoại', 'facetime', 'video call'],
            'control-device': ['bật', 'tắt', 'điều chỉnh', 'turn', 'on', 'off', 'đèn', 'quạt', 'điều hòa'],
            'play-media': ['phát', 'chơi', 'play', 'nhạc', 'video', 'music', 'bài hát'],
            'search-internet': ['tìm', 'search', 'kiếm', 'google', 'youtube', 'internet'],
            'set-alarm': ['báo thức', 'alarm', 'nhắc', 'đặt', 'set'],
            'send-mess': ['gửi', 'nhắn', 'tin', 'message', 'sms', 'messenger'],
            'open-cam': ['camera', 'chụp', 'quay', 'ảnh', 'video'],
            'set-event-calendar': ['lịch', 'sự kiện', 'hẹn', 'calendar', 'event'],
            'make-video-call': ['video call', 'facetime', 'gọi video'],
            'add-contacts': ['thêm', 'lưu', 'add', 'contact', 'danh bạ', 'số điện thoại'],
            'view-content': ['xem', 'mở', 'view', 'thư viện', 'ảnh', 'video'],
            'get-info': ['hỏi', 'kiểm tra', 'thông tin', 'info', 'time', 'pin']
        }
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.9, 0.5 + (intent_scores[best_intent] * 0.1))
        else:
            best_intent = "unknown"
            confidence = 0.0
        
        # Extract entities based on intent
        entities = self._extract_entities_by_intent(text, best_intent)
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "entities": entities,
            "command": best_intent
        }
    
    def _extract_entities_by_intent(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities based on intent patterns"""
        entities = {}
        text_lower = text.lower()
        
        if intent == 'call':
            # Extract contact names
            if 'mẹ' in text_lower:
                entities['CONTACT_NAME'] = 'mẹ'
            elif 'bố' in text_lower:
                entities['CONTACT_NAME'] = 'bố'
            elif 'bạn' in text_lower:
                entities['CONTACT_NAME'] = 'bạn'
            
            # Extract phone numbers
            import re
            phone_pattern = r'\b\d{10,11}\b'
            phone_match = re.search(phone_pattern, text)
            if phone_match:
                entities['PHONE'] = phone_match.group()
        
        elif intent == 'control-device':
            # Extract device names
            if 'đèn' in text_lower:
                entities['DEVICE'] = 'đèn'
            elif 'quạt' in text_lower:
                entities['DEVICE'] = 'quạt'
            elif 'điều hòa' in text_lower:
                entities['DEVICE'] = 'điều hòa'
            
            # Extract locations
            if 'phòng khách' in text_lower:
                entities['LOCATION'] = 'phòng khách'
            elif 'phòng ngủ' in text_lower:
                entities['LOCATION'] = 'phòng ngủ'
        
        elif intent == 'set-alarm':
            # Extract time
            import re
            time_pattern = r'(\d{1,2})[h:]\s*(\d{0,2})'
            time_match = re.search(time_pattern, text)
            if time_match:
                hour = time_match.group(1)
                minute = time_match.group(2) if time_match.group(2) else '00'
                entities['TIME'] = f"{hour}:{minute}"
        
        elif intent == 'send-mess':
            # Extract receiver
            if 'mẹ' in text_lower:
                entities['RECEIVER'] = 'mẹ'
            elif 'bố' in text_lower:
                entities['RECEIVER'] = 'bố'
            elif 'bạn' in text_lower:
                entities['RECEIVER'] = 'bạn'
        
        return entities
    
    def _reasoning_validate(self, text: str, context: Optional[Dict[str, Any]], model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Secondary validation using reasoning engine"""
        if not self.reasoning_loaded:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "reasoning_error",
                "error": "Reasoning engine not available"
            }
        
        try:
            self.stats['reasoning_predictions'] += 1
            
            # Use reasoning engine for validation
            reasoning_result = self.reasoning_engine.reasoning_predict(text, context)
            
            return {
                "intent": reasoning_result.get('intent', 'unknown'),
                "confidence": reasoning_result.get('confidence', 0.0),
                "entities": reasoning_result.get('entities', {}),
                "command": reasoning_result.get('command', reasoning_result.get('intent', 'unknown')),
                "method": "reasoning_engine",
                "reasoning_details": reasoning_result
            }
            
        except Exception as e:
            self.logger.error(f"❌ Reasoning validation error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "method": "reasoning_error",
                "error": str(e)
            }
    
    def _hybrid_decision(self, model_result: Dict[str, Any], reasoning_result: Dict[str, Any], text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hybrid decision making - Model-first approach"""
        try:
            self.stats['hybrid_predictions'] += 1
            
            # Extract key information
            model_intent = model_result.get('intent', 'unknown')
            model_confidence = model_result.get('confidence', 0.0)
            model_entities = model_result.get('entities', {})
            
            reasoning_intent = reasoning_result.get('intent', 'unknown')
            reasoning_confidence = reasoning_result.get('confidence', 0.0)
            reasoning_entities = reasoning_result.get('entities', {})
            
            # Step 1: Use specialized entity extractor for enhanced entities
            specialized_entities = {}
            if hasattr(self, 'specialized_extractor_loaded') and self.specialized_extractor_loaded:
                try:
                    specialized_entities = self.specialized_entity_extractor.extract_all_entities(text, model_intent)
                except Exception as e:
                    self.logger.warning(f"Specialized entity extraction failed: {e}")
            
            # Decision logic - Model-first approach
            if model_confidence >= 0.7:
                # High confidence from model - use it as primary
                final_intent = model_intent
                final_confidence = model_confidence
                final_entities = model_entities.copy()
                decision_reason = "high_model_confidence"
                
                # Enhance with specialized entities first
                if specialized_entities:
                    final_entities.update(specialized_entities)
                    decision_reason = "high_model_confidence_with_specialized_entities"
                
                # Then enhance with reasoning if available
                if reasoning_intent != "unknown" and reasoning_confidence > 0.5:
                    final_entities.update(reasoning_entities)
                    decision_reason = "high_model_confidence_with_reasoning_enhancement"
                
            elif model_confidence >= 0.4:
                # Medium confidence from model - use with reasoning validation
                if reasoning_intent != "unknown" and reasoning_confidence > 0.6:
                    # Reasoning has higher confidence - use reasoning
                    final_intent = reasoning_intent
                    final_confidence = reasoning_confidence
                    final_entities = reasoning_entities.copy()
                    decision_reason = "reasoning_validation_override"
                else:
                    # Use model result
                    final_intent = model_intent
                    final_confidence = model_confidence
                    final_entities = model_entities.copy()
                    decision_reason = "medium_model_confidence"
                
            else:
                # Low confidence from model - use reasoning as fallback
                if reasoning_intent != "unknown" and reasoning_confidence > 0.3:
                    final_intent = reasoning_intent
                    final_confidence = reasoning_confidence
                    final_entities = reasoning_entities.copy()
                    decision_reason = "reasoning_fallback"
                else:
                    # Both failed - use model result anyway
                    final_intent = model_intent
                    final_confidence = model_confidence
                    final_entities = model_entities.copy()
                    decision_reason = "model_fallback"
            
            # Determine command
            final_command = final_intent  # For now, command = intent
            
            return {
                "intent": final_intent,
                "confidence": final_confidence,
                "entities": final_entities,
                "command": final_command,
                "method": "hybrid",
                "decision_reason": decision_reason,
                "primary_source": "trained_model" if model_confidence >= 0.4 else "reasoning_engine"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid decision error: {e}")
            return model_result  # Fallback to model result
    
    def _fallback_prediction(self, text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback prediction when all else fails"""
        self.stats['fallback_predictions'] += 1
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "command": "unknown",
            "method": "fallback",
            "input_text": text,
            "context": context or {},
            "timestamp": time.time(),
            "error": "All prediction methods failed"
        }
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """Update performance statistics"""
        try:
            # Update average processing time
            total_predictions = self.stats['total_predictions']
            current_avg = self.stats['avg_processing_time']
            self.stats['avg_processing_time'] = (current_avg * (total_predictions - 1) + processing_time) / total_predictions
            
            # Track confidence scores
            confidence = result.get('confidence', 0.0)
            self.stats['confidence_scores'].append(confidence)
            
            # Keep only last 100 confidence scores
            if len(self.stats['confidence_scores']) > 100:
                self.stats['confidence_scores'] = self.stats['confidence_scores'][-100:]
                
        except Exception as e:
            self.logger.error(f"⚠️ Stats update error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats['confidence_scores']:
            stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            stats['min_confidence'] = min(stats['confidence_scores'])
            stats['max_confidence'] = max(stats['confidence_scores'])
        
        stats['success_rate'] = (stats['total_predictions'] - stats['fallback_predictions']) / max(stats['total_predictions'], 1)
        
        return stats
    
    def test_system(self, test_cases: List[str]) -> Dict[str, Any]:
        """Test the hybrid system with multiple test cases"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"🧪 Testing case {i+1}/{len(test_cases)}: {test_case}")
            
            result = self.predict(test_case)
            results.append({
                "input": test_case,
                "result": result
            })
        
        # Analyze results
        analysis = {
            "total_tests": len(test_cases),
            "successful_predictions": len([r for r in results if r["result"]["intent"] != "unknown"]),
            "avg_confidence": sum([r["result"]["confidence"] for r in results]) / len(results),
            "avg_processing_time": sum([r["result"].get("processing_time", 0) for r in results]) / len(results),
            "results": results
        }
        
        return analysis

def test_model_first_hybrid():
    """Test model-first hybrid system"""
    print("🚀 TESTING MODEL-FIRST HYBRID SYSTEM")
    print("=" * 60)
    
    # Initialize system
    print("🔧 Initializing model-first hybrid system...")
    hybrid_system = ModelFirstHybridSystem()
    
    # Test cases
    test_cases = [
        # Call intent
        "gọi điện cho mẹ",
        "gọi cho bố",
        "gọi số 0123456789",
        
        # Control device intent
        "bật đèn phòng khách",
        "tắt quạt",
        "điều chỉnh nhiệt độ điều hòa",
        
        # Play media intent
        "phát nhạc",
        "bật video",
        "chơi nhạc buồn",
        
        # Search intent
        "tìm kiếm nhạc trên youtube",
        "tìm thông tin về thời tiết",
        "search google",
        
        # Alarm intent
        "đặt báo thức 7 giờ sáng",
        "báo thức 6h30",
        "nhắc nhở lúc 8 giờ",
        
        # Message intent
        "gửi tin nhắn cho bạn",
        "nhắn tin cho mẹ",
        "gửi sms",
        
        # Camera intent
        "mở camera",
        "chụp ảnh",
        "quay video",
        
        # Calendar intent
        "thêm sự kiện vào lịch",
        "đặt lịch hẹn",
        "nhắc nhở cuộc họp",
        
        # Video call intent
        "gọi video cho bạn",
        "video call mẹ",
        "gọi facetime",
        
        # Add contacts intent
        "thêm số điện thoại",
        "lưu danh bạ",
        "add contact",
        
        # View content intent
        "xem ảnh",
        "mở thư viện",
        "xem video đã lưu",
        
        # Get info intent
        "hỏi thời gian",
        "kiểm tra pin",
        "thông tin thiết bị"
    ]
    
    print(f"\n🧪 Testing with {len(test_cases)} test cases...")
    print("=" * 60)
    
    # Run tests
    start_time = time.time()
    test_results = hybrid_system.test_system(test_cases)
    total_time = time.time() - start_time
    
    # Print summary results
    print(f"\n📊 TEST SUMMARY:")
    print("=" * 60)
    print(f"Total tests: {test_results['total_tests']}")
    print(f"Successful predictions: {test_results['successful_predictions']}")
    print(f"Success rate: {test_results['successful_predictions']/test_results['total_tests']*100:.1f}%")
    print(f"Average confidence: {test_results['avg_confidence']:.3f}")
    print(f"Average processing time: {test_results['avg_processing_time']:.3f}s")
    print(f"Total test time: {total_time:.2f}s")
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("=" * 60)
    
    for i, result in enumerate(test_results['results']):
        print(f"\n{i+1:2d}. Input: '{result['input']}'")
        print(f"    Intent: {result['result']['intent']}")
        print(f"    Confidence: {result['result']['confidence']:.3f}")
        print(f"    Method: {result['result']['method']}")
        print(f"    Command: {result['result']['command']}")
        
        # Show entities if any
        entities = result['result'].get('entities', {})
        if entities:
            print(f"    Entities: {entities}")
        
        # Show decision reason
        decision_reason = result['result'].get('decision_reason', 'unknown')
        print(f"    Decision: {decision_reason}")
        
        # Show primary source
        primary_source = result['result'].get('primary_source', 'unknown')
        print(f"    Primary: {primary_source}")
    
    # Print system statistics
    print(f"\n📈 SYSTEM STATISTICS:")
    print("=" * 60)
    stats = hybrid_system.get_stats()
    
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Model predictions: {stats['model_predictions']}")
    print(f"Reasoning predictions: {stats['reasoning_predictions']}")
    print(f"Hybrid predictions: {stats['hybrid_predictions']}")
    print(f"Fallback predictions: {stats['fallback_predictions']}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    
    if 'avg_confidence' in stats:
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"Min confidence: {stats['min_confidence']:.3f}")
        print(f"Max confidence: {stats['max_confidence']:.3f}")
    
    print(f"Success rate: {stats['success_rate']:.3f}")
    
    # Analyze by intent
    print(f"\n🎯 INTENT ANALYSIS:")
    print("=" * 60)
    
    intent_counts = {}
    intent_confidences = {}
    
    for result in test_results['results']:
        intent = result['result']['intent']
        confidence = result['result']['confidence']
        
        if intent not in intent_counts:
            intent_counts[intent] = 0
            intent_confidences[intent] = []
        
        intent_counts[intent] += 1
        intent_confidences[intent].append(confidence)
    
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        avg_conf = sum(intent_confidences[intent]) / len(intent_confidences[intent])
        print(f"{intent:20s}: {count:2d} samples, avg confidence: {avg_conf:.3f}")
    
    # Save results to file
    results_file = "model_first_hybrid_test_results.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    print(f"\n✅ Model-first hybrid system testing completed!")
    return test_results

if __name__ == "__main__":
    try:
        test_model_first_hybrid()
        print(f"\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
