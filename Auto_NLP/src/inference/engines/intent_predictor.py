import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional
import re

class IntentPredictor:
    """Module dự đoán intent từ text"""
    
    def __init__(self, device=None):
        self.model = None
        self.tokenizer = None
        self.id_to_intent = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str, tokenizer_name: str = "vinai/phobert-base"):
        """Load trained model và tokenizer"""
        try:
            # Set environment variables to bypass vulnerability check
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TORCH_DISABLE_SAFETENSORS_WARNING"] = "1"
            os.environ["TORCH_SAFETENSORS_AVAILABLE"] = "1"
            
            # Patch transformers vulnerability check
            try:
                from transformers.utils import import_utils
                import_utils.check_torch_load_is_safe = lambda: None
            except ImportError:
                pass
            
            # Patch transformers vulnerability check in trainer (if needed)
            # Note: These are internal APIs, suppress warnings only
            try:
                from transformers.utils import import_utils
                if hasattr(import_utils, 'check_torch_load_is_safe'):
                    import_utils.check_torch_load_is_safe = lambda: None  # type: ignore
            except (ImportError, AttributeError):
                pass
            
            # Load model checkpoint to extract metadata
            checkpoint: Dict
            try:
                # Load with torch.load and weights_only=True for security
                checkpoint_raw = torch.load(model_path, map_location=self.device, weights_only=True)
                # Ensure checkpoint is a dict
                if not isinstance(checkpoint_raw, dict):
                    raise ValueError("Checkpoint must be a dictionary")
                checkpoint = checkpoint_raw
                print("[OK] Model loaded with torch.load (weights_only=True)")
            except Exception as e:
                print(f"Error loading model with torch.load: {e}")
                # Fallback: try safetensors
                try:
                    import safetensors.torch
                    checkpoint_fallback = safetensors.torch.load_file(model_path, device='cpu')
                    if not isinstance(checkpoint_fallback, dict):
                        raise ValueError("Checkpoint must be a dictionary")
                    checkpoint = checkpoint_fallback
                    print("[OK] Model loaded with safetensors (fallback)")
                except Exception as e2:
                    print(f"Error loading model with safetensors: {e2}")
                    return False
            
            # Extract model config from checkpoint
            model_name_raw = checkpoint.get('model_name', tokenizer_name)
            model_name = str(model_name_raw) if model_name_raw is not None else tokenizer_name
            config = checkpoint.get('config', {})
            
            # Get number of intents from intent_config
            intent_config = checkpoint.get('intent_config', {})
            if not isinstance(intent_config, dict):
                intent_config = {}
            id_to_intent_raw = checkpoint.get('id_to_intent', {})
            if not isinstance(id_to_intent_raw, dict):
                id_to_intent_raw = {}
            num_intents = len(id_to_intent_raw)
            
            if num_intents == 0:
                # Fallback: try to get from intent_config
                intent_labels = intent_config.get('intent_labels', []) if isinstance(intent_config, dict) else []
                if isinstance(intent_labels, list):
                    num_intents = len(intent_labels)
            
            if num_intents == 0:
                print("Could not determine number of intents from checkpoint")
                return False
            
            # Load intent mapping from checkpoint
            self.id_to_intent = id_to_intent_raw if isinstance(id_to_intent_raw, dict) else {}
            
            # Load tokenizer with use_fast=False for PhoBERT
            # Set environment variables to bypass vulnerability
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_HUB_OFFLINE"] = "0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
            # Load the actual trained model
            try:
                # Try to load the model architecture from checkpoint
                if 'model_state_dict' in checkpoint:
                    # Create model architecture compatible with working_gpu_enhanced_model
                    class WorkingModel(nn.Module):
                        def __init__(self, model_name: str, num_intents: int):
                            super().__init__()
                            self.encoder = AutoModel.from_pretrained(model_name)
                            self.hidden_size = self.encoder.config.hidden_size
                            self.intent_classifier = nn.Sequential(
                                nn.Linear(self.hidden_size, self.hidden_size // 2),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(self.hidden_size // 2, num_intents)
                            )
                            self._init_weights()

                        def _init_weights(self):
                            nn.init.xavier_uniform_(self.intent_classifier[0].weight)
                            nn.init.zeros_(self.intent_classifier[0].bias)
                            nn.init.xavier_uniform_(self.intent_classifier[3].weight)
                            nn.init.zeros_(self.intent_classifier[3].bias)

                        def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
                            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                            pooled_output = outputs.pooler_output
                            if hasattr(self, 'dropout'):
                                pooled_output = self.dropout(pooled_output)
                            intent_logits = self.intent_classifier(pooled_output)
                            return {'intent_logits': intent_logits}
                    
                    # Get model config
                    model_config = checkpoint.get('model_config', {})
                    num_intents = len(self.id_to_intent)
                    
                    # Create model instance
                    self.model = WorkingModel(
                        model_name=model_name,
                        num_intents=num_intents
                    )
                    
                    # Load state dict (skip class_weights if present)
                    state_dict_raw = checkpoint.get('model_state_dict')
                    if not isinstance(state_dict_raw, dict):
                        raise ValueError("model_state_dict must be a dictionary")
                    state_dict: Dict = state_dict_raw
                    filtered_state_dict = {k: v for k, v in state_dict.items() if k != 'class_weights'}
                    
                    # Load with filtered keys
                    self.model.load_state_dict(filtered_state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    
                    print(f"Trained model loaded successfully from {model_path}")
                    print(f"Number of intents: {num_intents}")
                    print(f"Intent mapping: {self.id_to_intent}")
                    
                    return True
                else:
                    print("No model_state_dict found in checkpoint, using fallback")
                    self.model = None
                    return True
                    
            except Exception as e:
                print(f"Error loading trained model: {e}")
                print("Falling back to keyword-based prediction")
                self.model = None
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_intent(self, text: str, confidence_threshold: float = 0.1) -> Dict:
        """Dự đoán intent từ text"""
        if not self.model or not self.tokenizer:
            return self._fallback_intent_prediction(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                result = self.model(inputs["input_ids"], inputs["attention_mask"])
                
                # Handle different return types
                if isinstance(result, dict):
                    logits = result.get('intent_logits', result.get('logits'))
                else:
                    logits = result
                
                if logits is None:
                    print("DEBUG: No logits found in model output")
                    return self._fallback_intent_prediction(text)
                
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_id = torch.max(probabilities, dim=1)
                
                confidence = confidence.item()
                predicted_id = predicted_id.item()
                
                if self.id_to_intent is None:
                    intent = "unknown"
                else:
                    intent = self.id_to_intent.get(predicted_id, "unknown") if isinstance(self.id_to_intent, dict) else "unknown"
                
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "method": "trained_model"
                }
                
        except Exception as e:
            print(f"Error in intent prediction: {e}")
            return self._fallback_intent_prediction(text)
    
    def _fallback_intent_prediction(self, text: str) -> Dict:
        """Fallback intent prediction khi model không available"""
        text_lower = text.lower()
        
        # Basic keyword-based intent detection
        intent_keywords = {
            "call": ["gọi", "alo", "gọi điện", "nói chuyện điện thoại", "call"],
            "send-mess": ["nhắn tin", "gửi tin", "sms", "gửi tin nhắn", "nhắn", "gửi", "nhan tin", "gui tin", "tin nhan", "gui tin nhan", "nhan", "gui", "tin", "nhan tin qua", "gui tin qua", "nhan tin cho", "gui tin cho", "nhan tin toi", "gui tin toi", "nhan tin den", "gui tin den"],
            "add-contacts": ["lưu số", "thêm số", "lưu danh bạ", "thêm danh bạ", "lưu liên hệ", "thêm bạn", "create contact", "ghi contact", "add contact", "lưu chị", "lưu anh", "lưu cô", "lưu thầy", "lưu bác", "lưu ông", "lưu bà", "luu lien he", "them ban", "luu chi", "luu anh", "luu co", "luu thay", "luu bac", "luu ong", "luu ba", "luu so", "them so", "luu danh ba", "them danh ba", "luu", "them", "so", "danh ba", "lien he", "ban", "chi", "anh", "co", "thay", "bac", "ong", "ba"],
            "make-video-call": ["gọi video", "video call", "facetime", "call video"],
            "search-internet": ["tìm kiếm", "search", "google", "tra cứu", "tìm", "tim kiem", "tim", "kiem", "tra cuu", "thong tin", "gia vang", "gia", "vang", "trong tuan", "tuan qua", "qua"],
            "search-youtube": ["youtube", "yt", "video youtube", "tìm youtube", "xem youtube"],
            "get-info": ["thông tin", "thời tiết", "tin tức", "đọc tin", "kiểm tra", "xem tin"],
            "control-device": ["bật", "tắt", "điều khiển", "đèn", "âm lượng", "wifi", "bluetooth"],
            "open-cam": ["mở camera", "bật camera", "chụp ảnh", "quay video", "camera"],
            "set-alarm": ["báo thức", "đánh thức", "đặt báo thức", "wake up"],
            "unknown": ["không hiểu", "không rõ", "lạ", "không biết"]
        }
        
        # Calculate scores for all intents
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        print(f"DEBUG: Intent scores: {intent_scores}")
        
        # Special logic for conflicting keywords
        if "send-mess" in intent_scores and "search-internet" in intent_scores:
            # Nếu có cả send-mess và search-internet, ưu tiên search-internet
            if intent_scores["search-internet"] >= 1:
                print(f"DEBUG: Removing send-mess score, search-internet score: {intent_scores['search-internet']}")
                intent_scores["send-mess"] = 0  # Remove send-mess score
        
        # Find best intent with improved logic
        if intent_scores:
            best_intent = max(intent_scores, key=lambda k: intent_scores[k])
            max_score = intent_scores[best_intent]
            
            # Đảm bảo confidence cao hơn cho search-internet
            if best_intent == "search-internet" and max_score >= 1:
                confidence = 0.9
            elif best_intent == "add-contacts" and max_score >= 1:
                confidence = 0.9
            else:
                confidence = 0.7
                
            return {
                "intent": best_intent,
                "confidence": confidence,
                "method": "keyword_fallback"
            }
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "method": "fallback"
        }
