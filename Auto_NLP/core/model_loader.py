#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trained Model Loader và Inference
Sử dụng trained model làm chính với API đơn giản
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.training.scripts.train_gpu import OptimizedIntentModel
    from src.training.configs.config import ModelConfig, IntentConfig, EntityConfig, ValueConfig, CommandConfig
    print("✅ Imported training components")
except ImportError as e:
    print(f"❌ Failed to import training components: {e}")
    sys.exit(1)

class TrainedModelInference:
    """
    Trained Model Inference Class
    Load và sử dụng trained model cho prediction
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.config = None
        self.label_mappings = None
        self.model_loaded = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained model và tokenizer"""
        try:
            # Find best model checkpoint
            best_model_path = self.model_path / "model_best.pth"
            if not best_model_path.exists():
                # Fallback to any epoch model
                epoch_models = sorted(self.model_path.glob("model_epoch_*.pth"), reverse=True)
                if epoch_models:
                    best_model_path = epoch_models[0]
                else:
                    raise FileNotFoundError(f"No trained model found in {self.model_path}")
            
            self.logger.info(f"Loading model from: {best_model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Load configs
            self.config = ModelConfig()
            intent_config = IntentConfig()
            entity_config = EntityConfig()
            value_config = ValueConfig()
            command_config = CommandConfig()
            
            # Update configs from checkpoint if available
            if 'config' in checkpoint:
                for k, v in checkpoint['config'].items():
                    setattr(self.config, k, v)
            
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
            
            # Initialize tokenizer
            tokenizer_dir = self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Initialize model architecture
            enable_multi_task = checkpoint.get('enable_multi_task', False)
            
            self.model = OptimizedIntentModel(
                self.config.model_name,
                num_intents=intent_config.num_intents,
                config=self.config,
                num_entity_labels=len(self.label_mappings['entity_to_id']) if enable_multi_task else None,
                num_value_labels=len(self.label_mappings['value_to_id']) if enable_multi_task else None,
                num_commands=len(self.label_mappings['command_to_id']) if enable_multi_task else None,
                enable_multi_task=enable_multi_task
            ).to(self.device)
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_loaded = True
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"   Model type: {'Multi-task' if enable_multi_task else 'Single-task'}")
            self.logger.info(f"   Intents: {len(self.label_mappings['intent_to_id'])}")
            if enable_multi_task:
                self.logger.info(f"   Entities: {len(self.label_mappings['entity_to_id'])}")
                self.logger.info(f"   Values: {len(self.label_mappings['value_to_id'])}")
                self.logger.info(f"   Commands: {len(self.label_mappings['command_to_id'])}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent và entities từ text
        
        Args:
            text: Input text
            
        Returns:
            Dict với intent, confidence, entities, command
        """
        if not self.model_loaded:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "value": "N/A",
                "error": "Model not loaded"
            }
        
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
            
            # Process outputs
            if isinstance(outputs, dict) and 'intent_logits' in outputs:
                # Multi-task model
                intent_logits = outputs['intent_logits']
                entity_logits = outputs.get('entity_logits')
                command_logits = outputs.get('command_logits')
                
                # Intent prediction
                intent_probs = torch.softmax(intent_logits, dim=-1)
                intent_conf, intent_id = torch.max(intent_probs, dim=-1)
                predicted_intent = self.label_mappings['id_to_intent'].get(intent_id.item(), "unknown")
                
                # Entity prediction
                predicted_entities = {}
                if entity_logits is not None:
                    try:
                        entity_predictions = self.model.entity_crf.decode(entity_logits, attention_mask)
                        predicted_entities = self._decode_entities(
                            entity_predictions.squeeze().tolist(), 
                            input_ids.squeeze().tolist()
                        )
                    except Exception as e:
                        self.logger.warning(f"Entity decoding failed: {e}")
                
                # Command prediction
                predicted_command = predicted_intent
                if command_logits is not None:
                    command_probs = torch.softmax(command_logits, dim=-1)
                    command_conf, command_id = torch.max(command_probs, dim=-1)
                    predicted_command = self.label_mappings['id_to_command'].get(command_id.item(), predicted_intent)
                
                return {
                    "intent": predicted_intent,
                    "confidence": intent_conf.item(),
                    "entities": predicted_entities,
                    "command": predicted_command,
                    "value": "N/A",  # Value extraction disabled
                    "model_type": "multi-task"
                }
            else:
                # Single-task model
                logits = outputs
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_id = torch.max(probabilities, dim=1)
                predicted_intent = self.label_mappings['id_to_intent'].get(predicted_id.item(), "unknown")
                
                return {
                    "intent": predicted_intent,
                    "confidence": confidence.item(),
                    "entities": {},
                    "command": predicted_intent,
                    "value": "N/A",
                    "model_type": "single-task"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Prediction error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "command": "unknown",
                "value": "N/A",
                "error": str(e)
            }
    
    def _decode_entities(self, entity_ids: List[int], input_ids: List[int]) -> Dict[str, str]:
        """Decode entity predictions"""
        try:
            # Convert token IDs back to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Filter out special tokens and map entity IDs to labels
            decoded_labels = []
            for i, token_id in enumerate(input_ids):
                if token_id not in self.tokenizer.all_special_ids and i < len(entity_ids):
                    label_id = entity_ids[i]
                    decoded_labels.append(self.label_mappings['id_to_entity'].get(label_id, "O"))
                else:
                    decoded_labels.append("O")
            
            # Reconstruct entities from BIO labels
            entities = {}
            current_entity_type = None
            current_entity_tokens = []
            
            for i, label in enumerate(decoded_labels):
                if label.startswith("B-"):
                    if current_entity_type and current_entity_tokens:
                        entity_text = self.tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
                        entities[current_entity_type] = entity_text
                    current_entity_type = label[2:]
                    current_entity_tokens = [input_ids[i]]
                elif label.startswith("I-") and current_entity_type == label[2:]:
                    current_entity_tokens.append(input_ids[i])
                else:
                    if current_entity_type and current_entity_tokens:
                        entity_text = self.tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
                        entities[current_entity_type] = entity_text
                    current_entity_type = None
                    current_entity_tokens = []
            
            if current_entity_type and current_entity_tokens:
                entity_text = self.tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
                entities[current_entity_type] = entity_text
            
            return entities
            
        except Exception as e:
            self.logger.error(f"❌ Entity decoding error: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "max_length": self.config.max_length if self.config else None,
            "intent_labels": list(self.label_mappings['intent_to_id'].keys()) if self.label_mappings else [],
            "entity_labels": list(self.label_mappings['entity_to_id'].keys()) if self.label_mappings else [],
            "command_labels": list(self.label_mappings['command_to_id'].keys()) if self.label_mappings else []
        }

def load_trained_model(model_name: str = "phobert_large_intent_model", device: Optional[torch.device] = None) -> TrainedModelInference:
    """
    Load trained model
    
    Args:
        model_name: Name of the model directory
        device: Device to load model on
        
    Returns:
        TrainedModelInference instance
    """
    model_path = Path("models") / model_name
    return TrainedModelInference(str(model_path), device)

# Test function
if __name__ == "__main__":
    print("🚀 Testing TrainedModelInference...")
    
    try:
        # Load model
        model = load_trained_model("phobert_large_intent_model")
        
        # Test cases
        test_cases = [
            "gọi điện cho mẹ",
            "bật đèn phòng khách",
            "tìm kiếm nhạc trên youtube",
            "đặt báo thức 7 giờ sáng",
            "gửi tin nhắn cho bạn"
        ]
        
        print(f"\n🧪 Testing with {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1}. Testing: '{test_case}'")
            result = model.predict(test_case)
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Command: {result['command']}")
            print(f"   Entities: {result['entities']}")
            print(f"   Model type: {result['model_type']}")
        
        # Print model info
        print(f"\n📊 Model Info:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ TrainedModelInference test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
