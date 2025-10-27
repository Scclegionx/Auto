#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loading và Inference Scripts
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
import logging

class ModelLoader:
    """Class để load trained models"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def load_tokenizer(self, tokenizer_dir: str) -> AutoTokenizer:
        """Load tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            self.logger.info(f"✅ Loaded tokenizer from {tokenizer_dir}")
            return tokenizer
        except Exception as e:
            self.logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
    
    def load_model_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"✅ Loaded config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            raise

class IntentInference:
    """Intent classification inference"""
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str):
        self.loader = ModelLoader(model_path)
        self.checkpoint = self.loader.load_checkpoint(model_path)
        self.tokenizer = self.loader.load_tokenizer(tokenizer_path)
        self.config = self.loader.load_model_config(config_path)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load mappings
        self.intent_to_id = self.checkpoint.get('intent_to_id', {})
        self.id_to_intent = self.checkpoint.get('id_to_intent', {})
        
    def _load_model(self):
        """Load model architecture"""
        from models.base.optimized_intent_model import OptimizedIntentModel
        
        model = OptimizedIntentModel(
            model_name=self.config['model_name'],
            num_intents=self.config['num_intents'],
            config=self.config
        )
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.loader.device)
        
        return model
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent for given text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 128),
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.loader.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)
            predicted_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        # Get intent name
        intent_name = self.id_to_intent.get(predicted_id, f"unknown_{predicted_id}")
        
        return {
            'intent': intent_name,
            'confidence': confidence,
            'intent_id': predicted_id,
            'probabilities': {
                self.id_to_intent.get(i, f"unknown_{i}"): prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict intents for batch of texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

class MultiTaskInference:
    """Multi-task inference (Intent + Entity + Command)"""
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str):
        self.loader = ModelLoader(model_path)
        self.checkpoint = self.loader.load_checkpoint(model_path)
        self.tokenizer = self.loader.load_tokenizer(tokenizer_path)
        self.config = self.loader.load_model_config(config_path)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load mappings
        self.intent_to_id = self.checkpoint.get('intent_to_id', {})
        self.id_to_intent = self.checkpoint.get('id_to_intent', {})
        self.entity_to_id = self.checkpoint.get('entity_to_id', {})
        self.id_to_entity = self.checkpoint.get('id_to_entity', {})
        self.command_to_id = self.checkpoint.get('command_to_id', {})
        self.id_to_command = self.checkpoint.get('id_to_command', {})
        
    def _load_model(self):
        """Load multi-task model"""
        # Import từ training script
        import sys
        sys.path.append('src/training/scripts')
        from train_gpu import OptimizedIntentModel
        
        model = OptimizedIntentModel(
            model_name=self.config['model_name'],
            num_intents=self.config['num_intents'],
            config=self.config,
            num_entity_labels=len(self.entity_to_id),
            num_value_labels=len(self.checkpoint.get('value_to_id', {})),
            num_commands=len(self.command_to_id),
            enable_multi_task=True
        )
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.loader.device)
        
        return model
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent, entities, and command"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 128),
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.loader.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        # Process intent
        intent_logits = outputs['intent_logits']
        intent_probabilities = torch.softmax(intent_logits, dim=1)
        predicted_intent_id = torch.argmax(intent_probabilities, dim=1).item()
        intent_confidence = intent_probabilities[0][predicted_intent_id].item()
        
        # Process entities
        entities = []
        if 'entity_logits' in outputs:
            entity_logits = outputs['entity_logits']
            entity_predictions = self.model.entity_crf.decode(entity_logits, inputs['attention_mask'].float())
            
            # Convert to entity spans
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            entities = self._extract_entities(tokens, entity_predictions[0])
        
        # Process command
        command_name = "unknown"
        if 'command_logits' in outputs:
            command_logits = outputs['command_logits']
            command_probabilities = torch.softmax(command_logits, dim=1)
            predicted_command_id = torch.argmax(command_probabilities, dim=1).item()
            command_name = self.id_to_command.get(predicted_command_id, "unknown")
        
        return {
            'intent': self.id_to_intent.get(predicted_intent_id, f"unknown_{predicted_intent_id}"),
            'intent_confidence': intent_confidence,
            'command': command_name,
            'entities': entities,
            'text': text
        }
    
    def _extract_entities(self, tokens: List[str], predictions: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract entity spans from predictions"""
        entities = []
        current_entity = None
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            pred_label = self.id_to_entity.get(pred_id.item(), 'O')
            
            if pred_label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token.replace('▁', ''),
                    'label': pred_label[2:],
                    'start': i,
                    'end': i
                }
            elif pred_label.startswith('I-') and current_entity:
                # Continue entity
                current_entity['text'] += token.replace('▁', '')
                current_entity['end'] = i
            else:
                # End entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities

def load_trained_model(model_name: str = "phobert_large_intent_model") -> IntentInference:
    """Load trained model for inference"""
    model_dir = f"models/trained/{model_name}"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find model files
    model_files = list(Path(model_dir).glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Use best model if available, otherwise use first available
    best_model = None
    for model_file in model_files:
        if "best" in model_file.name:
            best_model = model_file
            break
    
    model_path = best_model or model_files[0]
    tokenizer_path = model_dir
    config_path = os.path.join(model_dir, "config.json")
    
    return IntentInference(str(model_path), tokenizer_path, config_path)

def load_multi_task_model(model_name: str = "phobert_large_intent_model") -> MultiTaskInference:
    """Load multi-task trained model for inference"""
    model_dir = f"models/trained/{model_name}"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find model files
    model_files = list(Path(model_dir).glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Use best model if available, otherwise use first available
    best_model = None
    for model_file in model_files:
        if "best" in model_file.name:
            best_model = model_file
            break
    
    model_path = best_model or model_files[0]
    tokenizer_path = model_dir
    config_path = os.path.join(model_dir, "config.json")
    
    return MultiTaskInference(str(model_path), tokenizer_path, config_path)

# Example usage
if __name__ == "__main__":
    # Load model
    model = load_trained_model()
    
    # Test prediction
    test_text = "gọi điện cho mẹ"
    result = model.predict(test_text)
    print(f"Text: {test_text}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.3f}")
