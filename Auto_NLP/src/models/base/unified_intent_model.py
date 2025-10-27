"""
Unified Intent Model for Training and Inference
Fixes architecture mismatch between training and inference
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional

class UnifiedIntentModel(nn.Module):
    """Unified model architecture for both training and inference"""
    
    def __init__(self, model_name: str, num_intents: int, config: Dict[str, Any] = None):
        super().__init__()
        self.model_name = model_name
        self.num_intents = num_intents
        self.config = config or {}
        
        # Load pre-trained PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Intent classification head - SIMPLIFIED
        self.intent_classifier = nn.Linear(self.hidden_size, num_intents)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.zeros_(self.intent_classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                intent_labels: torch.Tensor = None) -> Dict[str, Any]:
        """Forward pass for intent classification"""
        # Get encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        
        result = {
            'intent_logits': intent_logits
        }
        
        # Calculate loss if labels provided (for training)
        if intent_labels is not None:
            # Use weighted loss for class imbalance (24.38x imbalance ratio)
            class_weights = torch.tensor([
                1.0,    # send-mess (20.1%)
                1.0,    # call (17.7%)
                1.0,    # set-alarm (11.3%)
                1.0,    # set-event-calendar (11.3%)
                1.0,    # get-info (10.4%)
                2.0,    # add-contacts (6.4%) - 2x weight
                2.5,    # control-device (5.0%) - 2.5x weight
                2.5,    # make-video-call (4.8%) - 2.5x weight
                2.5,    # open-cam (4.8%) - 2.5x weight
                3.0,    # play-media (3.3%) - 3x weight
                3.0,    # search-internet (3.2%) - 3x weight
                10.0,   # view-content (0.9%) - 10x weight
                10.0    # search-youtube (0.8%) - 10x weight
            ], device=intent_logits.device)
            
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            intent_loss = loss_fct(intent_logits, intent_labels)
            result['loss'] = intent_loss
        
        return result
    
    def predict_intent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Predict intent with confidence"""
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            logits = result['intent_logits']
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_id = torch.max(probabilities, dim=1)
            
            return {
                "intent_id": predicted_id.item(),
                "confidence": confidence.item(),
                "probabilities": probabilities.cpu().numpy().tolist()
            }
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "num_intents": self.num_intents,
            "hidden_size": self.hidden_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "architecture": "UnifiedIntentModel"
        }
