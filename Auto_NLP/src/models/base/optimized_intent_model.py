"""
Optimized Intent Model for Vietnamese NLP
Tối ưu cho GPU 6GB và inference nhanh
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional

class OptimizedIntentModel(nn.Module):
    """Optimized Intent Model với tối ưu memory và speed"""
    
    def __init__(self, model_name: str, num_intents: int, config: Dict[str, Any] = None):
        super().__init__()
        self.model_name = model_name
        self.num_intents = num_intents
        self.config = config or {}
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.hidden_size, num_intents)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass cho intent classification"""
        # Get encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        
        return intent_logits
    
    def predict_intent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Predict intent với confidence"""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_id = torch.max(probabilities, dim=1)
            
            return {
                "intent_id": predicted_id.item(),
                "confidence": confidence.item(),
                "probabilities": probabilities.cpu().numpy().tolist()
            }
