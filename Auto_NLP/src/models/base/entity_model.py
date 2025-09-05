import torch
import torch.nn as nn
from transformers import AutoModel
from src.training.configs.config import model_config, entity_config

class EntityExtractionModel(nn.Module):
    """Mô hình PhoBERT cho Entity Extraction"""
    
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_config.model_name)
        self.dropout = nn.Dropout(model_config.dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, entity_config.num_entities)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)
        
        return logits 