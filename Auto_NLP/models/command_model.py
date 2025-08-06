import torch
import torch.nn as nn
from transformers import AutoModel
from config import model_config, command_config

class CommandProcessingModel(nn.Module):
    """Mô hình PhoBERT cho Command Processing"""
    
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_config.model_name)
        self.dropout = nn.Dropout(model_config.dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, command_config.num_commands)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits 