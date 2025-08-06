import torch
import torch.nn as nn
from transformers import AutoModel
from config import model_config, intent_config, entity_config, command_config

class UnifiedModel(nn.Module):
    """Mô hình thống nhất kết hợp cả 3 tác vụ: Intent, Entity, Command"""
    
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_config.model_name)
        self.dropout = nn.Dropout(model_config.dropout)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.phobert.config.hidden_size, intent_config.num_intents)
        
        # Entity classification head
        self.entity_classifier = nn.Linear(self.phobert.config.hidden_size, entity_config.num_entities)
        
        # Command classification head
        self.command_classifier = nn.Linear(self.phobert.config.hidden_size, command_config.num_commands)
        
    def forward(self, input_ids, attention_mask):
        # Lấy output từ PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooled output cho intent và command
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        intent_logits = self.intent_classifier(pooled_output)
        command_logits = self.command_classifier(pooled_output)
        
        # Sequence output cho entity
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        entity_logits = self.entity_classifier(sequence_output)
        
        return intent_logits, entity_logits, command_logits

class JointModel(nn.Module):
    """Mô hình kết hợp Intent Recognition và Entity Extraction"""
    
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_config.model_name)
        self.dropout = nn.Dropout(model_config.dropout)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.phobert.config.hidden_size, intent_config.num_intents)
        
        # Entity classification head
        self.entity_classifier = nn.Linear(self.phobert.config.hidden_size, entity_config.num_entities)
        
    def forward(self, input_ids, attention_mask):
        # Lấy output từ PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooled output cho intent
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        
        # Sequence output cho entity
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        entity_logits = self.entity_classifier(sequence_output)
        
        return intent_logits, entity_logits 