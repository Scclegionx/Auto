import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.training.configs.config import model_config, intent_config

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism cho intent recognition"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        output = self.output(context)
        return output

class IntentRecognitionModel(nn.Module):
    """Mô hình PhoBERT nâng cao cho Intent Recognition với nhiều lớp và attention"""
    
    def __init__(self, use_attention=True, use_crf=False, use_ensemble=False):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_config.model_name)
        self.hidden_size = self.phobert.config.hidden_size
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadAttention(self.hidden_size, num_heads=8, dropout=model_config.dropout)
            self.layer_norm1 = nn.LayerNorm(self.hidden_size)
            self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        self.classifier_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.Linear(self.hidden_size // 4, intent_config.num_intents)
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(model_config.dropout),
            nn.Dropout(model_config.dropout),
            nn.Dropout(model_config.dropout)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 4)
        ])
        
        self.confidence_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(intent_config.num_intents)
        
        self.use_ensemble = use_ensemble
        if use_ensemble:
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)  # 3 different classifiers
        
    def forward(self, input_ids, attention_mask, return_confidence=False):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        
        if self.use_attention:
            batch_size = pooled_output.size(0)
            pooled_output = pooled_output.unsqueeze(1)  # Add sequence dimension
            
            attended_output = self.attention(pooled_output, attention_mask)
            attended_output = self.layer_norm1(attended_output + pooled_output)  # Residual connection
            
            ff_output = F.relu(attended_output)
            ff_output = self.layer_norm2(ff_output + attended_output)  # Residual connection
            
            pooled_output = ff_output.squeeze(1)  # Remove sequence dimension
        
        x = pooled_output
        confidence_score = self.confidence_scorer(x)
        
        for i, (classifier, dropout, bn) in enumerate(zip(self.classifier_layers[:-1], 
                                                        self.dropout_layers[:-1], 
                                                        self.batch_norms)):
            residual = x
            x = classifier(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            if x.size(-1) == residual.size(-1):
                x = x + residual
        
        x = self.dropout_layers[-1](x)
        logits = self.classifier_layers[-1](x)
        
        if return_confidence:
            return logits, confidence_score
        
        return logits
    
    def get_intent_confidence(self, logits):
        """Tính confidence score cho intent prediction"""
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        return max_probs
    
    def predict_with_confidence(self, input_ids, attention_mask, confidence_threshold=0.7):
        """Predict intent với confidence threshold"""
        logits, confidence_score = self.forward(input_ids, attention_mask, return_confidence=True)
        intent_probs = F.softmax(logits, dim=-1)
        max_probs, predicted_intents = torch.max(intent_probs, dim=-1)
        
        final_confidence = (confidence_score.squeeze() + max_probs) / 2
        
        low_confidence_mask = final_confidence < confidence_threshold
        predicted_intents[low_confidence_mask] = -1  # Unknown intent
        
        return predicted_intents, final_confidence

class CRF(nn.Module):
    """Conditional Random Field layer cho intent sequence labeling"""
    
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def forward(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        
        score = self.start_transitions.view(1, -1) + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        
        score += self.end_transitions.view(1, -1)
        
        return score, history 