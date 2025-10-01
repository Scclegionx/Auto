import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

# Set memory management for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Adafactor
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, f1_score
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score, precision_score as seqeval_precision_score, recall_score as seqeval_recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
import gc
import time
import traceback
import random
from datetime import datetime

# Import config
import sys
sys.path.append('.')
from src.training.configs.config import ModelConfig, IntentConfig, EntityConfig, ValueConfig, CommandConfig

# Set seed for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

class CRFLayer(nn.Module):
    """CRF Layer cho sequence labeling"""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
        
        # Initialize transitions
        nn.init.xavier_uniform_(self.transitions)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate CRF loss"""
        # logits: [batch_size, seq_len, num_labels]
        # labels: [batch_size, seq_len]
        # mask: [batch_size, seq_len]
        
        # Ensure dtype/device compatibility
        mask = mask.to(logits.device).float()
        labels = labels.to(logits.device).long()
        
        # Check for out-of-range labels
        if (labels >= self.num_labels).any() or (labels < 0).any():
            raise RuntimeError(f"CRF labels out of range: max={labels.max().item()}, num_labels={self.num_labels}")
        
        batch_size, seq_len, num_labels = logits.size()
        
        # Calculate forward scores
        forward_scores = self._forward_algorithm(logits, mask)
        
        # Calculate gold scores
        gold_scores = self._gold_score(logits, labels, mask)
        
        # CRF loss = forward_scores - gold_scores
        loss = forward_scores - gold_scores
        
        return loss.mean()
    
    def decode(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding"""
        # Ensure dtype/device compatibility
        mask = mask.to(logits.device).float()
        
        batch_size, seq_len, num_labels = logits.size()
        
        # Calculate actual sequence lengths
        lengths = mask.sum(1).long()
        
        # Initialize
        scores = logits[:, 0] + self.start_transitions.unsqueeze(0)
        paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        
        # Forward pass
        for t in range(1, seq_len):
            # scores shape: [batch_size, num_labels]
            # transitions shape: [num_labels, num_labels] 
            # logits[:, t] shape: [batch_size, num_labels]
            scores = scores.unsqueeze(-1) + self.transitions.unsqueeze(0) + logits[:, t].unsqueeze(1)
            # scores shape after unsqueeze: [batch_size, num_labels, num_labels]
            # We want max over the last dimension (num_labels)
            scores, max_indices = torch.max(scores, dim=-1)
            # max_indices should be [batch_size, num_labels] -> we want [batch_size]
            # Take the last column (final state indices)
            max_indices = max_indices[:, -1]  # Shape: [batch_size]
            paths[:, t] = max_indices
            scores = scores * mask[:, t].unsqueeze(-1) + (1 - mask[:, t].unsqueeze(-1)) * scores
        
        # Add end transitions before backtracking
        scores = scores + self.end_transitions.unsqueeze(0)
        
        # Backward pass - trace back the best path
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        best_paths[:, -1] = torch.argmax(scores, dim=-1)
        
        for t in range(seq_len - 2, -1, -1):
            # Simple approach: use the paths from forward pass directly
            # This is a simplified version that should work
            best_paths[:, t] = paths[:, t + 1]
        
        return best_paths
    
    def _forward_algorithm(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward algorithm for CRF"""
        # Ensure dtype/device compatibility
        mask = mask.to(logits.device).float()
        
        batch_size, seq_len, num_labels = logits.size()
        
        # Initialize
        scores = logits[:, 0] + self.start_transitions.unsqueeze(0)
        
        # Forward pass
        for t in range(1, seq_len):
            scores = scores.unsqueeze(-1) + self.transitions.unsqueeze(0) + logits[:, t].unsqueeze(1)
            scores = torch.logsumexp(scores, dim=1)
            scores = scores * mask[:, t].unsqueeze(-1) + (1 - mask[:, t].unsqueeze(-1)) * scores
        
        # Add end transitions
        scores = scores + self.end_transitions.unsqueeze(0)
        
        return torch.logsumexp(scores, dim=-1)
    
    def _gold_score(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate gold score for given labels"""
        # Ensure dtype/device compatibility
        mask = mask.to(logits.device).float()
        labels = labels.to(logits.device).long()
        
        batch_size, seq_len, num_labels = logits.size()
        
        # Calculate actual sequence lengths
        lengths = mask.sum(1).long()
        
        # Initialize
        scores = logits[:, 0].gather(1, labels[:, 0].unsqueeze(1)).squeeze(1) + self.start_transitions[labels[:, 0]]
        
        # Forward pass
        for t in range(1, seq_len):
            scores += self.transitions[labels[:, t-1], labels[:, t]] + logits[:, t].gather(1, labels[:, t].unsqueeze(1)).squeeze(1)
            scores = scores * mask[:, t] + (1 - mask[:, t]) * scores
        
        # Add end transitions using actual last positions
        last_tags = labels.gather(1, (lengths-1).unsqueeze(1)).squeeze(1)
        scores += self.end_transitions[last_tags]
        
        return scores

def custom_collate_fn(batch):
    """Custom collate function for proper padding"""
    if not batch:
        raise ValueError("Received empty batch - check your dataset implementation")
        
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        # Pad input_ids
        padded_ids = ids + [0] * (max_len - len(ids))
        padded_input_ids.append(padded_ids)
        
        # Pad attention_mask
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.as_tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.as_tensor(padded_attention_masks, dtype=torch.long),
        'labels': torch.as_tensor(labels, dtype=torch.long)
    }

def multi_task_collate_fn(batch):
    """Custom collate function for multi-task learning"""
    if not batch:
        raise ValueError("Received empty batch - check your dataset implementation")
    
    # Get batch size and max length
    batch_size = len(batch)
    max_length = max(len(item['input_ids']) for item in batch)
    
    # Initialize tensors
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    intent_labels = torch.zeros(batch_size, dtype=torch.long)
    entity_labels = torch.zeros(batch_size, max_length, dtype=torch.long)
    value_labels = torch.zeros(batch_size, max_length, dtype=torch.long)
    command_labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        # Ensure input_ids and attention_mask are 1D tensors
        input_ids_tensor = torch.as_tensor(item['input_ids'], dtype=torch.long)
        attention_mask_tensor = torch.as_tensor(item['attention_mask'], dtype=torch.long)
        
        # Handle case where tensors might be 2D [1, seq_len]
        if input_ids_tensor.dim() > 1:
            input_ids_tensor = input_ids_tensor.squeeze()
        if attention_mask_tensor.dim() > 1:
            attention_mask_tensor = attention_mask_tensor.squeeze()
        
        # Get actual sequence length
        seq_len = len(input_ids_tensor)
        
        # Ensure we don't exceed max_length
        seq_len = min(seq_len, max_length)
        
        input_ids[i, :seq_len] = input_ids_tensor[:seq_len]
        attention_mask[i, :seq_len] = attention_mask_tensor[:seq_len]
        intent_labels[i] = item['intent_labels']
        entity_labels[i, :seq_len] = torch.as_tensor(item['entity_labels'][:seq_len], dtype=torch.long)
        value_labels[i, :seq_len] = torch.as_tensor(item['value_labels'][:seq_len], dtype=torch.long)
        command_labels[i] = item['command_labels']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'intent_labels': intent_labels,
        'entity_labels': entity_labels,
        'value_labels': value_labels,
        'command_labels': command_labels
    }

class IntentDataset(torch.utils.data.Dataset):
    """Dataset for Intent Recognition with enhanced robustness"""
    
    def __init__(self, data: List[Dict], tokenizer, intent_to_id: Dict[str, int], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.max_length = max_length
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate dataset to catch issues early"""
        if not self.data:
            raise ValueError("Empty dataset provided")
            
        required_keys = ['input', 'command']
        for i, item in enumerate(self.data[:10]):  # Check first 10 samples
            missing_keys = [k for k in required_keys if k not in item]
            if missing_keys:
                raise KeyError(f"Sample {i} is missing required keys: {missing_keys}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['input']
            intent = item['command']
            
            # Handle empty texts
            if not text.strip():
                text = "empty input"
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Convert intent to ID with fallback to unknown class
            intent_id = self.intent_to_id.get(intent, 0)
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': intent_id
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a default sample as fallback
            dummy_encoding = self.tokenizer(
                "error processing sample",
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            return {
                'input_ids': dummy_encoding['input_ids'],
                'attention_mask': dummy_encoding['attention_mask'],
                'labels': 0
            }

class MultiTaskDataset(torch.utils.data.Dataset):
    """Multi-task dataset for Intent, Entity, Value, Command learning"""
    
    def __init__(self, data: List[Dict], tokenizer, intent_to_id: Dict[str, int], 
                 entity_to_id: Dict[str, int], value_to_id: Dict[str, int], 
                 command_to_id: Dict[str, int], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.entity_to_id = entity_to_id
        self.value_to_id = value_to_id
        self.command_to_id = command_to_id
        self.max_length = max_length
        
        # Get "O" label IDs for padding
        self.entity_o_id = entity_to_id.get("O", 0)
        self.value_o_id = value_to_id.get("O", 0)
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate dataset to catch issues early"""
        if not self.data:
            raise ValueError("Empty dataset provided")
            
        required_keys = ['input', 'command']
        for i, item in enumerate(self.data[:10]):  # Check first 10 samples
            missing_keys = [k for k in required_keys if k not in item]
            if missing_keys:
                raise KeyError(f"Sample {i} is missing required keys: {missing_keys}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['input']
            command = item['command']
            entities = item.get('entities', [])
            values = item.get('values', [])
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get intent label
            intent_label = self.intent_to_id.get(command, 0)
            
            # Get entity labels (IOB2)
            entity_labels = self._align_entity_labels(text, entities)
            
            # Get value labels (IOB2)
            value_labels = self._align_value_labels(text, values)
            
            # Get command label
            command_label = self.command_to_id.get(command, 0)
            
            # Ensure label lengths match seq_len (including special tokens)
            seq_len = int(encoding['input_ids'].shape[1])  # = max_length
            
            # Pad entity labels to seq_len
            if len(entity_labels) < seq_len:
                entity_labels = entity_labels + [self.entity_o_id] * (seq_len - len(entity_labels))
            entity_labels = entity_labels[:seq_len]
            
            # Pad value labels to seq_len
            if len(value_labels) < seq_len:
                value_labels = value_labels + [self.value_o_id] * (seq_len - len(value_labels))
            value_labels = value_labels[:seq_len]
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'intent_labels': intent_label,
                'entity_labels': entity_labels,
                'value_labels': value_labels,
                'command_labels': command_label,
                'text': text
            }
            
        except Exception as e:
            # Return dummy data to prevent training crash
            dummy_encoding = self.tokenizer(
                "dummy text",
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': dummy_encoding['input_ids'],
                'attention_mask': dummy_encoding['attention_mask'],
                'intent_labels': 0,
                'entity_labels': [self.entity_o_id] * self.max_length,
                'value_labels': [self.value_o_id] * self.max_length,
                'command_labels': 0,
                'text': "dummy text"
            }
    
    def _align_entity_labels(self, text: str, entities: List[Dict]) -> List[int]:
        """Align entity labels with tokens using character-based matching (IOB2 format)"""
        # Tokenize without offset mapping (fallback method)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        # Initialize labels as 'O'
        entity_labels = ['O'] * len(tokens)
        
        # Simple character-based alignment (fallback method)
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('label', entity.get('type', ''))
            
            if not entity_text or not entity_type:
                continue
                
            # Find entity position in original text
            entity_start = text.find(entity_text)
            if entity_start == -1:
                continue
                
            # Simple heuristic: find tokens that might contain the entity
            # This is a fallback when offset_mapping is not available
            char_pos = 0
            token_start = None
            token_end = None
            
            for i, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                    
                # Estimate character position
                token_clean = token.replace('##', '')
                if char_pos <= entity_start < char_pos + len(token_clean):
                    token_start = i
                if char_pos < entity_start + len(entity_text) <= char_pos + len(token_clean):
                    token_end = i
                    break
                    
                char_pos += len(token_clean)
            
            # Assign IOB2 labels
            if token_start is not None and token_end is not None:
                entity_labels[token_start] = f"B-{entity_type}"
                for j in range(token_start + 1, token_end + 1):
                    if j < len(entity_labels):
                        entity_labels[j] = f"I-{entity_type}"
        
        # Convert string labels to IDs
        label_ids = []
        for label in entity_labels:
            if label in self.entity_to_id:
                label_ids.append(self.entity_to_id[label])
            else:
                label_ids.append(self.entity_o_id)
        
        return label_ids
    
    def _align_value_labels(self, text: str, values: List[Dict]) -> List[int]:
        """Align value labels with tokens using character-based matching (IOB2 format)"""
        # Tokenize without offset mapping (fallback method)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        # Initialize labels as 'O'
        value_labels = ['O'] * len(tokens)
        
        # Simple character-based alignment (fallback method)
        for value in values:
            value_text = value.get('text', '')
            value_type = value.get('label', value.get('type', ''))
            
            if not value_text or not value_type:
                continue
                
            # Find value position in original text
            value_start = text.find(value_text)
            if value_start == -1:
                continue
                
            # Simple heuristic: find tokens that might contain the value
            char_pos = 0
            token_start = None
            token_end = None
            
            for i, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                    
                # Estimate character position
                token_clean = token.replace('##', '')
                if char_pos <= value_start < char_pos + len(token_clean):
                    token_start = i
                if char_pos < value_start + len(value_text) <= char_pos + len(token_clean):
                    token_end = i
                    break
                    
                char_pos += len(token_clean)
            
            # Assign IOB2 labels
            if token_start is not None and token_end is not None:
                value_labels[token_start] = f"B-{value_type}"
                for j in range(token_start + 1, token_end + 1):
                    if j < len(value_labels):
                        value_labels[j] = f"I-{value_type}"
        
        # Convert string labels to IDs
        label_ids = []
        for label in value_labels:
            if label in self.value_to_id:
                label_ids.append(self.value_to_id[label])
            else:
                label_ids.append(self.value_o_id)
        
        return label_ids

class OptimizedIntentModel(nn.Module):
    """Optimized model for multi-task learning: Intent, Entity, Value, Command"""
    
    def __init__(self, model_name: str, num_intents: int, config: ModelConfig, 
                 num_entity_labels: int = None, num_value_labels: int = None, 
                 num_commands: int = None, enable_multi_task: bool = False):
        super().__init__()
        self.config = config
        self.enable_multi_task = enable_multi_task
        self.num_intents = num_intents
        self.num_entity_labels = num_entity_labels
        self.num_value_labels = num_value_labels
        self.num_commands = num_commands
        
        # Load pretrained model with proper error handling
        try:
            self.phobert = AutoModel.from_pretrained(
                model_name,
                # Gradient checkpointing disabled for stability
                use_safetensors=True,
                trust_remote_code=True,
                cache_dir="model_cache"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model {model_name}: {str(e)}")
        
        # Freeze layers if specified
        if config.freeze_layers > 0:
            modules = [self.phobert.embeddings]
            max_layers = len(self.phobert.encoder.layer)
            freeze_count = min(config.freeze_layers, max_layers)
            
            for i in range(freeze_count):
                modules.append(self.phobert.encoder.layer[i])
                
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        hidden_size = self.phobert.config.hidden_size
        
        # Shared dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Intent classification head (sentence-level) - Optimized for small batches
        if config.model_size == "large":
            self.intent_classifier = nn.Sequential(
                nn.Dropout(0.25),  # Slightly higher dropout for regularization
                nn.Linear(hidden_size, num_intents)  # Direct mapping, no hidden layers
            )
        else:
            self.intent_classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_intents)
            )
        
        # Multi-task heads (if enabled)
        if self.enable_multi_task:
            # Entity extraction head (token-level)
            if num_entity_labels:
                self.entity_classifier = nn.Linear(hidden_size, num_entity_labels)
                self.entity_crf = CRFLayer(num_entity_labels)
            
            # Value extraction head (token-level)
            if num_value_labels:
                self.value_classifier = nn.Linear(hidden_size, num_value_labels)
                self.value_crf = CRFLayer(num_value_labels)
            
            # Command classification head (sentence-level)
            if num_commands:
                self.command_classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden_size // 2, num_commands)
                )
        
        # Legacy support for single-task
        if not self.enable_multi_task:
            if config.model_size == "large":
                # Multi-layer architecture for large model
                self.dropout1 = nn.Dropout(config.dropout)
                self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
                self.activation1 = nn.GELU()
                self.layernorm1 = nn.LayerNorm(hidden_size // 2)
                
                self.dropout2 = nn.Dropout(config.dropout)
                self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.activation2 = nn.GELU()
                self.layernorm2 = nn.LayerNorm(hidden_size // 4)
                
                self.dropout3 = nn.Dropout(config.dropout)
                self.output = nn.Linear(hidden_size // 4, num_intents)
            else:
                # Simpler architecture for base model
                self.dropout = nn.Dropout(config.dropout)
                self.output = nn.Linear(hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask, intent_labels=None, 
                entity_labels=None, value_labels=None, command_labels=None,
                lambda_intent=1.0, lambda_entity=0.5, lambda_value=0.5, lambda_command=0.3):
        # Forward pass through PhoBERT
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False  # Disable cache to prevent memory issues
        )
        
        # Get token embeddings (for sequence labeling)
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Use mean pooling for sentence classification (better for Vietnamese)
        # Mean pooling is more stable for short Vietnamese sentences
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, token_embeddings.size(-1)).float()
        masked_embeddings = token_embeddings * attention_mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.sum(attention_mask.float(), dim=1, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Multi-task forward pass
        if self.enable_multi_task:
            results = {}
            
            # Intent classification
            intent_logits = self.intent_classifier(pooled_output)
            results['intent_logits'] = intent_logits
            
            # Entity extraction (if enabled)
            if hasattr(self, 'entity_classifier'):
                entity_logits = self.entity_classifier(token_embeddings)
                results['entity_logits'] = entity_logits
                
                if entity_labels is not None:
                    entity_loss = self.entity_crf(entity_logits, entity_labels, attention_mask)
                    results['entity_loss'] = entity_loss
            
            # Value extraction (if enabled)
            if hasattr(self, 'value_classifier'):
                value_logits = self.value_classifier(token_embeddings)
                results['value_logits'] = value_logits
                
                if value_labels is not None:
                    value_loss = self.value_crf(value_logits, value_labels, attention_mask)
                    results['value_loss'] = value_loss
            
            # Command classification (if enabled)
            if hasattr(self, 'command_classifier'):
                command_logits = self.command_classifier(pooled_output)
                results['command_logits'] = command_logits
            
            # Calculate total loss with lambda weights
            total_loss = 0.0
            if intent_labels is not None:
                intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)
                total_loss += lambda_intent * intent_loss
                results['intent_loss'] = intent_loss
            
            if command_labels is not None and hasattr(self, 'command_classifier'):
                command_loss = nn.CrossEntropyLoss()(command_logits, command_labels)
                total_loss += lambda_command * command_loss
                results['command_loss'] = command_loss
            
            if 'entity_loss' in results:
                total_loss += lambda_entity * results['entity_loss']
            
            if 'value_loss' in results:
                total_loss += lambda_value * results['value_loss']
            
            results['loss'] = total_loss
            return results
        
        else:
            # Legacy single-task forward pass
            if self.config.model_size == "large":
                x = self.dropout1(pooled_output)
                x = self.linear1(x)
                x = self.activation1(x)
                x = self.layernorm1(x)
                
                x = self.dropout2(x)
                x = self.linear2(x)
                x = self.activation2(x)
                x = self.layernorm2(x)
                
                x = self.dropout3(x)
                logits = self.output(x)
            else:
                x = self.dropout(pooled_output)
                logits = self.output(x)
                
            return logits

class GPUTrainer:
    """Enhanced trainer with better GPU optimization, error handling, and monitoring"""
    
    def __init__(self, config: ModelConfig, intent_config: IntentConfig, 
                 entity_config: EntityConfig = None, value_config: ValueConfig = None,
                 command_config: CommandConfig = None, training_config = None, 
                 enable_multi_task: bool = False):
        self.config = config
        self.intent_config = intent_config
        self.entity_config = entity_config
        self.value_config = value_config
        self.command_config = command_config
        self.training_config = training_config
        self.enable_multi_task = enable_multi_task
        self.device = self._setup_device()
        
        # Multi-task scheduling
        self.current_epoch = 0
        self.multi_task_start_epoch = 6
        
        # Setup logging with timestamp for unique files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{config.model_size}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log system info
        self._log_system_info()
        
        # Initialize mixed precision training
        self.scaler = self._setup_amp()
        
        # Load tokenizer with error handling
        self.tokenizer = self._load_tokenizer()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Performance monitoring
        self.train_start_time = None
        self.batch_times = []
        
        # Initialize training state
        self.train_dataset = None
        self.val_dataset = None
        self.intent_to_id = None
        self.id_to_intent = None
        
        # Set random seed for reproducibility
        set_seed(getattr(config, 'seed', 42))
        
    def _get_multi_task_weights(self, epoch: int) -> Dict[str, float]:
        """Get multi-task loss weights based on epoch"""
        if epoch < self.multi_task_start_epoch:
            # Single task (intent only) for first 5 epochs
            return {
                'lambda_intent': 1.0,
                'lambda_command': 0.0,
                'lambda_entity': 0.0,
                'lambda_value': 0.0
            }
        else:
            # Multi-task from epoch 6 onwards
            return {
                'lambda_intent': 1.0,
                'lambda_command': 0.3,  # Adjust if command != intent
                'lambda_entity': 0.5,
                'lambda_value': 0.5
            }
        
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device with better error handling"""
        # Use training_config.device if available, otherwise fallback to auto-detection
        if self.training_config and hasattr(self.training_config, 'device'):
            device_setting = self.training_config.device
        else:
            device_setting = "auto"
        
        if device_setting == "auto":
            # Force CPU training due to CUDA issues
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
                print("🖥️ CUDA disabled, using CPU")
                device = torch.device("cpu")
            elif torch.cuda.is_available():
                # Test CUDA operations before using
                try:
                    test_tensor = torch.randn(2, 2).cuda()
                    test_tensor.to('cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    device = torch.device("cuda")
                except RuntimeError as e:
                    if "operation not supported" in str(e):
                        print("⚠️ CUDA operations not supported, falling back to CPU")
                        device = torch.device("cpu")
                    else:
                        raise e
            else:
                device = torch.device("cpu")
        elif torch.cuda.is_available() and device_setting.startswith('cuda'):
            device = torch.device(device_setting)
            # Verify the specified GPU exists
            if device_setting != 'cuda' and int(device_setting.split(':')[1]) >= torch.cuda.device_count():
                print(f"Warning: Specified GPU {device_setting} doesn't exist. Falling back to cuda:0")
                device = torch.device('cuda:0')
        else:
            if device_setting.startswith('cuda'):
                print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
            
        return device
    
    def _initialize_model(self):
        """Initialize model with multi-task support"""
        try:
            if self.enable_multi_task:
                # Multi-task model initialization
                model = OptimizedIntentModel(
                    self.config.model_name,
                    num_intents=self.intent_config.num_intents,
                    config=self.config,
                    num_entity_labels=self.entity_config.num_entity_labels,
                    num_value_labels=self.value_config.num_value_labels,
                    num_commands=self.command_config.num_command_labels,
                    enable_multi_task=True
                ).to(self.device)
                
                self.logger.info("🎯 Initialized multi-task model with:")
                self.logger.info(f"   - Intent labels: {self.intent_config.num_intents}")
                self.logger.info(f"   - Entity labels: {self.entity_config.num_entity_labels}")
                self.logger.info(f"   - Value labels: {self.value_config.num_value_labels}")
                self.logger.info(f"   - Command labels: {self.command_config.num_command_labels}")
            else:
                # Single-task model initialization (legacy)
                model = OptimizedIntentModel(
                    self.config.model_name, 
                    self.intent_config.num_intents, 
                    self.config
                ).to(self.device)
                
                self.logger.info(f"🎯 Initialized single-task model with {self.intent_config.num_intents} intents")
            
            # Log model size and parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"📊 Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
            self.logger.info(f"📊 Model size on disk: ~{total_params * 4 / (1024**2):.1f} MB")
            
            # Disable gradient checkpointing for stability
            self.logger.info("🔧 Gradient Checkpointing disabled for stability")
            
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _log_system_info(self) -> None:
        """Log detailed system information"""
        self.logger.info(f"🖥️ Using device: {self.device}")
        
        if self.device.type == "cuda" and torch.cuda.device_count() > 0:
            try:
                self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
                self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                self.logger.info(f"📊 CUDA Version: {torch.version.cuda}")
                self.logger.info(f"📊 CUDNN Version: {torch.backends.cudnn.version()}")
                
                # Monitor initial GPU memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(0) / (1024**2)
                self.logger.info(f"💾 Initial GPU Memory Usage: {initial_memory:.1f} MB")
                
                # Enable cudnn benchmark for performance if not using deterministic algorithms
                torch.backends.cudnn.benchmark = not getattr(self.config, 'deterministic', False)
                
                # Log VRAM usage
                torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, AssertionError) as e:
                self.logger.warning(f"⚠️ Could not access GPU properties: {e}")
                self.logger.info("🖥️ Falling back to CPU training")
            
        self.logger.info(f"📊 PyTorch Version: {torch.__version__}")
        self.logger.info(f"📊 Training config: {vars(self.config)}")
    
    def _setup_amp(self) -> amp.GradScaler:
        """Setup Automatic Mixed Precision for better performance"""
        # Always disable FP16 to avoid dtype mismatch
        scaler = amp.GradScaler(enabled=False)
        self.logger.info("🔧 FP16 disabled to avoid dtype mismatch")
                
        return scaler
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer with error handling"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.info(f"ℹ️ Set pad_token to eos_token: {tokenizer.pad_token}")
            return tokenizer
        except Exception as e:
            self.logger.error(f"❌ Failed to load tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    
    def load_data(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load data with improved error handling and analysis"""
        self.logger.info(f"📂 Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"❌ Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"❌ Invalid JSON in data file: {file_path}")
            raise ValueError(f"Invalid JSON in data file: {file_path}")
        except UnicodeDecodeError:
            self.logger.error(f"❌ Encoding issue in data file: {file_path}")
            # Try with different encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    data = json.load(f)
                self.logger.warning("⚠️ Had to use latin-1 encoding for data file")
            except:
                raise ValueError(f"Unable to decode data file: {file_path}")
        
        if not data:
            self.logger.error("❌ Empty dataset loaded")
            raise ValueError("Empty dataset loaded")
            
        self.logger.info(f"📈 Loaded {len(data)} samples")
        
        # Analyze token lengths
        try:
            token_lengths = [len(self.tokenizer.encode(item['input'])) for item in data]
            max_len = max(token_lengths)
            avg_len = sum(token_lengths) / len(token_lengths)
            
            self.logger.info(f"📊 Token statistics - Max: {max_len}, Avg: {avg_len:.1f}")
            
            if max_len > self.config.max_length:
                self.logger.warning(f"⚠️ {sum(1 for l in token_lengths if l > self.config.max_length)} samples exceed max length {self.config.max_length} tokens (max: {max_len})")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not analyze token lengths: {str(e)}")
        
        # Analyze intent distribution
        intent_counts = {}
        for item in data:
            intent = item.get('command', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        self.logger.info("📊 Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count} samples ({count/len(data)*100:.1f}%)")
        
        # Check for class imbalance
        if len(intent_counts) > 1:
            min_count = min(intent_counts.values())
            max_count = max(intent_counts.values())
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 10:
                self.logger.warning(f"⚠️ Severe class imbalance detected - ratio of {imbalance_ratio:.1f}:1")
            elif imbalance_ratio > 3:
                self.logger.warning(f"⚠️ Moderate class imbalance detected - ratio of {imbalance_ratio:.1f}:1")
        
        # Create FIXED label mappings and save to file
        intents = sorted(list(intent_counts.keys()))
        self.intent_to_id = {intent: i for i, intent in enumerate(intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        self.logger.info(f"🎯 Found {len(intents)} intents: {intents}")
        self.logger.info("🔒 FIXED label mapping created - will be used for ALL train/val/test")
        
        # Create multi-task label mappings if enabled
        if self.enable_multi_task:
            # Entity labels (IOB2 format) - use config labels directly
            entity_labels = self.entity_config.entity_labels  # Already has O, B-*, I-*
            self.entity_to_id = {label: i for i, label in enumerate(entity_labels)}
            self.id_to_entity = {i: label for label, i in self.entity_to_id.items()}
            
            # Value labels (IOB2 format) - use config labels directly
            value_labels = self.value_config.value_labels  # Already has O, B-*, I-*
            self.value_to_id = {label: i for i, label in enumerate(value_labels)}
            self.id_to_value = {i: label for label, i in self.value_to_id.items()}
            
            # Command labels - use config labels to match model initialization
            command_labels = self.command_config.command_labels
            self.command_to_id = {label: i for i, label in enumerate(command_labels)}
            self.id_to_command = {i: label for label, i in self.command_to_id.items()}
            
            # Map dataset intents to config command labels
            for intent in intents:
                if intent not in self.command_to_id:
                    # Map unknown intents to 'unknown' command
                    self.command_to_id[intent] = self.command_to_id.get('unknown', 0)
            
            self.logger.info(f"🏷️ Entity labels: {len(entity_labels)}")
            self.logger.info(f"🏷️ Value labels: {len(value_labels)}")
            self.logger.info(f"🏷️ Command labels: {len(command_labels)}")
        
        # Save FIXED label mappings to file
        label_maps = {
            "intent": intents,
            "entity": entity_labels if self.enable_multi_task else [],
            "value": value_labels if self.enable_multi_task else [],
            "command": command_labels if self.enable_multi_task else []
        }
        
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Save label mappings
        with open("models/label_maps.json", 'w', encoding='utf-8') as f:
            json.dump(label_maps, f, ensure_ascii=False, indent=2)
        
        self.logger.info("💾 Saved FIXED label mappings to models/label_maps.json")
        
        # Perform train-validation split with error handling
        try:
            train_data, val_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=getattr(self.config, 'seed', 42),
                stratify=[item.get('command', 'unknown') for item in data]  # Stratified split
            )
            self.logger.info("📊 Used stratified split for train/val")
        except ValueError as e:
            self.logger.warning(f"⚠️ Stratified split failed: {str(e)}. Using random split.")
            train_data, val_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=getattr(self.config, 'seed', 42)
            )
        
        # SANITY CHECKS: Verify split doesn't lose any classes
        train_intents = set(item.get('command', 'unknown') for item in train_data)
        val_intents = set(item.get('command', 'unknown') for item in val_data)
        all_intents = set(intent_counts.keys())
        
        missing_in_train = all_intents - train_intents
        missing_in_val = all_intents - val_intents
        
        if missing_in_train:
            self.logger.warning(f"⚠️ Some intents missing from training set: {missing_in_train}")
        if missing_in_val:
            self.logger.warning(f"⚠️ Some intents missing from validation set: {missing_in_val}")
        
        # SANITY CHECK: Print intent distribution in train/val
        self.logger.info("🔍 SANITY CHECK - Train intents:")
        train_intent_counts = {}
        for item in train_data:
            intent = item.get('command', 'unknown')
            train_intent_counts[intent] = train_intent_counts.get(intent, 0) + 1
        for intent, count in sorted(train_intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count}")
        
        self.logger.info("🔍 SANITY CHECK - Val intents:")
        val_intent_counts = {}
        for item in val_data:
            intent = item.get('command', 'unknown')
            val_intent_counts[intent] = val_intent_counts.get(intent, 0) + 1
        for intent, count in sorted(val_intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count}")
        
        # SANITY CHECK: Check for suspicious distribution
        max_val_count = max(val_intent_counts.values()) if val_intent_counts else 0
        if max_val_count > 200:  # Suspicious if any class has >200 samples in val
            self.logger.error(f"🚨 SUSPICIOUS: Val has class with {max_val_count} samples - likely mapping error!")
            self.logger.error(f"🚨 Most common val intent: {max(val_intent_counts, key=val_intent_counts.get)}")
        
        # Create datasets
        try:
            if self.enable_multi_task:
                # Create multi-task datasets
                self.train_dataset = MultiTaskDataset(
                    train_data, 
                    self.tokenizer, 
                    self.intent_to_id,
                    self.entity_to_id,
                    self.value_to_id,
                    self.command_to_id,
                    self.config.max_length
                )
                self.val_dataset = MultiTaskDataset(
                    val_data, 
                    self.tokenizer, 
                    self.intent_to_id,
                    self.entity_to_id,
                    self.value_to_id,
                    self.command_to_id,
                    self.config.max_length
                )
            else:
                # Create single-task datasets
                self.train_dataset = IntentDataset(
                    train_data, 
                    self.tokenizer, 
                    self.intent_to_id, 
                    self.config.max_length
                )
                self.val_dataset = IntentDataset(
                    val_data, 
                    self.tokenizer, 
                    self.intent_to_id, 
                    self.config.max_length
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to create datasets: {str(e)}")
            raise RuntimeError(f"Failed to create datasets: {str(e)}")
        
        self.logger.info(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
        
        return train_data, val_data
    
    def _setup_training(self, train_loader, val_loader):
        """Setup optimizer, scheduler, and loss function"""
        # Setup optimizer with error handling
        try:
            optimizer = self._create_optimizer()
        except Exception as e:
            self.logger.error(f"❌ Failed to create optimizer: {str(e)}")
            raise RuntimeError(f"Failed to create optimizer: {str(e)}")
        
        # Setup learning rate scheduler with 10% warmup
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * 0.1)  # Exactly 10% warmup
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"📊 Training steps: {total_steps}, Warmup steps: {warmup_steps} (10%)")
        
        # Setup loss function with class weights for imbalance
        class_weights = self._calculate_class_weights()
        if class_weights is not None and len(class_weights) == self.intent_config.num_intents:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        return optimizer, scheduler, criterion
    
    def train(self) -> Dict[str, Any]:
        """Enhanced training loop with better error handling, monitoring and optimization"""
        if self.train_dataset is None or self.val_dataset is None:
            self.logger.error("❌ Datasets not initialized. Call load_data() first.")
            raise RuntimeError("Datasets not initialized. Call load_data() first.")
            
        self.logger.info("🚀 Starting training...")
        self.train_start_time = time.time()
        
        # Create data loaders with error handling and optimizations
        try:
            # Cấu hình tối ưu cho GPU 6GB
            num_workers = 0  # Windows - tránh multiprocessing issues
            pin_memory = self.device.type == 'cuda'
            persistent_workers = False  # Không cần khi num_workers=0
            
            self.logger.info(f"📊 DataLoader config: workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
            
            # Choose collate function based on multi-task mode
            collate_fn = multi_task_collate_fn if self.enable_multi_task else custom_collate_fn
            
            # Create WeightedRandomSampler for class imbalance
            sampler = self._create_weighted_sampler()
            
            # DataLoader kwargs - tối ưu cho GPU 6GB
            dl_kwargs = {
                'batch_size': self.config.batch_size,
                'sampler': sampler,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'persistent_workers': persistent_workers,
                'collate_fn': collate_fn,
                'drop_last': False
            }
            # Không set prefetch_factor khi num_workers=0
            
            train_loader = DataLoader(self.train_dataset, **dl_kwargs)
            
            # Validation loader (NO SAMPLER - critical fix!)
            val_kwargs = {
                'batch_size': self.config.batch_size,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'persistent_workers': persistent_workers,
                'collate_fn': collate_fn,
                'drop_last': False,
                'shuffle': False
            }
            # CRITICAL: No sampler for validation to avoid fake distribution
            val_loader = DataLoader(self.val_dataset, **val_kwargs)
            self.logger.info("🔒 Validation DataLoader: NO SAMPLER (prevents fake distribution)")
        except Exception as e:
            self.logger.error(f"❌ Failed to create data loaders: {str(e)}")
            raise RuntimeError(f"Failed to create data loaders: {str(e)}")
        
        # Setup training components
        optimizer, scheduler, criterion = self._setup_training(train_loader, val_loader)
        
        # Training state with early stopping
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_weighted_f1 = 0.0
        early_stopping_patience = 5
        early_stopping_counter = 0
        epoch_times = []
        
        metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Main training loop
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(
                train_loader, 
                optimizer, 
                scheduler, 
                criterion, 
                epoch
            )
            
            # Validation phase
            val_metrics = self._validate(val_loader, criterion)
            
            # Update metrics history
            metrics_history['train_loss'].append(train_metrics['loss'])
            metrics_history['train_acc'].append(train_metrics['acc'])
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['val_acc'].append(val_metrics['acc'])
            
            # Learning rate adjustment
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and (epoch + 1) % 2 == 0:
                # LR decay every 2 epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
                new_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"📉 Adjusted learning rate: {current_lr:.6f} → {new_lr:.6f}")
            
            # Save checkpoint for every epoch
            self.save_model(epoch=epoch+1, is_best=False)
            
            # Check if this is the best model
            current_weighted_f1 = val_metrics.get('weighted_f1', val_metrics.get('f1', 0.0))
            if current_weighted_f1 > best_weighted_f1:
                best_weighted_f1 = current_weighted_f1
                best_val_acc = val_metrics['acc']
                best_val_f1 = val_metrics.get('f1', 0.0)
                early_stopping_counter = 0  # Reset counter
                self.save_model(epoch=epoch+1, is_best=True)
                self.logger.info(f"💾 Saved best model with weighted F1: {best_weighted_f1:.4f}")
            else:
                early_stopping_counter += 1
                self.logger.info(f"⏳ No improvement for {early_stopping_counter} epochs (patience: {early_stopping_patience})")
            
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"🏆 Best weighted F1: {best_weighted_f1:.4f}")
                break
            
            # Calculate and log epoch time
            epoch_end = time.time()
            epoch_minutes = (epoch_end - epoch_start) / 60
            epoch_times.append(epoch_minutes)
            
            self.logger.info(f"⏱️ Epoch time: {epoch_minutes:.2f} minutes")
            
            # Estimate remaining time
            if len(epoch_times) > 1:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = self.config.num_epochs - (epoch + 1)
                est_remaining_time = avg_epoch_time * remaining_epochs
                
                self.logger.info(f"⏱️ Estimated remaining time: {est_remaining_time:.2f} minutes")
            
            # Memory cleanup - anti-fragment cho GPU 6GB
            if self.device.type == "cuda":
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    self.logger.info(f"🧹 Memory cleanup - Current GPU memory: {current_mem:.2f} GB")
        
        # Training complete
        total_time = (time.time() - self.train_start_time) / 60
        self.logger.info(f"\n🎉 Training completed in {total_time:.2f} minutes!")
        self.logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        self.logger.info(f"🏆 Best validation F1 score: {best_val_f1:.4f}")
        
        # Return training history
        return {
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'metrics_history': metrics_history,
            'training_time_minutes': total_time
        }
    
    def _create_weighted_sampler(self):
        """Tạo WeightedRandomSampler để xử lý class imbalance"""
        try:
            # Lấy tất cả intent labels từ dataset
            intent_labels = []
            for item in self.train_dataset.data:
                intent = item.get('command', 'unknown')
                intent_labels.append(intent)
            
            # Đếm số lượng mỗi class
            from collections import Counter
            class_counts = Counter(intent_labels)
            
            # Tính weights (inverse frequency)
            total_samples = len(intent_labels)
            class_weights = {}
            for intent, count in class_counts.items():
                class_weights[intent] = total_samples / (len(class_counts) * count)
            
            # Tạo weights cho mỗi sample
            sample_weights = []
            for item in self.train_dataset.data:
                intent = item.get('command', 'unknown')
                sample_weights.append(class_weights[intent])
            
            # Tạo WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            self.logger.info(f"📊 Created WeightedRandomSampler for {len(class_counts)} classes")
            self.logger.info(f"📊 Class weights: {dict(list(class_counts.items())[:5])}...")  # Show first 5
            
            return sampler
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create WeightedRandomSampler: {e}")
            self.logger.warning("⚠️ Using default shuffle instead")
            return None
    
    def _calculate_class_weights(self):
        """Tính class weights để xử lý class imbalance - FIXED mapping"""
        try:
            # Lấy tất cả intent labels từ dataset
            intent_labels = []
            for item in self.train_dataset.data:
                intent = item.get('command', 'unknown')
                intent_labels.append(intent)
            
            # Đếm số lượng mỗi class
            from collections import Counter
            class_counts = Counter(intent_labels)
            
            # CRITICAL FIX: Dùng self.intent_to_id đã cố định, không tạo mới
            num_classes = len(self.intent_to_id)
            class_weights = torch.zeros(num_classes)
            
            # Tính weights theo đúng mapping đã train
            total_samples = len(intent_labels)
            for intent, count in class_counts.items():
                if intent in self.intent_to_id:
                    intent_id = self.intent_to_id[intent]
                    weight = total_samples / (num_classes * count)
                    class_weights[intent_id] = weight
                else:
                    self.logger.warning(f"⚠️ Intent '{intent}' not in mapping, skipping")
            
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * num_classes
            
            self.logger.info(f"📊 Calculated class weights for {num_classes} classes using FIXED mapping")
            self.logger.info(f"📊 Weight range: {class_weights.min():.3f} - {class_weights.max():.3f}")
            
            return class_weights.to(self.device)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate class weights: {e}")
            self.logger.warning("⚠️ Using uniform weights")
            return None
    
    def _create_optimizer(self) -> Union[Adafactor, AdamW]:
        """Create and return the appropriate optimizer with separate learning rates - FIXED grouping"""
        if hasattr(self.config, 'optimizer') and self.config.optimizer.lower() == "adafactor":
            optimizer = Adafactor(
                self.model.parameters(),
                lr=self.config.learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                clip_threshold=getattr(self.config, 'clip_threshold', 1.0)
            )
            self.logger.info("🔧 Using Adafactor optimizer (memory efficient)")
        else:
            # FIXED: Build groups directly from named_parameters() in one loop
            no_decay = ['bias', 'LayerNorm.weight']
            enc_lr = 1e-5  # Lower LR for encoder
            head_lr = 3e-4  # Higher LR for heads
            
            enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                is_encoder = 'phobert' in name
                has_decay = not any(nd in name for nd in no_decay)
                
                if is_encoder and has_decay:
                    enc_decay.append(param)
                elif is_encoder:
                    enc_nodecay.append(param)
                elif not is_encoder and has_decay:
                    head_decay.append(param)
                else:
                    head_nodecay.append(param)
            
            param_groups = [
                {'params': enc_decay, 'lr': enc_lr, 'weight_decay': self.config.weight_decay},
                {'params': enc_nodecay, 'lr': enc_lr, 'weight_decay': 0.0},
                {'params': head_decay, 'lr': head_lr, 'weight_decay': self.config.weight_decay},
                {'params': head_nodecay, 'lr': head_lr, 'weight_decay': 0.0},
            ]
            
            optimizer = AdamW(
                param_groups,
                eps=getattr(self.config, 'adam_epsilon', 1e-8),
                betas=getattr(self.config, 'adam_betas', (0.9, 0.999))
            )
            self.logger.info("🔧 Using AdamW with FIXED grouping: Encoder=1e-5, Heads=3e-4")
        
        return optimizer
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: Union[Adafactor, AdamW], 
        scheduler: Any, 
        criterion: nn.Module, 
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with better error handling and monitoring"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        batch_start_time = time.time()
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device with error handling
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                except RuntimeError as e:
                    if "CUDA error: operation not supported" in str(e):
                        self.logger.warning(f"⚠️ CUDA operation not supported, skipping batch {batch_idx}")
                        continue
                    else:
                        raise e
                
                # Handle empty batches
                if input_ids.size(0) == 0:
                    self.logger.warning("⚠️ Received empty batch - skipping")
                    continue
                
                if self.enable_multi_task:
                    # Multi-task training with dynamic weights
                    intent_labels = batch['intent_labels'].to(self.device)
                    entity_labels = batch['entity_labels'].to(self.device)
                    value_labels = batch['value_labels'].to(self.device)
                    command_labels = batch['command_labels'].to(self.device)
                    
                    # Get multi-task weights based on current epoch
                    weights = self._get_multi_task_weights(epoch)
                    
                    outputs = self.model(
                        input_ids, 
                        attention_mask,
                        intent_labels=intent_labels,
                        entity_labels=entity_labels,
                        value_labels=value_labels,
                        command_labels=command_labels,
                        lambda_intent=weights['lambda_intent'],
                        lambda_entity=weights['lambda_entity'],
                        lambda_value=weights['lambda_value'],
                        lambda_command=weights['lambda_command']
                    )
                    
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                else:
                    # Single-task training (legacy)
                    labels = batch['labels'].to(self.device)
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels) / self.config.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Clear CUDA cache periodically to prevent memory issues
                    if batch_idx % 5 == 0:  # More frequent cleanup
                        torch.cuda.empty_cache()
                
                # Update metrics
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                
                if self.enable_multi_task:
                    # Multi-task metrics
                    intent_logits = outputs['intent_logits']
                    _, predicted = torch.max(intent_logits.data, 1)
                    labels = intent_labels  # để phần cộng dồn không lỗi
                    batch_correct = (predicted == intent_labels).sum().item()
                    train_total += intent_labels.size(0)
                    train_correct += batch_correct
                else:
                    # Single-task metrics
                    _, predicted = torch.max(logits.data, 1)
                    batch_correct = (predicted == labels).sum().item()
                    train_total += labels.size(0)
                    train_correct += batch_correct
                
                # Calculate batch time for monitoring
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                self.batch_times.append(batch_time)
                batch_start_time = time.time()
                
                # Update progress bar
                if len(self.batch_times) > 1:
                    avg_batch_time = sum(self.batch_times[-10:]) / min(len(self.batch_times), 10)
                    examples_per_second = self.config.batch_size / avg_batch_time
                    
                    # Add GPU memory info to progress bar
                    postfix_dict = {
                        'Loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%',
                        'ex/s': f'{examples_per_second:.1f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    }
                    
                    if self.device.type == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated(0) / (1024**2)
                        postfix_dict['GPU'] = f'{gpu_memory:.0f}MB'
                    
                    progress_bar.set_postfix(postfix_dict)
                else:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%'
                    })
                
                # Periodically log GPU memory usage and cleanup
                if self.device.type == "cuda" and batch_idx % 50 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    self.logger.debug(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                    
                    # Cleanup if memory usage is high
                    if mem_allocated > 4.0:  # If using more than 4GB
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                            self.logger.info(f"🧹 Memory cleanup - Current GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
                        except RuntimeError as e:
                            self.logger.warning(f"⚠️ Could not clear GPU cache: {e}")
                            gc.collect()
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Try to recover and continue
                optimizer.zero_grad()
                continue
        
        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100 * train_correct / max(train_total, 1)
        
        self.logger.info(f"📊 Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        return {
            'loss': epoch_loss,
            'acc': epoch_acc
        }
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate model with detailed metrics including NER/Value extraction"""
        self.model.eval()
        
        # Separate collections for different tasks
        all_intent_logits = []
        all_intent_labels = []
        all_entity_predictions = []
        all_entity_true_labels = []
        all_value_predictions = []
        all_value_true_labels = []
        
        with torch.no_grad():
            # Validation với memory optimization
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    # Move data to device with error handling
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                    except RuntimeError as e:
                        if "CUDA error: operation not supported" in str(e):
                            self.logger.warning(f"⚠️ CUDA operation not supported, skipping batch {batch_idx}")
                            continue
                        else:
                            raise e
                    
                    # Handle empty batches
                    if input_ids.size(0) == 0:
                        continue
                    
                    # Ensure tensors are properly shaped and on correct device
                    if attention_mask.dim() != 2:
                        self.logger.warning(f"⚠️ Invalid attention_mask shape: {attention_mask.shape}")
                        continue
                    
                    if input_ids.dim() != 2:
                        self.logger.warning(f"⚠️ Invalid input_ids shape: {input_ids.shape}")
                        continue
                    
                    if self.enable_multi_task:
                        # Multi-task validation
                        intent_labels = batch['intent_labels'].to(self.device)
                        entity_labels = batch['entity_labels'].to(self.device)
                        value_labels = batch['value_labels'].to(self.device)
                        command_labels = batch['command_labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids, 
                            attention_mask,
                            intent_labels=intent_labels,
                            entity_labels=entity_labels,
                            value_labels=value_labels,
                            command_labels=command_labels
                        )
                        
                        # Collect intent predictions - chuyển CPU ngay để tránh giữ tham chiếu lớn
                        intent_logits = outputs['intent_logits']
                        all_intent_logits.append(intent_logits.cpu())
                        all_intent_labels.append(intent_labels.cpu())
                        
                        # Collect entity predictions using CRF decode
                        if 'entity_logits' in outputs:
                            try:
                                entity_logits = outputs['entity_logits']
                                
                                # Ensure attention_mask is correct format for CRF
                                attention_mask_crf = attention_mask.float()  # Convert to float
                                
                                # Ensure entity_labels are in correct range
                                entity_labels_clamped = torch.clamp(entity_labels, 0, len(self.id_to_entity) - 1)
                                
                                # Use CRF decode to get best tag sequence
                                entity_predictions = self.model.entity_crf.decode(entity_logits, attention_mask_crf)
                                
                                # Convert predictions and labels to tag sequences
                                for i, (pred_seq, true_seq, mask) in enumerate(zip(entity_predictions, entity_labels_clamped, attention_mask_crf)):
                                    # Only consider non-padded tokens
                                    seq_len = int(mask.sum().item())
                                    if seq_len > 0:
                                        # Ensure indices are valid
                                        pred_tags = []
                                        true_tags = []
                                        
                                        for j in range(seq_len):
                                            pred_id = pred_seq[j].item()
                                            true_id = true_seq[j].item()
                                            
                                            if pred_id in self.id_to_entity:
                                                pred_tags.append(self.id_to_entity[pred_id])
                                            else:
                                                pred_tags.append('O')
                                                
                                            if true_id in self.id_to_entity:
                                                true_tags.append(self.id_to_entity[true_id])
                                            else:
                                                true_tags.append('O')
                                    else:
                                        pred_tags = []
                                        true_tags = []
                                    
                                    all_entity_predictions.append(pred_tags)
                                    all_entity_true_labels.append(true_tags)
                                    
                                    # Log sample predictions for debugging
                                    if i < 5:  # Log first 5 samples
                                        self.logger.info(f"🔍 Entity Sample {i}: pred={pred_tags[:10]}, true={true_tags[:10]}")
                                    
                                    # Log non-O labels for debugging
                                    non_o_pred = [tag for tag in pred_tags if tag != 'O']
                                    non_o_true = [tag for tag in true_tags if tag != 'O']
                                    if non_o_pred or non_o_true:
                                        self.logger.info(f"🔍 Entity Non-O Sample {i}: pred={non_o_pred}, true={non_o_true}")
                            except Exception as e:
                                # Skip entity evaluation if there's an error
                                self.logger.warning(f"⚠️ Skipping entity evaluation due to error: {e}")
                                # Clear GPU cache to prevent memory issues (with error handling)
                                if self.device.type == "cuda":
                                    try:
                                        torch.cuda.empty_cache()
                                    except RuntimeError as e:
                                        self.logger.warning(f"⚠️ Could not clear GPU cache: {e}")
                                        gc.collect()
                                continue
                        
                        # Collect value predictions using CRF decode
                        if 'value_logits' in outputs:
                            try:
                                value_logits = outputs['value_logits']
                                
                                # Ensure attention_mask is correct format for CRF
                                attention_mask_crf = attention_mask.float()  # Convert to float
                                
                                # Ensure value_labels are in correct range
                                value_labels_clamped = torch.clamp(value_labels, 0, len(self.id_to_value) - 1)
                                
                                # Use CRF decode to get best tag sequence
                                value_predictions = self.model.value_crf.decode(value_logits, attention_mask_crf)
                                
                                # Convert predictions and labels to tag sequences
                                for i, (pred_seq, true_seq, mask) in enumerate(zip(value_predictions, value_labels_clamped, attention_mask_crf)):
                                    # Only consider non-padded tokens
                                    seq_len = int(mask.sum().item())
                                    if seq_len > 0:
                                        # Ensure indices are valid
                                        pred_tags = []
                                        true_tags = []
                                        
                                        for j in range(seq_len):
                                            pred_id = pred_seq[j].item()
                                            true_id = true_seq[j].item()
                                            
                                            if pred_id in self.id_to_value:
                                                pred_tags.append(self.id_to_value[pred_id])
                                            else:
                                                pred_tags.append('O')
                                                
                                            if true_id in self.id_to_value:
                                                true_tags.append(self.id_to_value[true_id])
                                            else:
                                                true_tags.append('O')
                                    else:
                                        pred_tags = []
                                        true_tags = []
                                    
                                    all_value_predictions.append(pred_tags)
                                    all_value_true_labels.append(true_tags)
                                    
                                    # Log non-O labels for debugging
                                    non_o_pred = [tag for tag in pred_tags if tag != 'O']
                                    non_o_true = [tag for tag in true_tags if tag != 'O']
                                    if non_o_pred or non_o_true:
                                        self.logger.info(f"🔍 Value Non-O Sample {i}: pred={non_o_pred}, true={non_o_true}")
                            except Exception as e:
                                # Skip value evaluation if there's an error
                                self.logger.warning(f"⚠️ Skipping value evaluation due to error: {e}")
                                # Clear GPU cache to prevent memory issues (with error handling)
                                if self.device.type == "cuda":
                                    try:
                                        torch.cuda.empty_cache()
                                    except RuntimeError as e:
                                        self.logger.warning(f"⚠️ Could not clear GPU cache: {e}")
                                        gc.collect()
                                continue
                        
                    else:
                        # Single-task validation (legacy)
                        labels = batch['labels'].to(self.device)
                        logits = self.model(input_ids, attention_mask)
                        all_intent_logits.append(logits)
                        all_intent_labels.append(labels)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in validation batch: {str(e)}")
                    self.logger.error(f"❌ Batch details - input_ids: {input_ids.shape if 'input_ids' in locals() else 'N/A'}, attention_mask: {attention_mask.shape if 'attention_mask' in locals() else 'N/A'}")
                    # Clear GPU cache to prevent memory issues (with error handling)
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError as e:
                            self.logger.warning(f"⚠️ Could not clear GPU cache: {e}")
                            # Try alternative cleanup
                            gc.collect()
                    continue
        
        # Check if we have any valid results
        if not all_intent_logits:
            self.logger.error("❌ No valid validation results")
            return {'loss': float('inf'), 'acc': 0.0, 'f1': 0.0}
        
        # Initialize results dictionary
        results = {}
        
        # ===== INTENT CLASSIFICATION METRICS =====
        all_intent_logits = torch.cat(all_intent_logits, dim=0).to(self.device)
        all_intent_labels = torch.cat(all_intent_labels, dim=0).to(self.device)
        
        # SANITY CHECK: Counter(y_true_labels) trước metric
        true_labels = all_intent_labels.cpu().numpy()
        from collections import Counter
        label_counts = Counter(true_labels)
        self.logger.info("🔍 SANITY CHECK - Validation label distribution (top 10):")
        for label_id, count in label_counts.most_common(10):
            label_name = self.id_to_intent.get(label_id, f"unknown-{label_id}")
            self.logger.info(f"  {label_name} (id={label_id}): {count}")
        
        # Check for suspicious distribution
        max_count = max(label_counts.values()) if label_counts else 0
        if max_count > 200:  # Suspicious if any class has >200 samples
            self.logger.error(f"🚨 SUSPICIOUS: Val has class with {max_count} samples - likely mapping error!")
            most_common_id = label_counts.most_common(1)[0][0]
            most_common_name = self.id_to_intent.get(most_common_id, f"unknown-{most_common_id}")
            self.logger.error(f"🚨 Most common val label: {most_common_name} (id={most_common_id})")
        
        # Calculate intent loss
        intent_loss = criterion(all_intent_logits, all_intent_labels).item()
        results['loss'] = intent_loss
        
        # Calculate intent accuracy
        _, predicted = torch.max(all_intent_logits, 1)
        intent_acc = 100 * (predicted == all_intent_labels).sum().item() / all_intent_labels.size(0)
        results['acc'] = intent_acc
        
        # Intent classification report
        predictions = predicted.cpu().numpy()
        
        # Get unique labels actually present in the data
        unique_labels = sorted(set(true_labels) | set(predictions))
        if not unique_labels:
            self.logger.error("❌ No valid labels found in validation data")
            return {'loss': intent_loss, 'acc': intent_acc, 'f1': 0.0}
            
        # Get target names for report
        target_names = []
        for i in unique_labels:
            if i in self.id_to_intent:
                target_names.append(self.id_to_intent[i])
            else:
                target_names.append(f"unknown-{i}")
        
        # Generate and log classification report
        try:
            report = classification_report(
                true_labels, 
                predictions, 
                labels=unique_labels,
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
            
            # Extract and log key metrics
            macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)
            weighted_f1 = report.get('weighted avg', {}).get('f1-score', 0.0)
            
            self.logger.info(f"📊 Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
            
            # Log per-class metrics for important classes
            self.logger.info("\n📊 Per-class performance:")
            for label in sorted(report.keys()):
                if label not in ['macro avg', 'weighted avg', 'accuracy']:
                    metrics = report[label]
                    self.logger.info(f"  {label}: F1={metrics['f1-score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, n={metrics['support']}")
            
            # Log confusion matrix summary
            confusion = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
            for true_idx, pred_idx in zip(true_labels, predictions):
                true_pos = np.where(unique_labels == true_idx)[0][0]
                pred_pos = np.where(unique_labels == pred_idx)[0][0]
                confusion[true_pos, pred_pos] += 1
                
            self.logger.info("\n📊 Most confused classes:")
            confusion_pairs = []
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    if i != j and confusion[i, j] > 0:
                        true_name = target_names[i]
                        pred_name = target_names[j]
                        confusion_pairs.append((true_name, pred_name, confusion[i, j]))
            
            for true_name, pred_name, count in sorted(confusion_pairs, key=lambda x: x[2], reverse=True)[:5]:
                self.logger.info(f"  {true_name} → {pred_name}: {count} times")
            
            # Intent F1 score
            intent_f1 = weighted_f1
            results['f1'] = intent_f1
            results['intent_f1'] = intent_f1
            
            self.logger.info(f"📊 Intent - Loss: {intent_loss:.4f}, Acc: {intent_acc:.2f}%, F1: {intent_f1:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating intent F1: {str(e)}")
            results['f1'] = 0.0
            results['intent_f1'] = 0.0
        
        # ===== ENTITY EXTRACTION METRICS (NER) =====
        if all_entity_predictions and all_entity_true_labels:
            try:
                # Use seqeval with zero_division=0 to handle missing labels
                entity_f1 = seqeval_f1_score(all_entity_true_labels, all_entity_predictions, zero_division=0)
                entity_precision = seqeval_precision_score(all_entity_true_labels, all_entity_predictions, zero_division=0)
                entity_recall = seqeval_recall_score(all_entity_true_labels, all_entity_predictions, zero_division=0)
                
                results['entity_f1'] = entity_f1
                results['entity_precision'] = entity_precision
                results['entity_recall'] = entity_recall
                
                self.logger.info(f"📊 Entity - P: {entity_precision:.4f}, R: {entity_recall:.4f}, F1: {entity_f1:.4f}")
                
                # Detailed entity report with zero_division=0
                entity_report = seqeval_classification_report(
                    all_entity_true_labels, 
                    all_entity_predictions,
                    output_dict=True,
                    zero_division=0
                )
                self.logger.info("📊 Entity Classification Report:")
                for entity_type, metrics in entity_report.items():
                    if isinstance(metrics, dict) and entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                        p = metrics.get('precision', 0)
                        r = metrics.get('recall', 0)
                        f = metrics.get('f1-score', 0)
                        s = metrics.get('support', 0)
                        if s > 0:  # Only show non-zero support
                            self.logger.info(f"  {entity_type}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, S={s}")
                        
            except Exception as e:
                self.logger.error(f"❌ Error calculating entity metrics: {str(e)}")
                results['entity_f1'] = 0.0
                results['entity_precision'] = 0.0
                results['entity_recall'] = 0.0
        else:
            results['entity_f1'] = 0.0
            results['entity_precision'] = 0.0
            results['entity_recall'] = 0.0
        
        # ===== VALUE EXTRACTION METRICS =====
        if all_value_predictions and all_value_true_labels:
            try:
                # Use seqeval with zero_division=0 to handle missing labels
                value_f1 = seqeval_f1_score(all_value_true_labels, all_value_predictions, zero_division=0)
                value_precision = seqeval_precision_score(all_value_true_labels, all_value_predictions, zero_division=0)
                value_recall = seqeval_recall_score(all_value_true_labels, all_value_predictions, zero_division=0)
                
                results['value_f1'] = value_f1
                results['value_precision'] = value_precision
                results['value_recall'] = value_recall
                
                self.logger.info(f"📊 Value - P: {value_precision:.4f}, R: {value_recall:.4f}, F1: {value_f1:.4f}")
                
                # Detailed value report with zero_division=0
                value_report = seqeval_classification_report(
                    all_value_true_labels, 
                    all_value_predictions,
                    output_dict=True,
                    zero_division=0
                )
                self.logger.info("📊 Value Classification Report:")
                for value_type, metrics in value_report.items():
                    if isinstance(metrics, dict) and value_type not in ['micro avg', 'macro avg', 'weighted avg']:
                        p = metrics.get('precision', 0)
                        r = metrics.get('recall', 0)
                        f = metrics.get('f1-score', 0)
                        s = metrics.get('support', 0)
                        if s > 0:  # Only show non-zero support
                            self.logger.info(f"  {value_type}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, S={s}")
                        
            except Exception as e:
                self.logger.error(f"❌ Error calculating value metrics: {str(e)}")
                results['value_f1'] = 0.0
                results['value_precision'] = 0.0
                results['value_recall'] = 0.0
        else:
            results['value_f1'] = 0.0
            results['value_precision'] = 0.0
            results['value_recall'] = 0.0
        
        # ===== OVERALL SUMMARY =====
        overall_f1 = (results.get('intent_f1', 0) + results.get('entity_f1', 0) + results.get('value_f1', 0)) / 3
        results['overall_f1'] = overall_f1
        
        self.logger.info(f"📊 Overall F1: {overall_f1:.4f}")
        
        return results
    
    def save_model(self, epoch: Optional[int] = None, is_best: bool = False) -> None:
        """Save model with better organization and metadata"""
        # Create base directory
        save_dir = f"models/phobert_{self.config.model_size}_intent_model"
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine filename
        if epoch is not None:
            filename = f"model_epoch_{epoch}"
            if is_best:
                filename += "_best"
            filename += ".pth"
        else:
            filename = "model_final.pth"
        
        # Create full checkpoint with metadata
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent,
                'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')},
                'intent_config': {k: v for k, v in vars(self.intent_config).items() if not k.startswith('_')},
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'is_best': is_best,
                'model_size': self.config.model_size,
                'model_name': self.config.model_name,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None
            }
            
            # Add multi-task mappings if enabled
            if self.enable_multi_task:
                checkpoint.update({
                    'entity_to_id': getattr(self, 'entity_to_id', {}),
                    'id_to_entity': getattr(self, 'id_to_entity', {}),
                    'value_to_id': getattr(self, 'value_to_id', {}),
                    'id_to_value': getattr(self, 'id_to_value', {}),
                    'command_to_id': getattr(self, 'command_to_id', {}),
                    'id_to_command': getattr(self, 'id_to_command', {}),
                    'entity_config': {k: v for k, v in vars(self.entity_config).items() if not k.startswith('_')},
                    'value_config': {k: v for k, v in vars(self.value_config).items() if not k.startswith('_')},
                    'command_config': {k: v for k, v in vars(self.command_config).items() if not k.startswith('_')},
                    'enable_multi_task': self.enable_multi_task
                })
            
            # Save the model
            torch.save(checkpoint, f"{save_dir}/{filename}", _use_new_zipfile_serialization=True)
            
            # Save tokenizer separately
            self.tokenizer.save_pretrained(save_dir)
            
            # Save config as readable JSON for easier inspection
            config_data = {
                'model_size': self.config.model_size,
                'model_name': self.config.model_name,
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent,
                'max_length': self.config.max_length,
                'num_intents': self.intent_config.num_intents,
                'timestamp': datetime.now().isoformat(),
                'torch_version': torch.__version__
            }
            
            with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Model saved to {save_dir}/{filename}")
            
            # For best model, also save a specifically named version for easier loading
            if is_best:
                best_path = f"{save_dir}/model_best.pth"
                torch.save(checkpoint, best_path, _use_new_zipfile_serialization=True)
                self.logger.info(f"💾 Best model also saved to {best_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Enhanced main function with better error handling"""
    print("🚀 Starting GPU-optimized training with PhoBERT")
    
    # Clear GPU cache before starting (only if CUDA is actually available)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        except (RuntimeError, AssertionError) as e:
            print(f"⚠️ Could not access GPU properties: {e}")
            print("🖥️ Continuing with CPU training...")
    else:
        print("🖥️ Using CPU training (CUDA not available)")
    
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Load configs
        config = ModelConfig()
        intent_config = IntentConfig()
        entity_config = EntityConfig()
        value_config = ValueConfig()
        command_config = CommandConfig()
        
        # Enable multi-task learning
        enable_multi_task = True
        
        print(f"📋 Model: {config.model_name}")
        print(f"📋 Model size: {config.model_size}")
        print(f"📋 Max length: {config.max_length}")
        print(f"📋 Batch size: {config.batch_size}")
        print(f"📋 Learning rate: {config.learning_rate}")
        print(f"📋 Epochs: {config.num_epochs}")
        print(f"📋 Freeze layers: {config.freeze_layers}")
        print(f"📋 Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"📋 Multi-task learning: {enable_multi_task}")
        
        if enable_multi_task:
            print(f"📋 Intent labels: {len(intent_config.intent_labels)}")
            print(f"📋 Entity labels: {len(entity_config.entity_labels)}")
            print(f"📋 Value labels: {len(value_config.value_labels)}")
            print(f"📋 Command labels: {len(command_config.command_labels)}")
        
        # Load training config
        from src.training.configs.config import TrainingConfig
        training_config = TrainingConfig()
        
        # Create trainer
        trainer = GPUTrainer(config, intent_config, entity_config, value_config, 
                           command_config, training_config, enable_multi_task)
        
        # Load data with fallback paths
        dataset_files = [
            "src/data/raw/elderly_command_dataset_expanded.json",
            "src/data/raw/elderly_command_dataset_reduced.json",
            "data/elderly_command_dataset.json"
        ]
        
        dataset_file = None
        for file_path in dataset_files:
            if os.path.exists(file_path):
                dataset_file = file_path
                break
        
        if dataset_file is None:
            raise FileNotFoundError("Could not find any dataset file in the expected paths")
        
        print(f"📂 Using dataset: {dataset_file}")
        
        # Load data and train
        trainer.load_data(dataset_file)
        training_results = trainer.train()
        
        # Print final results
        print("\n🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {training_results['best_val_acc']:.2f}%")
        print(f"🏆 Best validation F1 score: {training_results['best_val_f1']:.4f}")
        print(f"⏱️ Total training time: {training_results['training_time_minutes']:.2f} minutes")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()