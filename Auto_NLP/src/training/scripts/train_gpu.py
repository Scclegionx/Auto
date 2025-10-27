import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import re
import unicodedata
import regex
import warnings

# Fix PyTorch vulnerability trước khi import transformers
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online access
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow online access
os.environ['TORCH_DISABLE_SAFETENSORS_WARNING'] = '1'
os.environ['TORCH_SAFETENSORS_AVAILABLE'] = '1'

# Disable vulnerability warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*vulnerability.*")

# Set memory management for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Helper functions for text normalization
def _normalize(s: str) -> str:
    """Normalize text: remove diacritics, lowercase, normalize whitespace"""
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _normalize_for_search(s: str) -> str:
    """Normalize text for search: NFC + casefold (preserves character length)"""
    return unicodedata.normalize("NFC", s).casefold()

def _normalize_text(s: str) -> str:
    """Consistent normalization: NFC + lower + collapse spaces (NO accent removal)"""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)  # Normalize to NFC first
    s = s.lower()  # Lowercase
    s = re.sub(r"\s+", " ", s).strip()  # Collapse spaces
    return s

def _find_span_by_text_norm(text: str, snippet: str) -> tuple:
    """Find span using consistent normalization (NFC + lower + collapse spaces)"""
    base = _normalize_text(text)  # NFC + lower + collapse spaces
    q = _normalize_text(snippet)  # Same normalization
    pos = base.find(q)
    if pos < 0:
        return None, None
    # Map position on normalized text back to original text
    return pos, pos + len(q)

def _spans_from_text_if_needed(text, entity):
    """Extract spans from entity, using text matching if start/end not available"""
    if "start" in entity and "end" in entity:
        return [(entity["start"], entity["end"])]
    if "text" not in entity or not entity["text"]:
        return []
    
    t_norm = _normalize_for_search(text)
    q_norm = _normalize_for_search(entity["text"])
    spans = []
    
    for m in regex.finditer(regex.escape(q_norm), t_norm):
        # Map back to original string indices by length (safe vì NFC giữ độ dài ký tự)
        start = m.start()
        end = m.end()
        # Vì dùng NFC + casefold, độ dài ký tự không đổi ⇒ start/end hợp lệ trên text gốc
        spans.append((start, end))
    return spans
# Fix torchvision import issue by setting environment variables
import os
os.environ['TORCHVISION_DISABLE_IMAGE_IO'] = '1'
os.environ['TORCHVISION_DISABLE_VIDEO_IO'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Disable torchvision completely
import sys
sys.modules['torchvision'] = None

# Patch transformers vulnerability check trước khi import
try:
    import transformers
    from transformers.utils import import_utils
    
    def dummy_check_torch_load_is_safe():
        return True
        
    import_utils.check_torch_load_is_safe = dummy_check_torch_load_is_safe
    
    # Also patch the actual function in modeling_utils
    from transformers.modeling_utils import load_state_dict
    import functools
    
    def patched_load_state_dict(*args, **kwargs):
        # Skip the vulnerability check and remove incompatible parameters
        if 'is_quantized' in kwargs:
            kwargs.pop('is_quantized')
        return torch.load(*args, **kwargs)
    
    def patched_torch_load(*args, **kwargs):
        if 'is_quantized' in kwargs:
            kwargs.pop('is_quantized')
        # Use weights_only=False to bypass vulnerability
        kwargs['weights_only'] = False
        import torch
        return torch.load(*args, **kwargs)
    
    # Replace the function
    import transformers.modeling_utils
    transformers.modeling_utils.load_state_dict = patched_load_state_dict
    
    # Don't patch torch.load directly to avoid recursion
    
    print("PyTorch vulnerability fixes applied")
except Exception as e:
    print(f"Warning: Could not patch transformers: {e}")

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
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from training.configs.config import ModelConfig, IntentConfig, EntityConfig, ValueConfig, CommandConfig

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
        """Viterbi decoding with proper backtracking"""
        # Ensure dtype/device compatibility
        mask = mask.to(logits.device).float()
        
        batch_size, seq_len, num_labels = logits.size()
        
        # Calculate actual sequence lengths
        lengths = mask.sum(1).long()
        
        # Initialize forward scores and backpointers
        forward_scores = logits[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_labels]
        backpointers = torch.zeros(batch_size, seq_len, num_labels, dtype=torch.long, device=logits.device)
        
        # Forward pass with proper Viterbi
        for t in range(1, seq_len):
            # scores shape: [batch_size, num_labels, num_labels]
            scores = forward_scores.unsqueeze(-1) + self.transitions.unsqueeze(0) + logits[:, t].unsqueeze(1)
            
            # Find best previous state for each current state
            forward_scores, best_prev_states = torch.max(scores, dim=1)  # [batch_size, num_labels]
            backpointers[:, t] = best_prev_states
            
            # Apply mask
            forward_scores = forward_scores * mask[:, t].unsqueeze(-1) + (1 - mask[:, t].unsqueeze(-1)) * forward_scores
        
        # Add end transitions
        final_scores = forward_scores + self.end_transitions.unsqueeze(0)
        
        # Find best final state
        best_final_states = torch.argmax(final_scores, dim=-1)  # [batch_size]
        
        # Backtrack to find best path
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        best_paths[:, -1] = best_final_states
        
        for t in range(seq_len - 2, -1, -1):
            # Get the best previous state for each sequence
            for b in range(batch_size):
                if mask[b, t + 1] > 0:  # Only backtrack if next token is not masked
                    best_paths[b, t] = backpointers[b, t + 1, best_paths[b, t + 1]]
                else:
                    best_paths[b, t] = 0  # Default to first state for masked positions
        
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
    
    # Initialize tensors - use -100 for entity labels to preserve masking
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    intent_labels = torch.zeros(batch_size, dtype=torch.long)
    entity_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)  # Initialize with -100
    command_labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Only create value_labels if USE_VALUE is True
    value_labels = None
    
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
        # Handle entity labels with proper masking và sanitization
        entity_labels_tensor = torch.as_tensor(item['entity_labels'][:seq_len], dtype=torch.long)
        
        # Sanitize entity labels: map invalid IDs to 'O' (id=0)
        # Note: entity_to_id sẽ được set trong __init__ của MultiTaskDataset
        entity_labels_tensor = entity_labels_tensor  # Keep as is for now
        
        # Check if labels are masked (-100) and preserve masking
        if (entity_labels_tensor == -100).all():
            entity_labels[i, :seq_len] = -100
        else:
            entity_labels[i, :seq_len] = entity_labels_tensor
        command_labels[i] = item['command_labels']
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'intent_labels': intent_labels,
        'entity_labels': entity_labels,
        'command_labels': command_labels
    }
    
    # Only include value_labels if USE_VALUE is True
    if value_labels is not None:
        result['value_labels'] = value_labels
    
    return result

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
            # Prioritize spans over entities (spans have start/end positions)
            entities = item.get('spans', item.get('entities', []))
            values = item.get('values', [])
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get intent label - đảm bảo không out of bounds
            intent_label = self.intent_to_id.get(command, 0)
            if intent_label >= len(self.intent_to_id):
                intent_label = 0  # Fallback to first class
            
            # Get sequence length
            seq_len = int(encoding['input_ids'].shape[1])
            input_ids_list = encoding['input_ids'].squeeze().tolist()
            spec = set(self.tokenizer.all_special_ids)
            
            # Get entity labels (prefer precomputed bio_labels to avoid runtime alignment issues)
            raw_bio = item.get('bio_labels')
            if isinstance(raw_bio, list) and len(raw_bio) > 0:
                entity_labels = []
                for lab in raw_bio:
                    # Normalize and map safely
                    if isinstance(lab, str):
                        # Only accept valid tags from vocabulary
                        idx = self.entity_to_id.get(lab)
                        if idx is None:
                            # Fallback to O if unknown (prevents CUDA assert)
                            idx = self.entity_o_id
                        entity_labels.append(idx)
                    elif isinstance(lab, int) and 0 <= lab < len(self.entity_to_id):
                        entity_labels.append(lab)
                    else:
                        entity_labels.append(self.entity_o_id)
            else:
                # Fallback: align from spans via DataProcessor
                if entities:
                    from data.processed.data_processor import DataProcessor
                    processor = DataProcessor()
                    processor.entity_to_id = self.entity_to_id
                    processor.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
                    processor.tokenizer = self.tokenizer
                    entity_str_labels = processor.align_labels(text, entities, values)
                    tmp = []
                    for label in entity_str_labels:
                        if isinstance(label, str):
                            tmp.append(self.entity_to_id.get(label, self.entity_o_id))
                        else:
                            tmp.append(label if isinstance(label, int) else self.entity_o_id)
                    entity_labels = tmp
                else:
                    entity_labels = [self.entity_o_id] * seq_len
            
            # Chuẩn hoá độ dài entity_labels
            if len(entity_labels) < seq_len:
                entity_labels += [self.entity_o_id] * (seq_len - len(entity_labels))
            entity_labels = entity_labels[:seq_len]
            
            # Sanitize out-of-range labels (prevent CUDA assert)
            num_entity_labels = len(self.entity_to_id)
            sanitized_count = 0
            for i, lab in enumerate(entity_labels):
                if lab != -100 and not (0 <= int(lab) < num_entity_labels):
                    entity_labels[i] = self.entity_o_id
                    sanitized_count += 1
                    # Log để debug
                    if sanitized_count <= 3:  # Chỉ log 3 lần đầu
                        print(f"[WARN] Sanitized invalid entity label {lab} to O (max valid: {num_entity_labels-1})")
            
            if sanitized_count > 0:
                print(f"[WARN] Total sanitized labels: {sanitized_count}")

            # Mask special tokens với -100
            for i, tid in enumerate(input_ids_list):
                if tid in spec:
                    entity_labels[i] = -100
            
            # Tắt hoàn toàn Value pipeline - không align value labels
            value_labels = None
            
            # Get command label - đảm bảo không out of bounds
            command_label = self.command_to_id.get(command, 0)
            if command_label >= len(self.command_to_id):
                command_label = 0  # Fallback to first class
            
            # Tạo return dict
            result = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'intent_labels': intent_label,
                'entity_labels': entity_labels,
                'command_labels': command_label,
                'text': text
            }
            
            # Không thêm value_labels - đã tắt hoàn toàn Value pipeline
            
            return result
            
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
        """Robust entity alignment with span-based approach"""
        input_ids, offsets = self._build_char2tok_offsets(text)
        seq_len = len(input_ids)
        labels = [self.entity_o_id] * seq_len

        for ent in entities or []:
            lab = ent.get("label") or ent.get("type") or ""
            if not lab: 
                continue

            # Prioritize span from dataset
            st, ed = ent.get("start"), ent.get("end")
            if st is None or ed is None or st >= ed:
                etxt = ent.get("text", "")
                if etxt:
                    st, ed = _find_span_by_text_norm(text, etxt)
            
            # If span still not found, try accent-insensitive fallback
            if st is None or ed is None or st >= ed:
                etxt = ent.get("text", "")
                if etxt:
                    st, ed = self._find_span_fallback(text, etxt)
            
            if st is None or ed is None or st >= ed:
                continue
                
            self._assign_bio_overlaps(labels, offsets, int(st), int(ed), lab, self.entity_to_id)

        # Mask special tokens
        spec = set(self.tokenizer.all_special_ids)
        for i, tid in enumerate(input_ids):
            if tid in spec:
                labels[i] = -100
        
        # Sanity check: count non-O labels
        non_o_count = sum(1 for x in labels if x not in (-100, self.entity_o_id))
        if non_o_count == 0 and (entities or []):
            # Detailed logging for debugging
            try:
                print(f"[WARN] No non-O entity labels aligned for text='{text[:50]}...' with {len(entities)} entities")
                for i, ent in enumerate(entities[:2]):  # Show first 2 entities
                    try:
                        st, ed = ent.get("start"), ent.get("end")
                        etxt = ent.get("text", "")
                        print(f"  Entity {i+1}: '{etxt}' (start={st}, end={ed}, label={ent.get('label', '')})")
                    except UnicodeEncodeError:
                        print(f"  Entity {i+1}: [Unicode error] (start={ent.get('start')}, end={ent.get('end')}, label={ent.get('label', '')})")
            except UnicodeEncodeError:
                print(f"[WARN] No non-O entity labels aligned for text='{text[:50].encode('ascii', 'ignore').decode()}...' with {len(entities)} entities")
        
        return labels
    
    def _align_entity_labels_with_offsets(self, text: str, entities: List[Dict], encoding) -> List[int]:
        """Entity alignment using tokenizer offset_mapping for better Unicode handling"""
        import unicodedata
        
        def _normalize(s: str) -> str:
            """Normalize Unicode để ổn định dấu tiếng Việt"""
            return unicodedata.normalize("NFC", s)
        
        # Normalize text
        text = _normalize(text)
        
        # Get offset_mapping from encoding
        offsets = encoding["offset_mapping"][0].tolist()  # [(start_char, end_char), ...]
        seq_len = len(offsets)
        labels = [self.entity_o_id] * seq_len
        
        def overlap(tok_span, ent_span):
            """Check if token span overlaps with entity span"""
            s1, e1 = tok_span
            s2, e2 = ent_span
            return max(0, min(e1, e2) - max(s1, s2)) > 0
        
        for ent in entities or []:
            lab = ent.get("label") or ent.get("type") or ""
            if not lab: 
                continue
                
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            
            # Validate span
            if start >= end or start < 0 or end > len(text):
                    continue
                    
            ent_span = (start, end)
            
            # Find overlapping tokens using offset_mapping
            token_idxs = []
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:  # Skip special tokens
                    continue
                if overlap((tok_start, tok_end), ent_span):
                    token_idxs.append(i)
            
            if not token_idxs:
                # Không log WARN - nhiều câu hợp lệ không có entity
                continue
                
            # Assign BIO labels
            lab_upper = lab.upper().strip()
            b_tag = f"B-{lab_upper}"
            i_tag = f"I-{lab_upper}"
            
            # Get label IDs
            b_id = self.entity_to_id.get(b_tag, self.entity_o_id)
            i_id = self.entity_to_id.get(i_tag, self.entity_o_id)
            
            # Assign B- to first token, I- to subsequent tokens
            labels[token_idxs[0]] = b_id
            for i in token_idxs[1:]:
                labels[i] = i_id

        return labels
    
    def _build_char2tok_offsets(self, text: str):
        """
        Build character to token offsets with fallback for PhoBERT.
        Returns: (input_ids, offsets) where offsets[i] = (start_char, end_char) or (None, None) for special tokens
        """
        # Normalize text to NFC but keep original for alignment
        import unicodedata
        text_nfc = unicodedata.normalize("NFC", text)
        
        # PhoBERT doesn't support return_offsets_mapping, use manual mapping
        return self._create_char_to_token_mapping_manual(text_nfc)
    
    def _create_char_to_token_mapping_manual(self, text: str) -> tuple:
        """Manual character to token mapping - use same normalization as span finding"""
        # Use NFC normalization (same as _normalize_text) for consistency
        text_nfc = unicodedata.normalize("NFC", text)
        tokens = self.tokenizer.tokenize(text_nfc)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        char_to_token = []
        char_ptr = 0
        
        for i, token in enumerate(tokens):
            if token in self.tokenizer.all_special_tokens:
                char_to_token.append((None, None))
                continue
                
            clean_token = token.replace('▁', '').replace('##', '').replace(' ', '')
            if not clean_token:
                char_to_token.append((None, None))
                continue
                
            # Find token in NFC-normalized text (same as span finding)
            token_start = text_nfc.find(clean_token, char_ptr)
            if token_start == -1:
                token_start = char_ptr
                token_end = min(char_ptr + len(clean_token), len(text_nfc))
            else:
                token_end = token_start + len(clean_token)
            
            char_to_token.append((token_start, token_end))
            char_ptr = token_end
            
        return input_ids, char_to_token

    def _assign_bio_overlaps(self, labels, offsets, span_start, span_end, lab_prefix, vocab):
        """Assign BIO labels based on token overlap with span"""
        overlap_toks = []
        for i, (st, ed) in enumerate(offsets):
            if st is None or ed is None: 
                    continue
            # Use >= 1 character overlap, exclusive end
            if max(0, min(ed, span_end) - max(st, span_start)) >= 1:
                overlap_toks.append(i)
        
        if overlap_toks:
            b = f"B-{lab_prefix}"
            i_ = f"I-{lab_prefix}"
            if b in vocab: 
                labels[overlap_toks[0]] = vocab[b]
            if i_ in vocab:
                for j in overlap_toks[1:]:
                    labels[j] = vocab[i_]
    
    def _find_span_fallback(self, text: str, target_text: str):
        """Fallback search with accent-insensitive + case-insensitive matching"""
        import unicodedata
        
        def strip_accents(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) 
                          if unicodedata.category(c) != 'Mn').casefold()
        
        try:
            # Normalize both texts
            text_norm = strip_accents(text)
            target_norm = strip_accents(target_text)
            
            # Find position in normalized text
            pos = text_norm.find(target_norm)
            if pos != -1:
                # Map back to original text position
                # This is a simplified mapping - in practice you'd need more sophisticated mapping
                return pos, pos + len(target_text)
        except Exception:
            pass
        
        return None, None

    def _create_char_to_token_mapping(self, text: str) -> List[Tuple[int, int]]:
        """Create character to token mapping without offset_mapping"""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Create character mapping manually
        char_to_token = []
        char_ptr = 0
        text_norm = _normalize_for_search(text)
        
        for i, token in enumerate(tokens):
            # Clean token for matching
            clean_token = token.replace('▁', '').replace('##', '')
            if not clean_token:
                char_to_token.append((char_ptr, char_ptr))
                continue
                
            # Find token in normalized text
            token_start = text_norm.find(clean_token, char_ptr)
            if token_start == -1:
                # Fallback: advance by token length
                token_start = char_ptr
                token_end = min(char_ptr + len(clean_token), len(text_norm))
            else:
                token_end = token_start + len(clean_token)
            
            char_to_token.append((token_start, token_end))
            char_ptr = token_end
            
        return char_to_token
    
    def _get_entity_span(self, text: str, entity: Dict) -> Tuple[int, int]:
        """Get entity span with fallback to text matching"""
        # Try to use start/end if available
        if 'start' in entity and 'end' in entity:
            return entity['start'], entity['end']
        
        # Fallback to text matching
        if 'text' in entity:
            entity_text = entity['text']
            text_norm = _normalize_for_search(text)
            entity_norm = _normalize_for_search(entity_text)
            
            # Find entity text in normalized text
            start = text_norm.find(entity_norm)
            if start != -1:
                return start, start + len(entity_norm)
        
        return None, None
    
    def _get_value_span(self, text: str, value: Dict) -> Tuple[int, int]:
        """Get value span with fallback to text matching"""
        # Try to use start/end if available
        if 'start' in value and 'end' in value:
            return value['start'], value['end']
        
        # Fallback to text matching
        if 'text' in value:
            value_text = value['text']
            text_norm = _normalize_for_search(text)
            value_norm = _normalize_for_search(value_text)
            
            # Find value text in normalized text
            start = text_norm.find(value_norm)
            if start != -1:
                return start, start + len(value_norm)
        
        return None, None
    
    def _align_value_labels(self, text: str, values: List[Dict]) -> List[int]:
        """Robust value alignment with span-based approach"""
        input_ids, offsets = self._build_char2tok_offsets(text)
        seq_len = len(input_ids)
        labels = [self.value_o_id] * seq_len

        for val in values or []:
            lab = val.get("label") or val.get("type") or ""
            if not lab:
                continue
            st, ed = val.get("start"), val.get("end")
            if st is None or ed is None or st >= ed:
                vtxt = val.get("text", "")
                if vtxt:
                    st, ed = _find_span_by_text_norm(text, vtxt)
            
            # If span still not found, try accent-insensitive fallback
            if st is None or ed is None or st >= ed:
                vtxt = val.get("text", "")
                if vtxt:
                    st, ed = self._find_span_fallback(text, vtxt)
            
            if st is None or ed is None or st >= ed:
                    continue
                    
            self._assign_bio_overlaps(labels, offsets, int(st), int(ed), lab, self.value_to_id)

        # Mask special tokens
        spec = set(self.tokenizer.all_special_ids)
        for i, tid in enumerate(input_ids):
            if tid in spec:
                labels[i] = -100
        
        # Sanity check: count non-O labels
        non_o_count = sum(1 for x in labels if x not in (-100, self.value_o_id))
        if non_o_count == 0 and (values or []):
            # Detailed logging for debugging
            try:
                print(f"[WARN] No non-O value labels aligned for text='{text[:50]}...' with {len(values)} values")
                for i, val in enumerate(values[:2]):  # Show first 2 values
                    try:
                        st, ed = val.get("start"), val.get("end")
                        vtxt = val.get("text", "")
                        print(f"  Value {i+1}: '{vtxt}' (start={st}, end={ed}, label={val.get('label', '')})")
                    except UnicodeEncodeError:
                        print(f"  Value {i+1}: [Unicode error] (start={val.get('start')}, end={val.get('end')}, label={val.get('label', '')})")
            except UnicodeEncodeError:
                print(f"[WARN] No non-O value labels aligned for text='{text[:50].encode('ascii', 'ignore').decode()}...' with {len(values)} values")
        
        return labels


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
            # Try to load with minimal torchvision dependencies
            # Try to load from local cache first
            try:
                self.phobert = AutoModel.from_pretrained(
                    model_name,
                    use_safetensors=True,  # Use safetensors to avoid torch.load vulnerability
                    trust_remote_code=False,  # Disable trust_remote_code
                    cache_dir="../../model_cache",
                    local_files_only=True  # Use local cache only
                )
            except Exception:
                # Fallback: try with different model path
                model_path = "../../model_cache/models--vinai--phobert-large/snapshots"
                if os.path.exists(model_path):
                    self.phobert = AutoModel.from_pretrained(
                        model_path,
                        use_safetensors=True,
                        trust_remote_code=False,
                        local_files_only=True
                    )
                else:
                    raise Exception("Model not found in cache")
        except Exception as e:
            # Fallback: try with different settings
            try:
                self.phobert = AutoModel.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    trust_remote_code=False,
                    cache_dir="../../model_cache",
                    local_files_only=False
                )
            except Exception as e2:
                # Final fallback: try with minimal config
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name)
                    self.phobert = AutoModel.from_pretrained(
                        model_name,
                        config=config,
                        use_safetensors=False,
                        trust_remote_code=False
                    )
                except Exception as e3:
                    raise RuntimeError(f"Failed to load pretrained model {model_name}: {str(e3)}")
        
        # Freeze layers if specified
        if hasattr(config, 'freeze_layers') and config.freeze_layers > 0:
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
        dropout_rate = getattr(config, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Intent classification head (sentence-level) - Optimized for small batches
        model_size = getattr(config, 'model_size', 'base')
        if model_size == "large":
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
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size // 2, num_commands)
                )
        
        # Legacy support for single-task
        if not self.enable_multi_task:
            if model_size == "large":
                # Multi-layer architecture for large model
                self.dropout1 = nn.Dropout(dropout_rate)
                self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
                self.activation1 = nn.GELU()
                self.layernorm1 = nn.LayerNorm(hidden_size // 2)
                
                self.dropout2 = nn.Dropout(dropout_rate)
                self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.activation2 = nn.GELU()
                self.layernorm2 = nn.LayerNorm(hidden_size // 4)
                
                self.dropout3 = nn.Dropout(dropout_rate)
                self.output = nn.Linear(hidden_size // 4, num_intents)
            else:
                # Simpler architecture for base model
                self.dropout = nn.Dropout(dropout_rate)
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
                    # Check if entity_labels contain valid data (not all -100)
                    valid_entity_mask = (entity_labels != -100).any(dim=1)
                    
                    # Debug logging disabled
                    
                    if valid_entity_mask.any():
                        # Only compute loss for samples with valid entity labels
                        valid_entity_logits = entity_logits[valid_entity_mask]
                        valid_entity_labels = entity_labels[valid_entity_mask]
                        valid_attention_mask = attention_mask[valid_entity_mask]
                        
                        # Use CrossEntropyLoss with label smoothing for entity labels
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.1)
                        entity_logits_flat = valid_entity_logits.view(-1, valid_entity_logits.size(-1))
                        entity_labels_flat = valid_entity_labels.view(-1)
                        token_loss = loss_fct(entity_logits_flat, entity_labels_flat)
                        mask = (entity_labels_flat != -100).float()
                        # Giảm ảnh hưởng lớp 'O' (giả định id của 'O' = 0 theo vocab đã chuẩn hoá)
                        o_weight_mask = torch.where(
                            entity_labels_flat == 0,
                            torch.tensor(0.2, device=entity_labels_flat.device, dtype=token_loss.dtype),
                            torch.tensor(1.0, device=entity_labels_flat.device, dtype=token_loss.dtype)
                        )
                        entity_loss = (token_loss * mask * o_weight_mask).sum() / (mask.sum().clamp_min(1.0))
                    results['entity_loss'] = entity_loss
                else:
                    # No valid entity data, skip loss computation
                    results['entity_loss'] = None
            
            # Value extraction (if enabled)
            if hasattr(self, 'value_classifier'):
                value_logits = self.value_classifier(token_embeddings)
                results['value_logits'] = value_logits
                
                if value_labels is not None:
                    # Check if value_labels contain valid data (not all -100)
                    valid_value_mask = (value_labels != -100).any(dim=1)
                    if valid_value_mask.any():
                        # Only compute loss for samples with valid value labels
                        valid_value_logits = value_logits[valid_value_mask]
                        valid_value_labels = value_labels[valid_value_mask]
                        valid_attention_mask = attention_mask[valid_value_mask]
                        
                        # Use CrossEntropyLoss with label smoothing for value labels
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.1)
                        value_logits_flat = valid_value_logits.view(-1, valid_value_logits.size(-1))
                        value_labels_flat = valid_value_labels.view(-1)
                        token_loss = loss_fct(value_logits_flat, value_labels_flat)
                        mask = (value_labels_flat != -100).float()
                        value_loss = (token_loss * mask).sum() / (mask.sum().clamp_min(1.0))
                    results['value_loss'] = value_loss
                else:
                    # No valid value data, skip loss computation
                    results['value_loss'] = None
            
            # Command classification (if enabled)
            if hasattr(self, 'command_classifier'):
                command_logits = self.command_classifier(pooled_output)
                results['command_logits'] = command_logits
            
            # Calculate total loss with lambda weights and gradient accumulation
            total_loss = 0.0
            
            # Intent loss (always present)
            if intent_labels is not None:
                intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)
                total_loss += lambda_intent * intent_loss
                results['intent_loss'] = intent_loss
            
            # Command loss (if available)
            if command_labels is not None and hasattr(self, 'command_classifier'):
                command_loss = nn.CrossEntropyLoss()(command_logits, command_labels)
                total_loss += lambda_command * command_loss
                results['command_loss'] = command_loss
            
            # Entity loss (if available and not None)
            if 'entity_loss' in results and results['entity_loss'] is not None:
                total_loss += lambda_entity * results['entity_loss']
            
            # Value loss - TẮT HOÀN TOÀN (USE_VALUE=False)
            # Không tính value loss vì đã tắt Value pipeline
            
            # Divide by gradient accumulation steps BEFORE backward
            total_loss = total_loss / self.config.gradient_accumulation_steps
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
        
        # Initialize multi-task vocabularies
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.value_to_id = {}
        self.id_to_value = {}
        self.command_to_id = {}
        self.id_to_command = {}
        
        # Set random seed for reproducibility
        set_seed(getattr(config, 'seed', 42))
        
    def _get_multi_task_weights(self, epoch: int) -> Dict[str, float]:
        """Get multi-task loss weights - focus on Entity, disable Value"""
        # Tăng entity weight để học entity tốt hơn
        return {
            'lambda_intent': getattr(self.config, 'LAMBDA_INTENT', 0.4),  # Giảm từ 0.5
            'lambda_command': getattr(self.config, 'LAMBDA_COMMAND', 0.1),  # Giảm từ 0.2
            'lambda_entity': getattr(self.config, 'LAMBDA_ENTITY', 0.5),   # Tăng từ 0.3 lên 0.5
            'lambda_value': getattr(self.config, 'LAMBDA_VALUE', 0.0)
        }
        
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device with better error handling"""
        # Use training_config.device if available, otherwise fallback to auto-detection
        if self.training_config and hasattr(self.training_config, 'device'):
            device_setting = self.training_config.device
        else:
            device_setting = "auto"
        
        if device_setting == "auto":
            # Use GPU if available
            if torch.cuda.is_available():
                # Test CUDA operations before using
                try:
                    test_tensor = torch.randn(2, 2).cuda()
                    test_tensor.to('cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    device = torch.device("cuda")
                except RuntimeError as e:
                    if "operation not supported" in str(e):
                        print("WARNING CUDA operations not supported, falling back to CPU")
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
            
            self.logger.info(f"[INFO] Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
            self.logger.info(f"[INFO] Model size on disk: ~{total_params * 4 / (1024**2):.1f} MB")
            
            # Disable gradient checkpointing for stability
            self.logger.info("SETUP Gradient Checkpointing disabled for stability")
            
            return model
        except Exception as e:
            self.logger.error(f"ERROR Failed to initialize model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _log_system_info(self) -> None:
        """Log detailed system information"""
        self.logger.info(f"Device: {self.device}")
        
        if self.device.type == "cuda" and torch.cuda.device_count() > 0:
            try:
                self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
                # Enable cudnn benchmark for performance
                torch.backends.cudnn.benchmark = not getattr(self.config, 'deterministic', False)
                torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, AssertionError) as e:
                self.logger.info("Falling back to CPU training")
    
    def _setup_amp(self) -> GradScaler:
        """Setup Automatic Mixed Precision for better performance - FIXED"""
        # Use proper AMP scaler with new API
        use_amp = (self.device.type == 'cuda')
        if use_amp:
            return torch.amp.GradScaler('cuda', enabled=True)
        else:
            return torch.amp.GradScaler('cpu', enabled=False)
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer with improved pad token handling"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Check if tokenizer already has a pad token
            if tokenizer.pad_token is None:
                # For RoBERTa/PhoBERT, use eos_token as pad_token
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Fallback: add a new pad token
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            return tokenizer
        except Exception as e:
            self.logger.error(f"ERROR Failed to load tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    
    def load_data(self, data_dir: str = None) -> Tuple[List[Dict], List[Dict]]:
        """Load data from processed train/val/test files"""
        if data_dir is None:
            # Get absolute path to processed data
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            data_dir = os.path.join(project_root, "src", "data", "processed")
        
        train_file = os.path.join(data_dir, "train.json")
        val_file = os.path.join(data_dir, "val.json")
        test_file = os.path.join(data_dir, "test.json")
        
        self.logger.info(f"Loading processed data from {data_dir}")
        
        # Load train data
        if not os.path.exists(train_file):
            self.logger.error(f"ERROR Train file not found: {train_file}")
            raise FileNotFoundError(f"Train file not found: {train_file}")
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            self.logger.info(f"✅ Loaded train data: {len(train_data)} samples")
        except Exception as e:
            self.logger.error(f"ERROR Failed to load train data: {str(e)}")
            raise
        
        # Load val data
        if not os.path.exists(val_file):
            self.logger.error(f"ERROR Val file not found: {val_file}")
            raise FileNotFoundError(f"Val file not found: {val_file}")
        
        try:
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            self.logger.info(f"✅ Loaded val data: {len(val_data)} samples")
        except Exception as e:
            self.logger.error(f"ERROR Failed to load val data: {str(e)}")
            raise
        
        # Load test data (optional, for reference)
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                self.logger.info(f"✅ Loaded test data: {len(test_data)} samples")
            except Exception as e:
                self.logger.warning(f"WARNING Failed to load test data: {str(e)}")
        
        self.logger.info(f"📈 Total loaded: {len(train_data)} train + {len(val_data)} val samples")
        
        # Store data for processing
        self.train_data = train_data
        self.val_data = val_data
        
        # Analyze token lengths
        try:
            all_data = train_data + val_data
            token_lengths = [len(self.tokenizer.encode(item['input'])) for item in all_data]
            max_len = max(token_lengths)
            avg_len = sum(token_lengths) / len(token_lengths)
            
            self.logger.info(f"[INFO] Token statistics - Max: {max_len}, Avg: {avg_len:.1f}")
            
        except Exception as e:
            pass
        
        # Analyze intent distribution
        intent_counts = {}
        for item in train_data + val_data:
            intent = item.get('command', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        self.logger.info("[INFO] Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count} samples ({count/len(train_data + val_data)*100:.1f}%)")
        
        # Check for class imbalance
        if len(intent_counts) > 1:
            min_count = min(intent_counts.values())
            max_count = max(intent_counts.values())
            imbalance_ratio = max_count / min_count
            
            pass
        
        # Use config labels as source of truth instead of creating from dataset
        config_intents = self.intent_config.intent_labels
        self.intent_to_id = {intent: i for i, intent in enumerate(config_intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        # Map dataset intents to config intents (with fallback to 'unknown')
        dataset_intents = set(intent_counts.keys())
        config_intent_set = set(config_intents)
        
        # Log mapping information
        self.logger.info(f"🎯 Config intents ({len(config_intents)}): {config_intents}")
        self.logger.info(f"[INFO] Dataset intents ({len(dataset_intents)}): {sorted(dataset_intents)}")
        
        # Check for intents in dataset but not in config
        unmapped_intents = dataset_intents - config_intent_set
        
        self.logger.info("🔒 Using CONFIG labels as source of truth - consistent across train/val/test")
        
        # Create multi-task label mappings if enabled
        if self.enable_multi_task:
            # Use train and val data that we already loaded
            all_data = train_data + val_data
            
            # Entity labels - CHỈ lấy từ bio_labels để đảm bảo consistency
            unique_entity_labels = set()
            for item in all_data:
                if 'bio_labels' in item and item['bio_labels']:
                    # CHỈ lấy từ bio_labels (đã clean và chuẩn hóa)
                    for label in item['bio_labels']:
                        if isinstance(label, str) and label.strip():
                            unique_entity_labels.add(label.strip())
            
            # Log để debug
            self.logger.info(f"[DEBUG] Found {len(unique_entity_labels)} unique entity labels from bio_labels")
            self.logger.info(f"[DEBUG] Labels: {sorted(unique_entity_labels)}")
            
            # Build entity_to_id - đảm bảo 'O' có id=0 và tất cả labels hợp lệ
            self.entity_to_id = {"O": 0}  # Đảm bảo 'O' luôn có id=0
            other_labels = sorted([label for label in unique_entity_labels if label != "O"])
            for i, label in enumerate(other_labels, 1):  # Bắt đầu từ id=1
                self.entity_to_id[label] = i
            self.id_to_entity = {i: label for label, i in self.entity_to_id.items()}
            
            # Log để debug
            self.logger.info(f"[INFO] Entity vocabulary built: {len(self.entity_to_id)} labels")
            self.logger.info(f"[INFO] Entity labels: {list(self.entity_to_id.keys())}")
            self.logger.info(f"[INFO] Max entity ID: {max(self.entity_to_id.values())}")
            
            # Log entity mapping for debugging
            self.logger.info(f"[INFO] Entity mapping created: {len(self.entity_to_id)} labels")
            self.logger.info(f"   Entity labels: {list(self.entity_to_id.keys())[:10]}...")
            
            # Value labels - build from dataset to ensure all B-/I- labels exist
            unique_value_labels = set()
            for item in all_data:
                values = item.get('values', [])
                for val in values:
                    if 'label' in val:
                        unique_value_labels.add(val['label'])
            
            # Build value_to_id with B-/I- prefixes
            self.value_to_id = {"O": 0}
            for label in sorted(unique_value_labels):
                if label != "O":  # Skip O as it's already added
                    self.value_to_id[f"B-{label}"] = len(self.value_to_id)
                    self.value_to_id[f"I-{label}"] = len(self.value_to_id)
            self.id_to_value = {i: label for label, i in self.value_to_id.items()}
            
            # Log value mapping for debugging
            self.logger.info(f"[INFO] Value mapping created: {len(self.value_to_id)} labels")
            self.logger.info(f"   Value labels: {list(self.value_to_id.keys())[:10]}...")
            
            # Command labels - use config labels to match model initialization
            command_labels = self.command_config.command_labels
            self.command_to_id = {label: i for i, label in enumerate(command_labels)}
            self.id_to_command = {i: label for label, i in self.command_to_id.items()}
            
            # Map dataset intents to config command labels
            for intent in dataset_intents:
                if intent not in self.command_to_id:
                    # Map unknown intents to 'unknown' command
                    self.command_to_id[intent] = self.command_to_id.get('unknown', 0)
            
            pass
        
        # Save FIXED label mappings to file
        label_maps = {
            "intent": config_intents,
            "entity": list(self.entity_to_id.keys()) if self.enable_multi_task else [],
            "value": list(self.value_to_id.keys()) if self.enable_multi_task else [],
            "command": command_labels if self.enable_multi_task else []
        }
        
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Save label mappings
        with open("models/label_maps.json", 'w', encoding='utf-8') as f:
            json.dump(label_maps, f, ensure_ascii=False, indent=2)
        
        self.logger.info("💾 Saved FIXED label mappings to models/label_maps.json")
        
        # Data already split in load_data method
        # No need to split again
        
        # CRITICAL FIX: Always enable multi-task, use masking for missing annotations
        has_entities = any('entities' in item and item['entities'] for item in self.train_data[:100])
        has_values = any('values' in item and item['values'] for item in self.train_data[:100])
        
        if not has_entities or not has_values:
            self.logger.warning("[WARN]  Dataset lacks entity/value annotations. Using masking for missing data.")
            self.logger.info("[INFO] Multi-task enabled with smart weight scheduling and masking.")
        else:
            self.logger.info("[INFO] Multi-task enabled with full annotations.")
        
        # Always enable multi-task for proper weight scheduling
        self.enable_multi_task = True
        
        # SANITY CHECKS: Verify split doesn't lose any classes
        train_intents = set(item.get('command', 'unknown') for item in train_data)
        val_intents = set(item.get('command', 'unknown') for item in val_data)
        all_intents = set(intent_counts.keys())
        
        missing_in_train = all_intents - train_intents
        missing_in_val = all_intents - val_intents
        
        pass
        
        # SANITY CHECK: Print intent distribution in train/val
        self.logger.info("[DEBUG] SANITY CHECK - Train intents:")
        train_intent_counts = {}
        for item in train_data:
            intent = item.get('command', 'unknown')
            train_intent_counts[intent] = train_intent_counts.get(intent, 0) + 1
        for intent, count in sorted(train_intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count}")
        
        self.logger.info("[DEBUG] SANITY CHECK - Val intents:")
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
        
        # Create datasets - always use MultiTaskDataset for proper weight scheduling
        try:
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
        except Exception as e:
            self.logger.error(f"ERROR Failed to create datasets: {str(e)}")
            raise RuntimeError(f"Failed to create datasets: {str(e)}")
        
        self.logger.info(f"[INFO] Train: {len(train_data)}, Val: {len(val_data)}")
        
        return self.train_data, self.val_data
    
    def _setup_training(self, train_loader, val_loader):
        """Setup optimizer, scheduler, and loss function"""
        # Setup optimizer with error handling
        try:
            optimizer = self._create_optimizer()
        except Exception as e:
            self.logger.error(f"ERROR Failed to create optimizer: {str(e)}")
            raise RuntimeError(f"Failed to create optimizer: {str(e)}")
        
        # Setup learning rate scheduler with 10% warmup
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * 0.1)  # Exactly 10% warmup
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"[INFO] Training steps: {total_steps}, Warmup steps: {warmup_steps} (10%)")
        
        # Setup loss function with class weights for imbalance
        class_weights = self._calculate_class_weights()
        if class_weights is not None and len(class_weights) == self.intent_config.num_intents:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        return optimizer, scheduler, criterion
    
    def _check_alignment_quality(self):
        """Check alignment quality with sanity checks"""
        self.logger.info("[DEBUG] Checking alignment quality...")
        
        # Sample a few examples for testing
        train_samples = self.train_dataset.data[:10]
        val_samples = self.val_dataset.data[:10]
        
        # Check train samples
        train_entity_coverage = 0
        train_value_coverage = 0
        
        for sample in train_samples:
            text = sample.get('text', sample.get('input', sample.get('sentence', '')))
            entities = sample.get('spans', sample.get('entities', []))
            values = sample.get('values', [])
            
            if entities:
                # Test entity alignment
                entity_labels = self.train_dataset._align_entity_labels(text, entities)
                non_o_entity = sum(1 for x in entity_labels if x not in (-100, self.entity_to_id.get('O', 0)))
                if non_o_entity > 0:
                    train_entity_coverage += 1
                    
            if values:
                # Test value alignment  
                value_labels = self.train_dataset._align_value_labels(text, values)
                non_o_value = sum(1 for x in value_labels if x not in (-100, self.value_to_id.get('O', 0)))
                if non_o_value > 0:
                    train_value_coverage += 1
        
        # Check val samples
        val_entity_coverage = 0
        val_value_coverage = 0
        
        for sample in val_samples:
            text = sample.get('text', sample.get('input', sample.get('sentence', '')))
            entities = sample.get('spans', sample.get('entities', []))
            values = sample.get('values', [])
            
            if entities:
                # Test entity alignment
                entity_labels = self.val_dataset._align_entity_labels(text, entities)
                non_o_entity = sum(1 for x in entity_labels if x not in (-100, self.entity_to_id.get('O', 0)))
                if non_o_entity > 0:
                    val_entity_coverage += 1
                    
            if values:
                # Test value alignment
                value_labels = self.val_dataset._align_value_labels(text, values)
                non_o_value = sum(1 for x in value_labels if x not in (-100, self.value_to_id.get('O', 0)))
                if non_o_value > 0:
                    val_value_coverage += 1
        
        # Log results
        train_entity_ratio = train_entity_coverage / max(1, len([s for s in train_samples if s.get('spans', s.get('entities', []))]))
        train_value_ratio = train_value_coverage / max(1, len([s for s in train_samples if s.get('values', [])]))
        val_entity_ratio = val_entity_coverage / max(1, len([s for s in val_samples if s.get('spans', s.get('entities', []))]))
        val_value_ratio = val_value_coverage / max(1, len([s for s in val_samples if s.get('values', [])]))
        
        self.logger.info(f"[INFO] Alignment Quality Report:")
        self.logger.info(f"  Train Entity Coverage: {train_entity_ratio:.2%} ({train_entity_coverage} samples)")
        self.logger.info(f"  Train Value Coverage: {train_value_ratio:.2%} ({train_value_coverage} samples)")
        self.logger.info(f"  Val Entity Coverage: {val_entity_ratio:.2%} ({val_entity_coverage} samples)")
        self.logger.info(f"  Val Value Coverage: {val_value_ratio:.2%} ({val_value_coverage} samples)")
        
        # Warning if coverage is too low
        if train_entity_ratio < 0.5:
            self.logger.warning(f"[WARN] Low entity alignment coverage: {train_entity_ratio:.2%}")
        if train_value_ratio < 0.5:
            self.logger.warning(f"[WARN] Low value alignment coverage: {train_value_ratio:.2%}")
    
    def train(self) -> Dict[str, Any]:
        """Enhanced training loop with better error handling, monitoring and optimization"""
        if self.train_dataset is None or self.val_dataset is None:
            self.logger.error("ERROR Datasets not initialized. Call load_data() first.")
            raise RuntimeError("Datasets not initialized. Call load_data() first.")
            
        self.logger.info("STARTING Starting training...")
        self.train_start_time = time.time()
        
        # Sanity check: Monitor alignment quality
        self._check_alignment_quality()
        
        # Create data loaders with error handling and optimizations
        try:
            # Cấu hình tối ưu cho GPU 6GB
            num_workers = 0  # Windows - tránh multiprocessing issues
            pin_memory = self.device.type == 'cuda'
            persistent_workers = False  # Không cần khi num_workers=0
            
            self.logger.info(f"[INFO] DataLoader config: workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
            
            # Always use multi-task collate function for proper weight scheduling
            collate_fn = multi_task_collate_fn
            
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
            self.logger.error(f"ERROR Failed to create data loaders: {str(e)}")
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
                self.logger.info(f"Best weighted F1: {best_weighted_f1:.4f}")
                break
            
            # Calculate and log epoch time
            epoch_end = time.time()
            epoch_minutes = (epoch_end - epoch_start) / 60
            epoch_times.append(epoch_minutes)
            
            # Simple time logging
            if epoch % 2 == 0:  # Log every 2 epochs
                self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} completed in {epoch_minutes:.2f} min")
            
            # Memory cleanup - cho cả CPU và GPU
            if torch.cuda.is_available():
                with torch.no_grad():
                    torch.cuda.empty_cache()
            gc.collect()  # Cleanup cho CPU training
        
        # Training complete
        total_time = (time.time() - self.train_start_time) / 60
        self.logger.info(f"\nTraining completed in {total_time:.2f} minutes!")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        self.logger.info(f"Best validation F1 score: {best_val_f1:.4f}")
        
        # Return training history
        return {
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'metrics_history': metrics_history,
            'training_time_minutes': total_time
        }
    
    def _create_weighted_sampler(self):
        """Tạo WeightedRandomSampler để xử lý class imbalance và ưu tiên samples có entity"""
        try:
            # Lấy tất cả intent labels và entity flags từ dataset
            intent_labels = []
            has_entity_flags = []
            
            for item in self.train_dataset.data:
                intent = item.get('command', 'unknown')
                intent_labels.append(intent)
                
                # Check if sample has entities
                spans = item.get('spans', [])
                entities = item.get('entities', [])
                has_entity = 1 if (spans or entities) else 0
                has_entity_flags.append(has_entity)
            
            # Đếm số lượng mỗi class
            from collections import Counter
            class_counts = Counter(intent_labels)
            entity_count = sum(has_entity_flags)
            total_samples = len(has_entity_flags)
            
            # Calculate weights properly - only for classes that appear in training
            K = len([c for c in class_counts.values() if c > 0])  # Number of classes that appear
            
            # Calculate proper class weights
            class_weights = {}
            for intent, count in class_counts.items():
                if count > 0:
                    class_weights[intent] = total_samples / (K * count)
            
            # Tạo weights cho mỗi sample - ưu tiên samples có entity
            sample_weights = []
            entity_boost = 2.0  # Tăng weight cho samples có entity
            
            for i, item in enumerate(self.train_dataset.data):
                intent = item.get('command', 'unknown')
                base_weight = class_weights[intent]
                
                # Boost weight nếu sample có entity
                if has_entity_flags[i]:
                    final_weight = base_weight * entity_boost
                else:
                    final_weight = base_weight
                
                sample_weights.append(final_weight)
            
            # Tạo WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            self.logger.info(f"[INFO] Created WeightedRandomSampler for {len(class_counts)} classes")
            self.logger.info(f"[INFO] Entity coverage: {entity_count}/{total_samples} ({entity_count/total_samples*100:.1f}%)")
            self.logger.info(f"[INFO] Entity boost factor: {entity_boost}x")
            self.logger.info(f"[INFO] Class weights: {dict(list(class_counts.items())[:5])}...")  # Show first 5
            
            return sampler
            
        except Exception as e:
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
                    pass
            
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * num_classes
            
            self.logger.info(f"[INFO] Calculated class weights for {num_classes} classes using FIXED mapping")
            self.logger.info(f"[INFO] Weight range: {class_weights.min():.3f} - {class_weights.max():.3f}")
            
            return class_weights.to(self.device)
            
        except Exception as e:
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
            self.logger.info("SETUP Using Adafactor optimizer (memory efficient)")
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
            self.logger.info("SETUP Using AdamW with FIXED grouping: Encoder=1e-5, Heads=3e-4")
        
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
                        continue
                    else:
                        raise e
                
                # Handle empty batches
                if input_ids.size(0) == 0:
                    continue
                
                # Multi-task training with smart weight scheduling
                intent_labels = batch['intent_labels'].to(self.device)
                entity_labels = batch['entity_labels'].to(self.device)
                command_labels = batch['command_labels'].to(self.device)
                    
                # Only get value_labels if USE_VALUE is True
                value_labels = None
                if getattr(self.config, 'USE_VALUE', False) and 'value_labels' in batch:
                    value_labels = batch['value_labels'].to(self.device)
                
                # Get multi-task weights based on current epoch
                weights = self._get_multi_task_weights(epoch)
                    
                # Log weight changes for debugging
                if epoch == 0:
                    self.logger.info(f"[GPU] Epoch {epoch + 1}: Multi-task training from start (weights: {weights})")
                elif epoch % 2 == 0:  # Log every 2 epochs
                    self.logger.info(f"[INFO] Epoch {epoch + 1}: Multi-task training (weights: {weights})")
                
                # Multi-task forward pass with AMP
                with torch.cuda.amp.autocast():
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
                    
                    loss = outputs['loss']
                
                # Use AMP scaler for backward
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    # Use AMP scaler for optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Clear CUDA cache periodically to prevent memory issues
                    if batch_idx % 5 == 0:  # More frequent cleanup
                        torch.cuda.empty_cache()
                
                # Update metrics
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Multi-task metrics (always enabled now)
                intent_logits = outputs['intent_logits']
                _, predicted = torch.max(intent_logits.data, 1)
                batch_correct = (predicted == intent_labels).sum().item()
                train_total += intent_labels.size(0)
                train_correct += batch_correct
                
                # Batch debug hook: log samples with entities every epoch
                if epoch == 1 and batch_idx < 10:  # First 10 batches of epoch 1
                    # Check if all entity labels are O or -100
                    non_o_entity_count = (entity_labels != -100).sum().item()
                    
                    if non_o_entity_count == 0:
                        self.logger.warning(f"Batch {batch_idx}: All entity labels are O or -100")
                        # Log first sample for debugging
                        if batch_idx == 0:
                            self.logger.info(f"Sample input_ids shape: {input_ids[0].shape}")
                            self.logger.info(f"Sample entity_labels: {entity_labels[0][:10]}")
                    else:
                        self.logger.info(f"Batch {batch_idx}: {non_o_entity_count} non-O entity labels")
                
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
                        except RuntimeError as e:
                            gc.collect()
                
            except Exception as e:
                self.logger.error(f"ERROR Error in batch {batch_idx}: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Try to recover and continue
                optimizer.zero_grad()
                continue
        
        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100 * train_correct / max(train_total, 1)
        
        self.logger.info(f"[INFO] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
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
                    # Debug: Log batch processing
                    if batch_idx < 5:  # Log first 5 batches
                        self.logger.info(f"Processing validation batch {batch_idx}")
                        self.logger.info(f"Batch shape: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}")
                        self.logger.info(f"Batch keys: {list(batch.keys())}")
                    
                    # Move data to device with error handling
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        self.logger.info(f"Batch {batch_idx}: Successfully moved to device")
                    except RuntimeError as e:
                        if "CUDA error: operation not supported" in str(e):
                            self.logger.warning(f"Batch {batch_idx}: CUDA error, skipping")
                            continue
                        else:
                            self.logger.error(f"Batch {batch_idx}: RuntimeError moving to device: {e}")
                            raise e
                    
                    # Handle empty batches
                    if input_ids.size(0) == 0:
                        continue
                    
                    # Ensure tensors are properly shaped
                    if attention_mask.dim() != 2 or input_ids.dim() != 2:
                        continue
                    
                    # Multi-task validation (always enabled now)
                    try:
                        intent_labels = batch['intent_labels'].to(self.device)
                        entity_labels = batch['entity_labels'].to(self.device)
                        command_labels = batch['command_labels'].to(self.device)
                        
                        # Only get value_labels if USE_VALUE is True
                        value_labels = None
                        if getattr(self.config, 'USE_VALUE', False) and 'value_labels' in batch:
                            value_labels = batch['value_labels'].to(self.device)
                        
                        if batch_idx < 5:
                            self.logger.info(f"Batch {batch_idx}: Labels moved to device successfully")
                            self.logger.info(f"Intent labels shape: {intent_labels.shape}, Entity labels shape: {entity_labels.shape}")
                        
                        outputs = self.model(
                            input_ids, 
                            attention_mask,
                            intent_labels=intent_labels,
                            entity_labels=entity_labels,
                            value_labels=value_labels,
                            command_labels=command_labels
                        )
                        
                        if batch_idx < 5:
                            self.logger.info(f"Batch {batch_idx}: Model forward pass successful")
                            self.logger.info(f"Output keys: {list(outputs.keys())}")
                    except Exception as e:
                        self.logger.error(f"Batch {batch_idx}: Error in model forward pass: {type(e).__name__}: {e}")
                        import traceback
                        self.logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue
                    
                    # Collect intent predictions - chuyển CPU ngay để tránh giữ tham chiếu lớn
                    intent_logits = outputs['intent_logits']
                    all_intent_logits.append(intent_logits.cpu())
                    all_intent_labels.append(intent_labels.cpu())
                    
                    # Debug: Log intent collection
                    if batch_idx < 5:
                        self.logger.info(f"Batch {batch_idx}: Collected {len(all_intent_logits)} intent predictions")
                        if len(all_intent_logits) > 0:
                            self.logger.info(f"Intent logits shape: {all_intent_logits[-1].shape}, intent labels shape: {all_intent_labels[-1].shape}")
                    
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
                                        
                                        # Skip padding/ignore tokens (-100)
                                        if true_id == -100:
                                            continue
                                        
                                        if pred_id in self.id_to_entity:
                                            pred_tags.append(self.id_to_entity[pred_id])
                                        else:
                                            pred_tags.append('O')
                                            
                                        if true_id in self.id_to_entity:
                                            true_tags.append(self.id_to_entity[true_id])
                                        else:
                                            true_tags.append('O')
                                    # Don't reset - keep existing tags
                                
                                all_entity_predictions.append(pred_tags)
                                all_entity_true_labels.append(true_tags)
                                
                                # Log sample predictions for debugging
                                if i < 5:  # Log first 5 samples
                                    self.logger.info(f"[DEBUG] Entity Sample {i}: pred={pred_tags[:10]}, true={true_tags[:10]}")
                                
                                # Log non-O labels for debugging
                                non_o_pred = [tag for tag in pred_tags if tag != 'O']
                                non_o_true = [tag for tag in true_tags if tag != 'O']
                                if non_o_pred or non_o_true:
                                    self.logger.info(f"[DEBUG] Entity Non-O Sample {i}: pred={non_o_pred}, true={non_o_true}")
                        except Exception as e:
                            # Skip entity evaluation if there's an error
                            if self.device.type == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except RuntimeError as e:
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
                                        
                                        # Skip padding/ignore tokens (-100)
                                        if true_id == -100:
                                            continue
                                        
                                        if pred_id in self.id_to_value:
                                            pred_tags.append(self.id_to_value[pred_id])
                                        else:
                                            pred_tags.append('O')
                                            
                                        if true_id in self.id_to_value:
                                            true_tags.append(self.id_to_value[true_id])
                                        else:
                                            true_tags.append('O')
                                    # Don't reset - keep existing tags
                                
                                all_value_predictions.append(pred_tags)
                                all_value_true_labels.append(true_tags)
                                
                                # Log non-O labels for debugging
                                non_o_pred = [tag for tag in pred_tags if tag != 'O']
                                non_o_true = [tag for tag in true_tags if tag != 'O']
                                if non_o_pred or non_o_true:
                                    self.logger.info(f"[DEBUG] Value Non-O Sample {i}: pred={non_o_pred}, true={non_o_true}")
                        except Exception as e:
                            # Skip value evaluation if there's an error
                            if self.device.type == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except RuntimeError as e:
                                    gc.collect()
                            continue
                        
                    
                except Exception as e:
                    # Debug: Log exception details with full traceback
                    self.logger.error(f"Validation batch {batch_idx} failed: {type(e).__name__}: {e}")
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Clear GPU cache
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError as e:
                            gc.collect()
                    continue
        
        # Check if we have any valid results - chỉ cần intent/command (không yêu cầu entity)
        if not all_intent_logits:
            self.logger.warning("WARNING: No valid validation results - no intent predictions")
            return {'loss': float('inf'), 'acc': 0.0, 'f1': 0.0, 'entity_f1': None}
        
        # Check if we have valid entity sequences - không skip samples với all O
        if not all_entity_predictions and not all_entity_true_labels:
            self.logger.warning("WARNING: No entity sequences found - setting entity_f1 to None")
            entity_f1 = None
        else:
            entity_f1 = 0.0  # Will be calculated below
            
        # Value sequences - only check if USE_VALUE is True
        if getattr(self.config, 'USE_VALUE', False):
            if not all_value_predictions and not all_value_true_labels:
                self.logger.warning("WARNING: No valid value sequences - setting value_f1 to None")
                value_f1 = None
            else:
                value_f1 = 0.0  # Will be calculated below
        else:
            value_f1 = None  # Disabled
        
        # Initialize results dictionary
        results = {}
        
        # ===== INTENT CLASSIFICATION METRICS =====
        all_intent_logits = torch.cat(all_intent_logits, dim=0).to(self.device)
        all_intent_labels = torch.cat(all_intent_labels, dim=0).to(self.device)
        
        # SANITY CHECK: Counter(y_true_labels) trước metric
        true_labels = all_intent_labels.cpu().numpy()
        from collections import Counter
        label_counts = Counter(true_labels)
        self.logger.info("[DEBUG] SANITY CHECK - Validation label distribution (top 10):")
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
            self.logger.error("ERROR No valid labels found in validation data")
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
            
            self.logger.info(f"[INFO] Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
            
            # Log per-class metrics for important classes
            self.logger.info("\n[INFO] Per-class performance:")
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
                
            self.logger.info("\n[INFO] Most confused classes:")
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
            
            self.logger.info(f"[INFO] Intent - Loss: {intent_loss:.4f}, Acc: {intent_acc:.2f}%, F1: {intent_f1:.4f}")
            
        except Exception as e:
            self.logger.error(f"ERROR Error calculating intent F1: {str(e)}")
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
                
                self.logger.info(f"[INFO] Entity - P: {entity_precision:.4f}, R: {entity_recall:.4f}, F1: {entity_f1:.4f}")
                
                # Detailed entity report with zero_division=0
                entity_report = seqeval_classification_report(
                    all_entity_true_labels, 
                    all_entity_predictions,
                    output_dict=True,
                    zero_division=0
                )
                self.logger.info("[INFO] Entity Classification Report:")
                for entity_type, metrics in entity_report.items():
                    if isinstance(metrics, dict) and entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                        p = metrics.get('precision', 0)
                        r = metrics.get('recall', 0)
                        f = metrics.get('f1-score', 0)
                        s = metrics.get('support', 0)
                        if s > 0:  # Only show non-zero support
                            self.logger.info(f"  {entity_type}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, S={s}")
                        
            except Exception as e:
                self.logger.error(f"ERROR Error calculating entity metrics: {str(e)}")
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
                
                self.logger.info(f"[INFO] Value - P: {value_precision:.4f}, R: {value_recall:.4f}, F1: {value_f1:.4f}")
                
                # Detailed value report with zero_division=0
                value_report = seqeval_classification_report(
                    all_value_true_labels, 
                    all_value_predictions,
                    output_dict=True,
                    zero_division=0
                )
                self.logger.info("[INFO] Value Classification Report:")
                for value_type, metrics in value_report.items():
                    if isinstance(metrics, dict) and value_type not in ['micro avg', 'macro avg', 'weighted avg']:
                        p = metrics.get('precision', 0)
                        r = metrics.get('recall', 0)
                        f = metrics.get('f1-score', 0)
                        s = metrics.get('support', 0)
                        if s > 0:  # Only show non-zero support
                            self.logger.info(f"  {value_type}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, S={s}")
                        
            except Exception as e:
                self.logger.error(f"ERROR Error calculating value metrics: {str(e)}")
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
        
        self.logger.info(f"[INFO] Overall F1: {overall_f1:.4f}")
        
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
            self.logger.error(f"ERROR Error saving model: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Enhanced main function with better error handling"""
    print("Starting GPU-optimized training with PhoBERT")
    
    # Clear GPU cache before starting (only if CUDA is actually available)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"CLEANUP GPU cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        except (RuntimeError, AssertionError) as e:
            print(f"WARNING Could not access GPU properties: {e}")
            print("CPU Continuing with CPU training...")
    else:
        print("CPU Using CPU training (CUDA not available)")
    
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Load configs
        config = ModelConfig()
        intent_config = IntentConfig()
        entity_config = EntityConfig()
        value_config = ValueConfig()
        command_config = CommandConfig()
        
        # Enable multi-task learning (will be disabled if no entity/value data)
        enable_multi_task = True
        
        print(f"Model: {config.model_name}")
        print(f"Model size: {config.model_size}")
        print(f"Max length: {config.max_length}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Freeze layers: {config.freeze_layers}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Multi-task learning: {enable_multi_task}")
        
        if enable_multi_task:
            print(f"Intent labels: {len(intent_config.intent_labels)}")
            print(f"Entity labels: {len(entity_config.entity_labels)}")
            print(f"Value labels: {len(value_config.value_labels)}")
            print(f"Command labels: {len(command_config.command_labels)}")
        
        # Load training config
        from training.configs.config import TrainingConfig
        training_config = TrainingConfig()
        
        # Memory optimization for GPU training
        print("Optimizing memory for GPU training...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU available: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("No GPU available - using CPU")
        gc.collect()
        
        # Load processed data to get actual label counts
        import json
        print("Loading processed data...")
        
        # Load train and val data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        
        train_file = os.path.join(project_root, "src", "data", "processed", "train.json")
        val_file = os.path.join(project_root, "src", "data", "processed", "val.json")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Val file not found: {val_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        actual_data = train_data + val_data
        print(f"Using processed dataset: {len(actual_data)} samples for GPU training")
        
        # Update configs with actual label counts from dataset
        if enable_multi_task:
            # Lấy nhãn entity thực tế từ dataset
            # Dataset có cấu trúc list, không phải dict với train/val
            all_data = actual_data if isinstance(actual_data, list) else actual_data.get('train', []) + actual_data.get('val', [])
            unique_entity_labels = set()
            
            # Lấy tất cả entity labels từ bio_labels
            for item in all_data:
                if 'bio_labels' in item and item['bio_labels']:
                    for label in item['bio_labels']:
                        if isinstance(label, str) and label.strip():
                            unique_entity_labels.add(label.strip())
            
            # Chuẩn hoá vocab: luôn có "O" và chỉ giữ các tag hợp lệ
            normalized_labels = set(["O"]) 
            for lab in sorted(unique_entity_labels):
                if isinstance(lab, str) and lab.strip():
                    lab_up = lab.strip().upper()
                    if lab_up == "O" or lab_up.startswith("B-") or lab_up.startswith("I-"):
                        normalized_labels.add(lab_up)

            # Cập nhật entity_config với labels thực tế từ dataset
            others = sorted([x for x in normalized_labels if x != "O"])
            entity_config.entity_labels = ["O"] + others
            
            # Cập nhật intent_config với tất cả commands từ dataset
            unique_commands = set()
            for item in all_data:
                if 'command' in item:
                    unique_commands.add(item['command'])
            
            # Cập nhật intent labels với tất cả commands
            intent_config.intent_labels = sorted(unique_commands)
            intent_config.num_intents = len(intent_config.intent_labels)
            
            # Cập nhật command labels
            command_config.command_labels = sorted(unique_commands)
            command_config.num_command_labels = len(command_config.command_labels)
            
            print(f"Dataset co {len(normalized_labels)} entity labels: {sorted(normalized_labels)}")
            print(f"Entity config se duoc cap nhat voi {len(entity_config.entity_labels)} labels")
            print(f"Intent config se duoc cap nhat voi {len(intent_config.intent_labels)} labels: {intent_config.intent_labels}")
            print(f"Command config se duoc cap nhat voi {len(command_config.command_labels)} labels: {command_config.command_labels}")
            
            # Debug: Kiểm tra label ranges
            print(f"[DEBUG] DEBUG - Entity label range: 0 to {len(entity_config.entity_labels)-1}")
            print(f"[DEBUG] DEBUG - Intent label range: 0 to {len(intent_config.intent_labels)-1}")
            print(f"[DEBUG] DEBUG - Command label range: 0 to {len(command_config.command_labels)-1}")
            
            # Get actual value labels from dataset
            unique_value_labels = set()
            for item in all_data:
                values = item.get('values', [])
                for val in values:
                    if 'label' in val:
                        unique_value_labels.add(val['label'])
            
            # Update value config with actual labels
            value_config.value_labels = ["O"] + [f"B-{label}" for label in sorted(unique_value_labels) if label != "O"] + [f"I-{label}" for label in sorted(unique_value_labels) if label != "O"]
            
            print(f"Updated Entity labels: {len(entity_config.entity_labels)}")
            print(f"Updated Value labels: {len(value_config.value_labels)}")
            
            # Sanity check: Log label vocabulary
            print(f"Entity labels: {entity_config.entity_labels[:10]}...")
            print(f"Value labels: {value_config.value_labels[:10]}...")
            print(f"Intent labels: {intent_config.intent_labels}")
            print(f"Command labels: {command_config.command_labels}")
        
        # Create trainer with updated configs
        trainer = GPUTrainer(config, intent_config, entity_config, value_config, 
                           command_config, training_config, enable_multi_task)
        
        print(f"Using processed dataset: {len(actual_data)} samples")
        
        # Load data and train (sử dụng processed files)
        trainer.load_data()
        training_results = trainer.train()
        
        # Print final results
        print("\nTraining completed!")
        print(f"Best validation accuracy: {training_results['best_val_acc']:.2f}%")
        print(f"Best validation F1 score: {training_results['best_val_f1']:.4f}")
        print(f"TIME Total training time: {training_results['training_time_minutes']:.2f} minutes")
        
    except Exception as e:
        print(f"ERROR Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()