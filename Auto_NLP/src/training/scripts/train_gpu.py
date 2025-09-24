import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Adafactor
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
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
from src.training.configs.config import ModelConfig, IntentConfig

# Set seed for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

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
        'input_ids': torch.tensor(padded_input_ids),
        'attention_mask': torch.tensor(padded_attention_masks),
        'labels': torch.tensor(labels)
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

class OptimizedIntentModel(nn.Module):
    """Optimized model for intent classification with improved architecture"""
    
    def __init__(self, model_name: str, num_intents: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained model with proper error handling
        try:
            self.phobert = AutoModel.from_pretrained(
                model_name,
                gradient_checkpointing=config.gradient_checkpointing,
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
        
        # Model architecture based on size
        if config.model_size == "large":
            # Multi-layer architecture for large model
            self.dropout1 = nn.Dropout(config.dropout)
            self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
            self.activation1 = nn.GELU()
            self.batchnorm1 = nn.BatchNorm1d(hidden_size // 2)
            
            self.dropout2 = nn.Dropout(config.dropout)
            self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4)
            self.activation2 = nn.GELU()
            self.batchnorm2 = nn.BatchNorm1d(hidden_size // 4)
            
            self.dropout3 = nn.Dropout(config.dropout)
            self.output = nn.Linear(hidden_size // 4, num_intents)
        else:
            # Simpler architecture for base model
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        # Forward pass with memory optimization
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=not (self.config.gradient_checkpointing and self.training)
        )
        
        # Attention-weighted pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Forward through layers based on model size
        if self.config.model_size == "large":
            x = self.dropout1(pooled_output)
            x = self.linear1(x)
            x = self.activation1(x)
            x = self.batchnorm1(x)
            
            x = self.dropout2(x)
            x = self.linear2(x)
            x = self.activation2(x)
            x = self.batchnorm2(x)
            
            x = self.dropout3(x)
            logits = self.output(x)
        else:
            x = self.dropout(pooled_output)
            logits = self.output(x)
            
        return logits

class GPUTrainer:
    """Enhanced trainer with better GPU optimization, error handling, and monitoring"""
    
    def __init__(self, config: ModelConfig, intent_config: IntentConfig):
        self.config = config
        self.intent_config = intent_config
        self.device = self._setup_device()
        
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
        
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device with better error handling"""
        if torch.cuda.is_available() and self.config.device.startswith('cuda'):
            device = torch.device(self.config.device)
            # Verify the specified GPU exists
            if self.config.device != 'cuda' and int(self.config.device.split(':')[1]) >= torch.cuda.device_count():
                print(f"Warning: Specified GPU {self.config.device} doesn't exist. Falling back to cuda:0")
                device = torch.device('cuda:0')
        else:
            if self.config.device.startswith('cuda'):
                print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
            
        return device

    def _log_system_info(self) -> None:
        """Log detailed system information"""
        self.logger.info(f"🖥️ Using device: {self.device}")
        
        if self.device.type == "cuda":
            self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            self.logger.info(f"📊 CUDA Version: {torch.version.cuda}")
            self.logger.info(f"📊 CUDNN Version: {torch.backends.cudnn.version()}")
            
            # Enable cudnn benchmark for performance if not using deterministic algorithms
            torch.backends.cudnn.benchmark = not getattr(self.config, 'deterministic', False)
            
            # Log VRAM usage
            torch.cuda.reset_peak_memory_stats()
            
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
    
    def _initialize_model(self) -> OptimizedIntentModel:
        """Initialize model with proper error handling"""
        try:
            model = OptimizedIntentModel(
                self.config.model_name, 
                self.intent_config.num_intents, 
                self.config
            ).to(self.device)
            
            # Log model size and parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"📊 Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
            self.logger.info(f"📊 Model size on disk: ~{total_params * 4 / (1024**2):.1f} MB")
            
            # Enable gradient checkpointing if configured
            if self.config.gradient_checkpointing:
                model.phobert.gradient_checkpointing_enable()
                self.logger.info("🔧 Enabled Gradient Checkpointing")
            
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
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
        
        # Create intent mappings
        intents = sorted(list(intent_counts.keys()))
        self.intent_to_id = {intent: i for i, intent in enumerate(intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        self.logger.info(f"🎯 Found {len(intents)} intents: {intents}")
        
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
        
        # Verify split doesn't lose any classes
        train_intents = set(item.get('command', 'unknown') for item in train_data)
        val_intents = set(item.get('command', 'unknown') for item in val_data)
        all_intents = set(intent_counts.keys())
        
        missing_in_train = all_intents - train_intents
        missing_in_val = all_intents - val_intents
        
        if missing_in_train:
            self.logger.warning(f"⚠️ Some intents missing from training set: {missing_in_train}")
        if missing_in_val:
            self.logger.warning(f"⚠️ Some intents missing from validation set: {missing_in_val}")
        
        # Create datasets
        try:
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
    
    def train(self) -> Dict[str, Any]:
        """Enhanced training loop with better error handling, monitoring and optimization"""
        if self.train_dataset is None or self.val_dataset is None:
            self.logger.error("❌ Datasets not initialized. Call load_data() first.")
            raise RuntimeError("Datasets not initialized. Call load_data() first.")
            
        self.logger.info("🚀 Starting training...")
        self.train_start_time = time.time()
        
        # Create data loaders with error handling and optimizations
        try:
            num_workers = 0 if self.device.type == 'cpu' else min(4, os.cpu_count() or 1)
            pin_memory = self.device.type == 'cuda'
            
            self.logger.info(f"📊 DataLoader config: workers={num_workers}, pin_memory={pin_memory}")
            
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn,
                drop_last=False,
                prefetch_factor=2 if num_workers > 0 else None
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn,
                drop_last=False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to create data loaders: {str(e)}")
            raise RuntimeError(f"Failed to create data loaders: {str(e)}")
        
        # Setup optimizer with error handling
        try:
            optimizer = self._create_optimizer()
        except Exception as e:
            self.logger.error(f"❌ Failed to create optimizer: {str(e)}")
            raise RuntimeError(f"Failed to create optimizer: {str(e)}")
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = min(
            self.config.warmup_steps,
            int(total_steps * 0.1)  # Cap at 10% of total steps
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"📊 Training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training state
        best_val_acc = 0.0
        best_val_f1 = 0.0
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
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_val_f1 = val_metrics.get('f1', 0.0)
                self.save_model(epoch=epoch+1, is_best=True)
                self.logger.info(f"💾 Saved best model with val_acc: {best_val_acc:.2f}%")
            
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
            
            # Memory cleanup
            if self.device.type == "cuda":
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
    
    def _create_optimizer(self) -> Union[Adafactor, AdamW]:
        """Create and return the appropriate optimizer based on configuration"""
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
            # Weight decay differentiation for AdamW
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay
                },
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
            
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                eps=getattr(self.config, 'adam_epsilon', 1e-8),
                betas=getattr(self.config, 'adam_betas', (0.9, 0.999))
            )
            self.logger.info("🔧 Using AdamW optimizer with weight decay differentiation")
        
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
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Handle empty batches
                if input_ids.size(0) == 0:
                    self.logger.warning("⚠️ Received empty batch - skipping")
                    continue
                
                # Standard precision training (FP16 disabled)
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
                
                # Update metrics
                train_loss += loss.item() * self.config.gradient_accumulation_steps
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
                    
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%',
                        'ex/s': f'{examples_per_second:.1f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
                else:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%'
                    })
                
                # Periodically log GPU memory usage
                if self.device.type == "cuda" and batch_idx % 50 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    self.logger.debug(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                
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
        """Validate model with detailed metrics"""
        self.model.eval()
        all_val_logits = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Move data to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Handle empty batches
                    if input_ids.size(0) == 0:
                        continue
                    
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    
                    all_val_logits.append(logits)
                    all_val_labels.append(labels)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in validation batch: {str(e)}")
                    continue
        
        # Check if we have any valid results
        if not all_val_logits:
            self.logger.error("❌ No valid validation results")
            return {'loss': float('inf'), 'acc': 0.0, 'f1': 0.0}
        
        # Concatenate all batches
        all_val_logits = torch.cat(all_val_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        
        # Calculate loss
        val_loss = criterion(all_val_logits, all_val_labels).item()
        
        # Calculate accuracy
        _, predicted = torch.max(all_val_logits, 1)
        val_acc = 100 * (predicted == all_val_labels).sum().item() / all_val_labels.size(0)
        
        self.logger.info(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Generate detailed classification report
        predictions = predicted.cpu().numpy()
        true_labels = all_val_labels.cpu().numpy()
        
        # Get unique labels actually present in the data
        unique_labels = sorted(set(true_labels) | set(predictions))
        if not unique_labels:
            self.logger.error("❌ No valid labels found in validation data")
            return {'loss': val_loss, 'acc': val_acc, 'f1': 0.0}
            
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
            
            # Return metrics
            return {
                'loss': val_loss,
                'acc': val_acc,
                'f1': weighted_f1
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating classification report: {str(e)}")
            return {'loss': val_loss, 'acc': val_acc, 'f1': 0.0}
    
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
    
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Load configs
        config = ModelConfig()
        intent_config = IntentConfig()
        
        print(f"📋 Model: {config.model_name}")
        print(f"📋 Model size: {config.model_size}")
        print(f"📋 Max length: {config.max_length}")
        print(f"📋 Batch size: {config.batch_size}")
        print(f"📋 Learning rate: {config.learning_rate}")
        print(f"📋 Epochs: {config.num_epochs}")
        print(f"📋 Freeze layers: {config.freeze_layers}")
        print(f"📋 Gradient accumulation: {config.gradient_accumulation_steps}")
        
        # Create trainer
        trainer = GPUTrainer(config, intent_config)
        
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