#!/usr/bin/env python3
"""
Script training tối ưu cho GPU với PhoBERT-Large
"""

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
from typing import List, Dict, Tuple
import gc

# Import config
from config import ModelConfig, IntentConfig

class IntentDataset(torch.utils.data.Dataset):
    """Dataset cho Intent Recognition"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['input']
        intent = item['command']
        
        # Tokenize với padding đúng cách
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent': intent
        }

class OptimizedIntentModel(nn.Module):
    def __init__(self, model_name: str, num_intents: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Cải thiện khởi tạo model
        self.phobert = AutoModel.from_pretrained(
            model_name,
            gradient_checkpointing=config.gradient_checkpointing,
            use_safetensors=True,
            trust_remote_code=True,
            cache_dir="model_cache"  # Thêm cache_dir để tránh tải lại
        )
        
        # Đóng băng một số layer để tiết kiệm bộ nhớ và tăng tốc
        if config.freeze_layers > 0:
            modules = [self.phobert.embeddings]
            for i in range(min(config.freeze_layers, len(self.phobert.encoder.layer))):
                modules.append(self.phobert.encoder.layer[i])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        hidden_size = self.phobert.config.hidden_size
        
        # Cải thiện kiến trúc classifier
        if config.model_size == "large":
            self.dropout1 = nn.Dropout(config.dropout)
            self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
            self.activation1 = nn.GELU()  # GELU thay vì ReLU
            self.batchnorm1 = nn.BatchNorm1d(hidden_size // 2)
            
            self.dropout2 = nn.Dropout(config.dropout)
            self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4)
            self.activation2 = nn.GELU()
            self.batchnorm2 = nn.BatchNorm1d(hidden_size // 4)
            
            self.dropout3 = nn.Dropout(config.dropout)
            self.output = nn.Linear(hidden_size // 4, num_intents)
        else:
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        # Tối ưu forward pass
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,  # Tiết kiệm bộ nhớ
            return_dict=True,
            use_cache=not (self.config.gradient_checkpointing and self.training)
        )
        
        # Mean pooling tốt hơn và ổn định hơn
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Forward qua classifier
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
    """Trainer tối ưu cho GPU với mixed precision và logging nâng cao"""
    
    def __init__(self, config: ModelConfig, intent_config: IntentConfig):
        self.config = config
        self.intent_config = intent_config
        self.device = torch.device(config.device)
        
        # Khởi tạo scaler cho mixed precision
        self.scaler = amp.GradScaler(enabled=config.use_fp16 and self.device.type == "cuda")
        
        # Setup logging với encoding UTF-8
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"training_{config.model_size}.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🖥️ Using device: {self.device}")
        if self.device.type == "cuda":
            self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Enable cuDNN benchmark
            torch.backends.cudnn.benchmark = True
        
        # Load tokenizer với padding token
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model
        self.model = OptimizedIntentModel(
            config.model_name, 
            intent_config.num_intents, 
            config
        ).to(self.device)
        
        # Enable mixed precision và gradient checkpointing cho GPU
        if config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            self.logger.info("🔧 Enabled FP16")
        
        if config.gradient_checkpointing:
            self.model.phobert.gradient_checkpointing_enable()
            self.logger.info("🔧 Enabled Gradient Checkpointing")
        
        self.logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self, file_path: str):
        """Load và prepare data với kiểm tra và phân tích nâng cao"""
        self.logger.info(f"📂 Loading data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"📈 Loaded {len(data)} samples")
        
        # Thêm kiểm tra dữ liệu đầu vào
        # Đảm bảo không có mẫu nào quá dài
        max_len = max(len(self.tokenizer.encode(item['input'])) for item in data)
        if max_len > self.config.max_length:
            self.logger.warning(f"⚠️ Một số mẫu có độ dài > {self.config.max_length} tokens (max: {max_len})")
        
        # Kiểm tra phân phối intent
        intent_counts = {}
        for item in data:
            intent = item['command']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        self.logger.info("Phân phối intent:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {intent}: {count} mẫu ({count/len(data)*100:.1f}%)")
        
        # Create intent mapping
        intents = list(set(item['command'] for item in data))
        intents.sort()
        self.intent_to_id = {intent: i for i, intent in enumerate(intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        self.logger.info(f"🎯 Found {len(intents)} intents: {intents}")
        
        # Cải thiện phân chia train/val - sử dụng stratified nếu có thể
        try:
            train_data, val_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=42,
                stratify=[item['command'] for item in data]  # Stratified split
            )
        except ValueError:
            # Nếu stratified split thất bại (quá ít mẫu cho một số class), dùng random split
            self.logger.warning("⚠️ Không thể sử dụng stratified split, chuyển sang random split")
            train_data, val_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=42
            )
        
        # Create datasets
        self.train_dataset = IntentDataset(train_data, self.tokenizer, self.config.max_length)
        self.val_dataset = IntentDataset(val_data, self.tokenizer, self.config.max_length)
        
        self.logger.info(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
        
        return train_data, val_data
    
    def train(self):
        """Training loop tối ưu với mixed precision và logging nâng cao"""
        self.logger.info("🚀 Starting training...")
        
        # Data loaders không sử dụng data collator để tránh lỗi với intent field
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Sử dụng optimizer tốt hơn
        # Nếu bộ nhớ GPU thấp, dùng Adafactor thay AdamW
        if hasattr(self.config, 'optimizer') and self.config.optimizer == "adafactor":
            optimizer = Adafactor(
                self.model.parameters(),
                lr=self.config.learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.logger.info("🔧 Using Adafactor optimizer (memory efficient)")
        else:
            # Grouped parameters để tối ưu hóa riêng biệt
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
                eps=getattr(self.config, 'adam_epsilon', 1e-8)
            )
            self.logger.info("🔧 Using AdamW optimizer with weight decay differentiation")
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop với mixed precision và gradient accumulation
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Convert to intents
                intents = [self.intent_to_id[intent] for intent in batch['intent']]
                labels = torch.tensor(intents).to(self.device)
                
                # Mixed precision training
                with amp.autocast(enabled=self.config.use_fp16 and self.device.type == "cuda"):
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels) / self.config.gradient_accumulation_steps
                
                # Backward pass với gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_fp16 and self.device.type == "cuda":
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    if self.config.use_fp16 and self.device.type == "cuda":
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Statistics
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            # Learning rate adjustment giữa các epoch
            if epoch > 0 and epoch % 2 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9  # Giảm học tốc độ học tập 10% sau mỗi 2 epoch
                self.logger.info(f"📉 Adjusted learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            
            # Validation với kỹ thuật mới
            self.model.eval()
            all_val_logits = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    intents = [self.intent_to_id[intent] for intent in batch['intent']]
                    labels = torch.tensor(intents).to(self.device)
                    
                    # Mixed precision inference
                    with amp.autocast(enabled=self.config.use_fp16 and self.device.type == "cuda"):
                        logits = self.model(input_ids, attention_mask)
                    
                    # Thu thập tất cả logits và labels để đánh giá sau
                    all_val_logits.append(logits)
                    all_val_labels.append(labels)
            
            # Ghép tất cả logits và labels lại
            all_val_logits = torch.cat(all_val_logits, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            
            # Tính loss và metrics
            val_loss = criterion(all_val_logits, all_val_labels).item()
            _, predicted = torch.max(all_val_logits, 1)
            val_acc = 100 * (predicted == all_val_labels).sum().item() / all_val_labels.size(0)
            
            # Calculate training metrics
            train_acc = 100 * train_correct / train_total
            
            self.logger.info(f"📊 Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Detailed report
            predictions = predicted.cpu().numpy()
            true_labels = all_val_labels.cpu().numpy()
            
            # Detailed classification report
            unique_labels = sorted(set(true_labels) | set(predictions))
            target_names = [self.id_to_intent[i] for i in unique_labels]
            report = classification_report(
                true_labels, 
                predictions, 
                labels=unique_labels,
                target_names=target_names,
                zero_division=0
            )
            
            self.logger.info(f"\n{report}")
            
            # Save model mỗi epoch và best model
            self.save_model(epoch=epoch+1, is_best=False)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(epoch=epoch+1, is_best=True)
                self.logger.info(f"💾 Saved best model with val_acc: {val_acc:.2f}%")
            
            # Clear GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        
        self.logger.info(f"\n🎉 Training completed! Best val_acc: {best_val_acc:.2f}%")
    
    def save_model(self, epoch: int = None, is_best: bool = False):
        """Save model và tokenizer với tối ưu cho large model"""
        save_dir = f"models/phobert_{self.config.model_size}_intent_model"
        os.makedirs(save_dir, exist_ok=True)
        
        # Tạo tên file với epoch và best flag
        if epoch is not None:
            filename = f"model_epoch_{epoch}"
            if is_best:
                filename += "_best"
            filename += ".pth"
        else:
            filename = "model.pth"
        
        # Save model state với đầy đủ thông tin
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'intent_to_id': self.intent_to_id,
            'id_to_intent': self.id_to_intent,
            'config': self.config,
            'intent_config': self.intent_config,
            'epoch': epoch,
            'is_best': is_best,
            'model_size': self.config.model_size,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Save với compression để tiết kiệm disk space
        torch.save(checkpoint, f"{save_dir}/{filename}", _use_new_zipfile_serialization=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config riêng biệt
        import json
        with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump({
                'model_size': self.config.model_size,
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent,
                'max_length': self.config.max_length,
                'num_intents': self.intent_config.num_intents
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Model saved to {save_dir}/{filename}")
        print(f"📊 Model size: {os.path.getsize(f'{save_dir}/{filename}') / 1024**2:.2f} MB")

def main():
    """Main function"""
    print("🚀 Starting GPU-optimized training with PhoBERT-Large")
    
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
    
    # Load data - ưu tiên dataset mở rộng
    dataset_file = "elderly_command_dataset_expanded.json"
    if not os.path.exists(dataset_file):
        print(f"⚠️ Expanded dataset not found, using original dataset")
        dataset_file = "elderly_command_dataset_reduced.json"
    
    trainer.load_data(dataset_file)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
