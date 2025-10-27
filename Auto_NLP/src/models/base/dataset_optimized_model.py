"""
Dataset-Optimized Model Architecture
Tối ưu cho dataset thực tế với class imbalance và intent-only focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional
import math

class DatasetOptimizedModel(nn.Module):
    """Model architecture tối ưu cho dataset thực tế"""
    
    def __init__(self, model_name: str, num_intents: int, config: Dict[str, Any] = None):
        super().__init__()
        self.model_name = model_name
        self.num_intents = num_intents
        self.config = config or {}
        
        # Load pre-trained PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Intent classification head với dropout và normalization
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.2),  # Higher dropout for regularization
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_intents)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Class weights for imbalanced dataset
        self.class_weights = None
        
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.intent_classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def set_class_weights(self, class_counts: Dict[int, int]):
        """Set class weights for imbalanced dataset"""
        total_samples = sum(class_counts.values())
        max_count = max(class_counts.values())
        
        # Calculate inverse frequency weights
        weights = []
        for i in range(self.num_intents):
            count = class_counts.get(i, 1)  # Avoid division by zero
            weight = max_count / count
            weights.append(weight)
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights set: {self.class_weights}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                intent_labels: torch.Tensor = None) -> Dict[str, Any]:
        """Forward pass for intent classification"""
        # Get encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        
        result = {
            'intent_logits': intent_logits
        }
        
        # Calculate loss if labels provided (for training)
        if intent_labels is not None:
            # Use weighted loss for class imbalance
            if self.class_weights is not None:
                class_weights = self.class_weights.to(intent_logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            
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
            "architecture": "DatasetOptimizedModel",
            "class_weights": self.class_weights is not None
        }

class DatasetOptimizedTrainer:
    """Trainer tối ưu cho dataset thực tế"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Training parameters
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.batch_size = 16
        self.num_epochs = 10
        
        # Optimizer với learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler với warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = 3
        self.patience_counter = 0
    
    def calculate_class_weights(self, dataset):
        """Tính toán class weights từ dataset"""
        class_counts = {}
        for item in dataset:
            intent = item.get('command', 'unknown')
            # Map intent to index (cần implement mapping)
            intent_idx = self._get_intent_index(intent)
            class_counts[intent_idx] = class_counts.get(intent_idx, 0) + 1
        
        self.model.set_class_weights(class_counts)
        return class_counts
    
    def _get_intent_index(self, intent):
        """Map intent string to index"""
        intent_mapping = {
            'send-mess': 0, 'call': 1, 'set-alarm': 2, 'set-event-calendar': 3,
            'get-info': 4, 'add-contacts': 5, 'control-device': 6, 'make-video-call': 7,
            'open-cam': 8, 'play-media': 9, 'search-internet': 10, 'view-content': 11,
            'search-youtube': 12
        }
        return intent_mapping.get(intent, 0)
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            result = self.model(input_ids, attention_mask, intent_labels)
            loss = result['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(result['intent_logits'], dim=1)
            correct += (predictions == intent_labels).sum().item()
            total += intent_labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent_labels'].to(self.device)
                
                result = self.model(input_ids, attention_mask, intent_labels)
                loss = result['loss']
                
                total_loss += loss.item()
                predictions = torch.argmax(result['intent_logits'], dim=1)
                correct += (predictions == intent_labels).sum().item()
                total += intent_labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("Starting dataset-optimized training...")
        
        # Calculate class weights
        print("Calculating class weights...")
        # self.calculate_class_weights(train_dataset)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                print(f"New best validation accuracy: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")
        return self.best_val_acc

