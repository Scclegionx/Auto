import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from typing import List, Dict, Tuple
from config import model_config, training_config, command_config, entity_config
from models import create_model
from data import DataProcessor

class IntentDataset(Dataset):
    """Dataset cho Intent Recognition"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "intent_id": torch.tensor(item["intent_id"], dtype=torch.long)
        }

class EntityDataset(Dataset):
    """Dataset cho Entity Extraction"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

class CommandDataset(Dataset):
    """Dataset cho Command Processing"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "command_id": torch.tensor(item["command_id"], dtype=torch.long)
        }

class UnifiedDataset(Dataset):
    """Dataset cho Unified Model (Intent + Entity + Command)"""
    
    def __init__(self, intent_data: List[Dict], entity_data: List[Dict], command_data: List[Dict]):
        self.intent_data = intent_data
        self.entity_data = entity_data
        self.command_data = command_data
    
    def __len__(self):
        return len(self.intent_data)
    
    def __getitem__(self, idx):
        intent_item = self.intent_data[idx]
        entity_item = self.entity_data[idx]
        command_item = self.command_data[idx]
        
        return {
            "input_ids": intent_item["input_ids"],
            "attention_mask": intent_item["attention_mask"],
            "intent_id": torch.tensor(intent_item["intent_id"], dtype=torch.long),
            "entity_labels": torch.tensor(entity_item["labels"], dtype=torch.long),
            "command_id": torch.tensor(command_item["command_id"], dtype=torch.long)
        }

class Trainer:
    """Trainer cho các mô hình PhoBERT"""
    
    def __init__(self, model_type: str = "unified"):
        self.model_type = model_type
        self.device = torch.device(training_config.device)
        self.model = create_model(model_type).to(self.device)
        self.data_processor = DataProcessor()
        
    def train_intent_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Huấn luyện mô hình Intent Recognition"""
        print("Bắt đầu huấn luyện Intent Recognition Model...")
        
        # Tạo datasets
        train_dataset = IntentDataset(train_data)
        val_dataset = IntentDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
        total_steps = len(train_loader) * model_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=model_config.warmup_steps, num_training_steps=total_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(model_config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config.num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_ids = batch["intent_id"].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, intent_ids)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += intent_ids.size(0)
                train_correct += (predicted == intent_ids).sum().item()
            
            # Validation
            val_acc, val_loss = self._evaluate_intent(val_loader, criterion)
            train_acc = train_correct / train_total
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{training_config.output_dir}/best_intent_model.pth")
        
        print(f"Intent Recognition training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    def train_entity_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Huấn luyện mô hình Entity Extraction"""
        print("Bắt đầu huấn luyện Entity Extraction Model...")
        
        # Tạo datasets
        train_dataset = EntityDataset(train_data)
        val_dataset = EntityDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
        total_steps = len(train_loader) * model_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=model_config.warmup_steps, num_training_steps=total_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        best_val_f1 = 0.0
        
        for epoch in range(model_config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config.num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                
                # Reshape logits và labels cho loss calculation
                logits = logits.view(-1, entity_config.num_entities)
                labels = labels.view(-1)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            val_f1, val_loss = self._evaluate_entity(val_loader, criterion)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{training_config.output_dir}/best_entity_model.pth")
        
        print(f"Entity Extraction training completed. Best validation F1: {best_val_f1:.4f}")
    
    def train_command_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Huấn luyện mô hình Command Processing"""
        print("Bắt đầu huấn luyện Command Processing Model...")
        
        # Tạo datasets
        train_dataset = CommandDataset(train_data)
        val_dataset = CommandDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
        total_steps = len(train_loader) * model_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=model_config.warmup_steps, num_training_steps=total_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(model_config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config.num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                command_ids = batch["command_id"].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, command_ids)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += command_ids.size(0)
                train_correct += (predicted == command_ids).sum().item()
            
            # Validation
            val_acc, val_loss = self._evaluate_command(val_loader, criterion)
            train_acc = train_correct / train_total
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{training_config.output_dir}/best_command_model.pth")
        
        print(f"Command Processing training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    def train_unified_model(self, train_intent: List[Dict], val_intent: List[Dict],
                           train_entity: List[Dict], val_entity: List[Dict],
                           train_command: List[Dict], val_command: List[Dict]):
        """Huấn luyện Unified Model kết hợp cả 3 tác vụ"""
        print("Bắt đầu huấn luyện Unified Model...")
        
        # Tạo datasets
        train_dataset = UnifiedDataset(train_intent, train_entity, train_command)
        val_dataset = UnifiedDataset(val_intent, val_entity, val_command)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
        total_steps = len(train_loader) * model_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=model_config.warmup_steps, num_training_steps=total_steps)
        
        # Loss functions
        intent_criterion = nn.CrossEntropyLoss()
        entity_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        command_criterion = nn.CrossEntropyLoss()
        
        best_val_score = 0.0
        
        for epoch in range(model_config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_intent_correct = 0
            train_command_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config.num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_ids = batch["intent_id"].to(self.device)
                entity_labels = batch["entity_labels"].to(self.device)
                command_ids = batch["command_id"].to(self.device)
                
                optimizer.zero_grad()
                intent_logits, entity_logits, command_logits = self.model(input_ids, attention_mask)
                
                # Tính loss cho từng tác vụ
                intent_loss = intent_criterion(intent_logits, intent_ids)
                entity_loss = entity_criterion(entity_logits.view(-1, entity_config.num_entities), entity_labels.view(-1))
                command_loss = command_criterion(command_logits, command_ids)
                
                # Tổng loss
                total_loss = intent_loss + entity_loss + command_loss
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += total_loss.item()
                
                # Tính accuracy cho intent và command
                _, intent_pred = torch.max(intent_logits, 1)
                _, command_pred = torch.max(command_logits, 1)
                train_total += intent_ids.size(0)
                train_intent_correct += (intent_pred == intent_ids).sum().item()
                train_command_correct += (command_pred == command_ids).sum().item()
            
            # Validation
            val_metrics = self._evaluate_unified(val_loader, intent_criterion, entity_criterion, command_criterion)
            train_intent_acc = train_intent_correct / train_total
            train_command_acc = train_command_correct / train_total
            
            print(f"Epoch {epoch+1}:")
            print(f"  Train - Intent Acc: {train_intent_acc:.4f}, Command Acc: {train_command_acc:.4f}")
            print(f"  Val - Intent Acc: {val_metrics['intent_acc']:.4f}, Command Acc: {val_metrics['command_acc']:.4f}, Entity F1: {val_metrics['entity_f1']:.4f}")
            
            # Save best model
            val_score = val_metrics['intent_acc'] + val_metrics['command_acc'] + val_metrics['entity_f1']
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(self.model.state_dict(), f"{training_config.output_dir}/best_unified_model.pth")
        
        print(f"Unified Model training completed. Best validation score: {best_val_score:.4f}")
    
    def _evaluate_intent(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Đánh giá mô hình Intent Recognition"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_ids = batch["intent_id"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, intent_ids)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(intent_ids.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = val_loss / len(val_loader)
        
        return accuracy, avg_loss
    
    def _evaluate_entity(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Đánh giá mô hình Entity Extraction"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                # Reshape cho loss calculation
                logits = logits.view(-1, entity_config.num_entities)
                labels = labels.view(-1)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Chỉ tính F1 cho non-padding tokens
                mask = labels != -100
                if mask.sum() > 0:
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds[mask].cpu().numpy())
                    all_labels.extend(labels[mask].cpu().numpy())
        
        # Tính F1 score
        if len(all_preds) > 0:
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0.0
        
        avg_loss = val_loss / len(val_loader)
        
        return f1, avg_loss
    
    def _evaluate_command(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Đánh giá mô hình Command Processing"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                command_ids = batch["command_id"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, command_ids)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(command_ids.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = val_loss / len(val_loader)
        
        return accuracy, avg_loss
    
    def _evaluate_unified(self, val_loader: DataLoader, intent_criterion: nn.Module, 
                         entity_criterion: nn.Module, command_criterion: nn.Module) -> Dict[str, float]:
        """Đánh giá Unified Model"""
        self.model.eval()
        val_loss = 0.0
        all_intent_preds = []
        all_intent_labels = []
        all_command_preds = []
        all_command_labels = []
        all_entity_preds = []
        all_entity_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_ids = batch["intent_id"].to(self.device)
                entity_labels = batch["entity_labels"].to(self.device)
                command_ids = batch["command_id"].to(self.device)
                
                intent_logits, entity_logits, command_logits = self.model(input_ids, attention_mask)
                
                # Tính loss
                intent_loss = intent_criterion(intent_logits, intent_ids)
                entity_loss = entity_criterion(entity_logits.view(-1, entity_config.num_entities), entity_labels.view(-1))
                command_loss = command_criterion(command_logits, command_ids)
                total_loss = intent_loss + entity_loss + command_loss
                val_loss += total_loss.item()
                
                # Intent predictions
                _, intent_pred = torch.max(intent_logits, 1)
                all_intent_preds.extend(intent_pred.cpu().numpy())
                all_intent_labels.extend(intent_ids.cpu().numpy())
                
                # Command predictions
                _, command_pred = torch.max(command_logits, 1)
                all_command_preds.extend(command_pred.cpu().numpy())
                all_command_labels.extend(command_ids.cpu().numpy())
                
                # Entity predictions
                entity_preds = torch.argmax(entity_logits, dim=2)
                mask = entity_labels != -100
                if mask.sum() > 0:
                    all_entity_preds.extend(entity_preds[mask].cpu().numpy())
                    all_entity_labels.extend(entity_labels[mask].cpu().numpy())
        
        # Tính metrics
        intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
        command_acc = accuracy_score(all_command_labels, all_command_preds)
        
        if len(all_entity_preds) > 0:
            from sklearn.metrics import f1_score
            entity_f1 = f1_score(all_entity_labels, all_entity_preds, average='weighted')
        else:
            entity_f1 = 0.0
        
        avg_loss = val_loss / len(val_loader)
        
        return {
            'intent_acc': intent_acc,
            'command_acc': command_acc,
            'entity_f1': entity_f1,
            'loss': avg_loss
        } 