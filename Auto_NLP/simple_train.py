#!/usr/bin/env python3
"""
Script training đơn giản nhất cho demo
Chỉ train Intent Recognition model
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from config import model_config, training_config

class SimpleIntentDataset(Dataset):
    """Dataset đơn giản cho Intent Recognition"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tạo label mapping
        self.intents = list(set(item["command"] for item in data))
        self.intent_to_id = {intent: idx for idx, intent in enumerate(self.intents)}
        self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["input"]
        intent = item["command"]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.intent_to_id[intent], dtype=torch.long),
            "text": text,
            "intent": intent
        }

class SimpleIntentModel(nn.Module):
    """Model đơn giản cho Intent Recognition"""
    
    def __init__(self, model_name, num_intents):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_dataset(file_path):
    """Load dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_dataset(data, train_ratio=0.8):
    """Split dataset thành train và validation"""
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def train_simple_model():
    """Training model đơn giản"""
    print("🚀 BẮT ĐẦU TRAINING ĐƠN GIẢN")
    print("=" * 50)
    
    # Load dataset
    dataset_file = "elderly_augmented.json"
    if not os.path.exists(dataset_file):
        dataset_file = "elderly_command_dataset_reduced.json"
    
    print(f"Loading dataset: {dataset_file}")
    dataset = load_dataset(dataset_file)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split dataset
    train_data, val_data = split_dataset(dataset)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # Create datasets
    train_dataset = SimpleIntentDataset(train_data, tokenizer, model_config.max_length)
    val_dataset = SimpleIntentDataset(val_data, tokenizer, model_config.max_length)
    
    print(f"Number of intents: {len(train_dataset.intents)}")
    print(f"Intents: {train_dataset.intents}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    device = torch.device("cpu")
    model = SimpleIntentModel(model_config.model_name, len(train_dataset.intents)).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * model_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training config:")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {model_config.batch_size}")
    print(f"  - Learning rate: {model_config.learning_rate}")
    print(f"  - Epochs: {model_config.num_epochs}")
    print(f"  - Total steps: {total_steps}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(model_config.num_epochs):
        print(f"\nEpoch {epoch+1}/{model_config.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'intent_to_id': train_dataset.intent_to_id,
                'id_to_intent': train_dataset.id_to_intent
            }, "models/best_simple_intent_model.pth")
            print(f"💾 Lưu model tốt nhất với accuracy: {val_acc:.2f}%")
    
    print(f"\n🎉 Training hoàn thành!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model được lưu tại: models/best_simple_intent_model.pth")
    
    return model, train_dataset

def test_simple_model(model, dataset):
    """Test model đơn giản"""
    print("\n🧪 TESTING MODEL")
    print("=" * 30)
    
    device = torch.device("cpu")
    model.eval()
    
    test_samples = [
        "nhắc tôi lúc 5 giờ chiều",
        "alo cho bố",
        "gửi tin nhắn cho mẹ",
        "đặt báo thức lúc 7 giờ sáng",
        "thời tiết hôm nay thế nào",
        "tôi muốn nghe nhạc",
        "đọc tin tức cho tôi"
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    for text in test_samples:
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            logits = model(encoding["input_ids"], encoding["attention_mask"])
            predicted = torch.argmax(logits, dim=1)
            intent = dataset.id_to_intent[predicted.item()]
            confidence = torch.softmax(logits, dim=1).max().item()
        
        print(f"Input: '{text}'")
        print(f"Predicted: {intent} (confidence: {confidence:.3f})")
        print("-" * 50)

def main():
    """Main function"""
    print("🎯 PHOBERT_SAM - SIMPLE TRAINING")
    print("=" * 50)
    print(f"⏰ Bắt đầu lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💻 Cấu hình: Intel i5-11400F, 16GB RAM, CPU training")
    print("=" * 50)
    
    try:
        # Training
        model, dataset = train_simple_model()
        
        # Testing
        test_simple_model(model, dataset)
        
        print(f"\n🎉 HOÀN THÀNH!")
        print(f"⏰ Kết thúc lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    main()
