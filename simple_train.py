import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from config import model_config, training_config, intent_config
from reasoning_engine import ReasoningEngine

class SimpleIntentDataset(Dataset):
    """Dataset đơn giản cho Intent Recognition - Cập nhật cho dataset mới"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Sử dụng intent labels từ config
        self.intent_to_id = {intent: idx for idx, intent in enumerate(intent_config.intent_labels)}
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
        
        # Sử dụng intent mapping từ config
        intent_id = self.intent_to_id.get(intent, 0)  # Default to 0 if not found
        if intent not in self.intent_to_id:
            print(f"Warning: Intent '{intent}' not found in config, using default")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(intent_id, dtype=torch.long),
            "text": text,
            "intent": intent
        }

class SimpleIntentModel(nn.Module):
    """Model đơn giản cho Intent Recognition - Cập nhật cho dataset mới"""
    
    def __init__(self, model_name, num_intents):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)  # Tăng dropout cho 27 classes
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_intents)
        
        # Thêm attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.phobert.config.hidden_size,
            num_heads=8,
            dropout=0.2,  # Tăng dropout cho 27 classes
            batch_first=True
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Sử dụng attention
        sequence_output = outputs.last_hidden_state
        attention_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        
        # Global average pooling
        pooled_output = torch.mean(attention_output, dim=1)
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
    """Training model đơn giản - Cập nhật cho dataset mới"""
    print("🚀 BẮT ĐẦU TRAINING ĐƠN GIẢN - DATASET MỚI")
    print("=" * 60)
    
    # Load dataset mới
    dataset_file = "elderly_command_dataset_reduced.json"
    
    print(f"Loading dataset: {dataset_file}")
    dataset = load_dataset(dataset_file)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Hiển thị thống kê dataset
    commands = [item["command"] for item in dataset]
    unique_commands = set(commands)
    print(f"Unique commands: {len(unique_commands)}")
    print(f"Commands: {sorted(unique_commands)}")
    
    # Split dataset
    train_data, val_data = split_dataset(dataset)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Khởi tạo tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model = SimpleIntentModel(model_config.model_name, intent_config.num_intents)
    
    print(f"Model initialized with {intent_config.num_intents} intents")
    print(f"Intent labels: {intent_config.intent_labels}")
    print(f"Dataset commands: {sorted(set(item['command'] for item in dataset))}")
    
    # Tạo datasets
    train_dataset = SimpleIntentDataset(train_data, tokenizer, model_config.max_length)
    val_dataset = SimpleIntentDataset(val_data, tokenizer, model_config.max_length)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True,
        num_workers=model_config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False,
        num_workers=model_config.num_workers
    )
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    total_steps = len(train_loader) * model_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=model_config.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Training on device: {device}")
    print(f"Batch size: {model_config.batch_size}")
    print(f"Effective batch size: {model_config.batch_size * model_config.gradient_accumulation_steps}")
    print(f"Learning rate: {model_config.learning_rate}")
    print(f"Epochs: {model_config.num_epochs}")
    print(f"Gradient accumulation steps: {model_config.gradient_accumulation_steps}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(model_config.num_epochs):
        print(f"\n📚 EPOCH {epoch + 1}/{model_config.num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        accumulated_loss = 0.0
        step_count = 0
        
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / model_config.gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            step_count += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % model_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update metrics
                train_loss += accumulated_loss * model_config.gradient_accumulation_steps
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                accumulated_loss = 0.0
                step_count = 0
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item() * model_config.gradient_accumulation_steps:.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%' if train_total > 0 else '0.00%'
            })
        
        train_acc = 100 * train_correct / train_total
        # Tính avg_train_loss dựa trên số steps thực tế đã update
        effective_steps = len(train_loader) // model_config.gradient_accumulation_steps
        avg_train_loss = train_loss / max(effective_steps, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(training_config.output_dir, "best_simple_intent_model.pth"))
            print(f"✅ Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Print classification report every 3 epochs (vì có nhiều classes hơn)
        if (epoch + 1) % 3 == 0:
            print("\n📊 CLASSIFICATION REPORT:")
            # Lấy unique labels từ validation set
            unique_labels = sorted(set(all_labels))
            unique_target_names = [intent_config.intent_labels[i] for i in unique_labels]
            print(classification_report(
                all_labels, 
                all_predictions, 
                labels=unique_labels,
                target_names=unique_target_names,
                zero_division=0
            ))
    
    print(f"\n🎉 TRAINING HOÀN THÀNH!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(training_config.output_dir, 'best_simple_intent_model.pth')}")
    
    return model, tokenizer

def test_simple_model(model, tokenizer):
    """Test model đơn giản với reasoning engine fallback"""
    print("\n🧪 TESTING MODEL VỚI REASONING ENGINE FALLBACK")
    print("=" * 50)
    
    device = torch.device("cpu")
    model.eval()
    reasoning_engine = ReasoningEngine()
    
    # Tạo intent mapping từ config
    intent_to_id = {intent: idx for idx, intent in enumerate(intent_config.intent_labels)}
    id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}
    
    test_samples = [
        "nhắc tôi lúc 5 giờ chiều",
        "alo cho bố",
        "gửi tin nhắn cho mẹ",
        "đặt báo thức lúc 7 giờ sáng",
        "thời tiết hôm nay thế nào",
        "tôi muốn nghe nhạc",
        "đọc tin tức cho tôi",
        "mở ứng dụng Zalo",
        "tìm kiếm thông tin về bệnh tiểu đường",
        "gọi video cho con gái"
    ]
    
    for text in test_samples:
        print(f"\nInput: '{text}'")
        
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        # Predict với trained model
        with torch.no_grad():
            logits = model(encoding["input_ids"], encoding["attention_mask"])
            predicted = torch.argmax(logits, dim=1)
            intent = id_to_intent[predicted.item()]
            confidence = torch.softmax(logits, dim=1).max().item()
        
        print(f"Trained Model: {intent} (confidence: {confidence:.3f})")
        
        # Kiểm tra confidence và sử dụng reasoning engine nếu cần
        if confidence < intent_config.confidence_threshold or intent == "unknown":
            print(f"⚠️ Confidence thấp, sử dụng reasoning engine...")
            reasoning_result = reasoning_engine.reasoning_predict(text)
            print(f"Reasoning Engine: {reasoning_result['intent']} (confidence: {reasoning_result['confidence']:.3f})")
            if 'explanation' in reasoning_result and reasoning_result['explanation']:
                print(f"Explanation: {reasoning_result['explanation']}")
        else:
            print(f"✅ Sử dụng kết quả từ trained model")
        
        print("-" * 50)

def main():
    """Main function"""
    print("🎯 PHOBERT_SAM - SIMPLE TRAINING")
    print("=" * 50)
    print(f"⏰ Bắt đầu lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💻 Cấu hình: Intel i5-11400F, 16GB RAM, CPU training")
    print(f"📊 Dataset: {intent_config.num_intents} intent classes")
    print(f"⚙️ Batch size: {model_config.batch_size}, Gradient accumulation: {model_config.gradient_accumulation_steps}")
    print(f"📈 Learning rate: {model_config.learning_rate}, Epochs: {model_config.num_epochs}")
    print("=" * 50)
    
    try:
        # Training
        model, tokenizer = train_simple_model()
        
        # Testing
        test_simple_model(model, tokenizer)
        
        print(f"\n🎉 HOÀN THÀNH!")
        print(f"⏰ Kết thúc lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
