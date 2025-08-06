#!/usr/bin/env python3
"""
Script chính cho dự án PhoBERT_SAM
Huấn luyện và đánh giá các mô hình Intent Recognition, Entity Extraction và Command Processing
"""

import argparse
import torch
import numpy as np
from data import DataProcessor
from training import Trainer
from models import create_model
from config import training_config, model_config, command_config
from utils import extract_entities_from_predictions, format_prediction_output
import json

def set_seed(seed: int):
    """Set random seed để đảm bảo reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_models():
    """Huấn luyện các mô hình Intent Recognition, Entity Extraction và Command Processing"""
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHOBERT_SAM ===")
    
    # Set seed
    set_seed(training_config.seed)
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Tải dataset
    print("Đang tải dataset...")
    dataset = data_processor.load_dataset("nlp_command_dataset.json")
    print(f"Đã tải {len(dataset)} mẫu dữ liệu")
    
    # Chuẩn bị dữ liệu cho Intent Recognition
    print("Chuẩn bị dữ liệu cho Intent Recognition...")
    intent_data = data_processor.prepare_intent_data(dataset)
    train_intent, val_intent = data_processor.split_dataset(intent_data)
    print(f"Intent data - Train: {len(train_intent)}, Val: {len(val_intent)}")
    
    # Chuẩn bị dữ liệu cho Entity Extraction
    print("Chuẩn bị dữ liệu cho Entity Extraction...")
    entity_data = data_processor.prepare_entity_data(dataset)
    train_entity, val_entity = data_processor.split_dataset(entity_data)
    print(f"Entity data - Train: {len(train_entity)}, Val: {len(val_entity)}")
    
    # Chuẩn bị dữ liệu cho Command Processing
    print("Chuẩn bị dữ liệu cho Command Processing...")
    command_data = data_processor.prepare_command_data(dataset)
    train_command, val_command = data_processor.split_dataset(command_data)
    print(f"Command data - Train: {len(train_command)}, Val: {len(val_command)}")
    
    # Huấn luyện Intent Recognition Model
    print("\n=== HUẤN LUYỆN INTENT RECOGNITION MODEL ===")
    intent_trainer = Trainer(model_type="intent")
    intent_trainer.train_intent_model(train_intent, val_intent)
    
    # Huấn luyện Entity Extraction Model
    print("\n=== HUẤN LUYỆN ENTITY EXTRACTION MODEL ===")
    entity_trainer = Trainer(model_type="entity")
    entity_trainer.train_entity_model(train_entity, val_entity)
    
    # Huấn luyện Command Processing Model
    print("\n=== HUẤN LUYỆN COMMAND PROCESSING MODEL ===")
    command_trainer = Trainer(model_type="command")
    command_trainer.train_command_model(train_command, val_command)
    
    # Huấn luyện Unified Model
    print("\n=== HUẤN LUYỆN UNIFIED MODEL ===")
    unified_trainer = Trainer(model_type="unified")
    unified_trainer.train_unified_model(train_intent, val_intent, train_entity, val_entity, train_command, val_command)
    
    print("\n=== HOÀN THÀNH HUẤN LUYỆN ===")

def test_models():
    """Test các mô hình đã huấn luyện"""
    print("=== TESTING MÔ HÌNH PHOBERT_SAM ===")
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Tải dataset
    dataset = data_processor.load_dataset("nlp_command_dataset.json")
    
    # Test Intent Recognition
    print("\n--- Test Intent Recognition ---")
    intent_model = create_model("intent")
    intent_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_intent_model.pth"))
    intent_model.eval()
    
    # Test một số mẫu
    test_samples = [
        "nhắc tôi lúc 5 giờ chiều",
        "alo cho bố",
        "gửi tin nhắn cho mẹ: yêu mẹ nhiều",
        "đặt báo thức lúc 7 giờ sáng",
        "gọi Minh ngay"
    ]
    
    for sample in test_samples:
        # Encode text
        encoding = data_processor.tokenizer(
            sample,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            logits = intent_model(encoding["input_ids"], encoding["attention_mask"])
            predicted = torch.argmax(logits, dim=1)
            intent = data_processor.intent_id2label[predicted.item()]
        
        print(f"Input: {sample}")
        print(f"Predicted Intent: {intent}")
        print("-" * 50)
    
    # Test Entity Extraction
    print("\n--- Test Entity Extraction ---")
    entity_model = create_model("entity")
    entity_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_entity_model.pth"))
    entity_model.eval()
    
    test_sample = "gửi tin nhắn cho Minh rằng yêu mẹ nhiều"
    encoding = data_processor.tokenizer(
        test_sample,
        truncation=True,
        padding="max_length",
        max_length=model_config.max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits = entity_model(encoding["input_ids"], encoding["attention_mask"])
        predictions = torch.argmax(logits, dim=2)
    
    # Decode predictions
    tokens = data_processor.tokenizer.tokenize(test_sample)
    predicted_labels = predictions[0][:len(tokens)].cpu().numpy()
    
    print(f"Input: {test_sample}")
    print("Token-level predictions:")
    for token, label_id in zip(tokens, predicted_labels):
        label = data_processor.entity_id2label[label_id]
        print(f"  {token}: {label}")
    print("-" * 50)
    
    # Test Command Processing
    print("\n--- Test Command Processing ---")
    command_model = create_model("command")
    command_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_command_model.pth"))
    command_model.eval()
    
    for sample in test_samples:
        encoding = data_processor.tokenizer(
            sample,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = command_model(encoding["input_ids"], encoding["attention_mask"])
            predicted = torch.argmax(logits, dim=1)
            command = data_processor.command_id2label[predicted.item()]
        
        print(f"Input: {sample}")
        print(f"Predicted Command: {command}")
        print("-" * 50)
    
    # Test Unified Model
    print("\n--- Test Unified Model ---")
    unified_model = create_model("unified")
    unified_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_unified_model.pth"))
    unified_model.eval()
    
    test_sample = "gửi tin nhắn cho Minh rằng yêu mẹ nhiều"
    encoding = data_processor.tokenizer(
        test_sample,
        truncation=True,
        padding="max_length",
        max_length=model_config.max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        intent_logits, entity_logits, command_logits = unified_model(encoding["input_ids"], encoding["attention_mask"])
        
        # Intent prediction
        intent_pred = torch.argmax(intent_logits, dim=1)
        intent = data_processor.intent_id2label[intent_pred.item()]
        
        # Command prediction
        command_pred = torch.argmax(command_logits, dim=1)
        command = data_processor.command_id2label[command_pred.item()]
        
        # Entity prediction
        entity_preds = torch.argmax(entity_logits, dim=2)
        tokens = data_processor.tokenizer.tokenize(test_sample)
        predicted_labels = entity_preds[0][:len(tokens)].cpu().numpy()
        
        # Extract entities
        entities = extract_entities_from_predictions(tokens, predicted_labels, data_processor.entity_id2label)
    
    print(f"Input: {test_sample}")
    print(f"Intent: {intent}")
    print(f"Command: {command}")
    print("Entities:")
    for entity in entities:
        print(f"  - {entity['type']}: '{entity['text']}'")
    print("-" * 50)

def predict_command(text: str):
    """Dự đoán command cho một câu input"""
    print(f"=== DỰ ĐOÁN COMMAND CHO: '{text}' ===")
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Load Unified Model
    unified_model = create_model("unified")
    unified_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_unified_model.pth"))
    unified_model.eval()
    
    # Encode text
    encoding = data_processor.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=model_config.max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        intent_logits, entity_logits, command_logits = unified_model(encoding["input_ids"], encoding["attention_mask"])
        
        # Intent prediction
        intent_pred = torch.argmax(intent_logits, dim=1)
        intent = data_processor.intent_id2label[intent_pred.item()]
        
        # Command prediction
        command_pred = torch.argmax(command_logits, dim=1)
        command = data_processor.command_id2label[command_pred.item()]
        
        # Entity prediction
        entity_preds = torch.argmax(entity_logits, dim=2)
        tokens = data_processor.tokenizer.tokenize(text)
        predicted_labels = entity_preds[0][:len(tokens)].cpu().numpy()
        
        # Extract entities
        entities = extract_entities_from_predictions(tokens, predicted_labels, data_processor.entity_id2label)
    
    # Format output
    result = {
        "input": text,
        "intent": intent,
        "command": command,
        "entities": entities
    }
    
    print("Kết quả dự đoán:")
    print(f"  Input: {result['input']}")
    print(f"  Intent: {result['intent']}")
    print(f"  Command: {result['command']}")
    print("  Entities:")
    for entity in result['entities']:
        print(f"    - {entity['type']}: '{entity['text']}'")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="PhoBERT_SAM - Intent Recognition, Entity Extraction và Command Processing")
    parser.add_argument("--mode", choices=["train", "test", "predict", "both"], default="both",
                       help="Chế độ chạy: train, test, predict, hoặc both")
    parser.add_argument("--text", type=str, default="",
                       help="Text để dự đoán (chỉ dùng với mode predict)")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        train_models()
    
    if args.mode in ["test", "both"]:
        test_models()
    
    if args.mode == "predict":
        if args.text:
            predict_command(args.text)
        else:
            print("Vui lòng cung cấp text để dự đoán với --text")

if __name__ == "__main__":
    main() 