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
from config import training_config, model_config, command_config, intent_config
from utils import extract_entities_from_predictions, format_prediction_output
import json

def set_seed(seed: int):
    """Set random seed để đảm bảo reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_models():
    """Huấn luyện các mô hình Intent Recognition, Entity Extraction và Command Processing cho người cao tuổi"""
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHOBERT_SAM CHO NGƯỜI CAO TUỔI ===")
    
    # Set seed
    set_seed(training_config.seed)
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Tải dataset cho người cao tuổi
    print("Đang tải dataset cho người cao tuổi...")
    dataset = data_processor.load_dataset("elderly_command_dataset_reduced.json")
    print(f"Đã tải {len(dataset)} mẫu dữ liệu cho người cao tuổi")
    
    # Data augmentation cho Intent Recognition
    print("Thực hiện data augmentation cho Intent Recognition...")
    augmented_dataset = data_processor.augment_intent_data(dataset, augmentation_factor=0.3)
    print(f"Sau augmentation: {len(augmented_dataset)} mẫu dữ liệu")
    
    # Chuẩn bị dữ liệu cho Intent Recognition với confidence
    print("Chuẩn bị dữ liệu cho Intent Recognition nâng cao...")
    intent_data = data_processor.prepare_intent_data_with_confidence(augmented_dataset)
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
    
    # Huấn luyện Intent Recognition Model nâng cao
    print("\n=== HUẤN LUYỆN INTENT RECOGNITION MODEL NÂNG CAO ===")
    print(f"Sử dụng confidence threshold: {intent_config.confidence_threshold}")
    print(f"Số intent classes: {intent_config.num_intents}")
    print(f"Intent labels: {intent_config.intent_labels}")
    
    trainer = Trainer()
    intent_model = trainer.train_intent_model_with_confidence(
        train_intent, val_intent
    )
    
    # Huấn luyện Entity Extraction Model
    print("\n=== HUẤN LUYỆN ENTITY EXTRACTION MODEL ===")
    print(f"Số entity classes: {command_config.num_commands}")
    print(f"Entity labels: {command_config.command_labels}")
    
    entity_model = trainer.train_entity_model(
        train_entity, val_entity
    )
    
    # Huấn luyện Command Processing Model
    print("\n=== HUẤN LUYỆN COMMAND PROCESSING MODEL ===")
    print(f"Số command classes: {command_config.num_commands}")
    print(f"Command labels: {command_config.command_labels}")
    
    command_model = trainer.train_command_model(
        train_command, val_command
    )
    
    # Huấn luyện Unified Model
    print("\n=== HUẤN LUYỆN UNIFIED MODEL ===")
    print("Kết hợp cả 3 tác vụ: Intent + Entity + Command")
    
    unified_model = trainer.train_unified_model(
        train_intent, val_intent,
        train_entity, val_entity,
        train_command, val_command
    )
    
    print("\n=== HOÀN THÀNH HUẤN LUYỆN ===")
    print("Tất cả các mô hình đã được huấn luyện thành công!")
    print("Các mô hình được tối ưu cho người cao tuổi với 9 tác vụ cơ bản:")
    print("1. Gọi điện thoại (call)")
    print("2. Gửi tin nhắn (send-mess)")
    print("3. Xem video/nghe nhạc (play-media)")
    print("4. Kiểm tra thời tiết (check-weather)")
    print("5. Đặt báo thức (set-alarm)")
    print("6. Nhắc nhở thuốc (set-reminder)")
    print("7. Đọc tin tức (read-news)")
    print("8. Kiểm tra sức khỏe (check-health-status)")
    print("9. Trò chuyện chung (general-conversation)")

def test_models():
    """Test các mô hình đã huấn luyện với confidence validation"""
    print("=== TESTING MÔ HÌNH PHOBERT_SAM NÂNG CAO ===")
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Tải dataset
    dataset = data_processor.load_dataset("nlp_command_dataset.json")
    
    # Test Intent Recognition nâng cao
    print("\n--- Test Intent Recognition nâng cao ---")
    intent_model = create_model("intent")
    intent_model.load_state_dict(torch.load(f"{training_config.output_dir}/best_intent_model.pth"))
    intent_model.eval()
    
    # Test một số mẫu với confidence validation
    test_samples = [
        "nhắc tôi lúc 5 giờ chiều",
        "alo cho bố",
        "gửi tin nhắn cho mẹ: yêu mẹ nhiều",
        "đặt báo thức lúc 7 giờ sáng",
        "gọi Minh ngay",
        "thời tiết hôm nay thế nào",
        "tôi muốn nghe nhạc",
        "đọc tin tức cho tôi",
        "tôi cảm thấy mệt mỏi",
        "hôm nay trời đẹp quá"
    ]
    
    for text in test_samples:
        # Tokenize
        encoding = data_processor.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        # Predict với confidence
        predicted_intents, confidences = intent_model.predict_with_confidence(
            encoding["input_ids"], 
            encoding["attention_mask"], 
            intent_config.confidence_threshold
        )
        
        # Validate prediction
        if predicted_intents[0] != -1:
            predicted_intent = intent_config.intent_labels[predicted_intents[0]]
            confidence = confidences[0].item()
            
            # Validation với rule-based checks
            validation = data_processor.validate_intent_prediction(
                predicted_intent, confidence, text
            )
            
            print(f"\nText: '{text}'")
            print(f"Predicted Intent: {predicted_intent}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Validation: {'✓' if validation['is_valid'] else '✗'}")
            
            if validation['warnings']:
                print(f"Warnings: {', '.join(validation['warnings'])}")
            if validation['suggestions']:
                print(f"Suggestions: {', '.join(validation['suggestions'])}")
        else:
            print(f"\nText: '{text}'")
            print("Predicted Intent: UNKNOWN (confidence too low)")
            print(f"Confidence: {confidences[0].item():.3f}")
    
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