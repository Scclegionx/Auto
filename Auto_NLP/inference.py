#!/usr/bin/env python3
"""
Inference script cho PhoBERT_SAM
Sử dụng để dự đoán command từ text input trong production
"""

import torch
import json
from typing import Dict, List, Optional
from data import DataProcessor
from models import create_model
from config import training_config, model_config
from utils import extract_entities_from_predictions, convert_entities_to_dict_list

class PhoBERTSAMInference:
    """Class để thực hiện inference với PhoBERT_SAM"""
    
    def __init__(self, model_path: str = None):
        """
        Khởi tạo inference engine
        
        Args:
            model_path: Đường dẫn đến unified model (nếu None sẽ dùng default)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataProcessor()
        
        # Load model
        if model_path is None:
            model_path = f"{training_config.output_dir}/best_unified_model.pth"
        
        self.model = create_model("unified")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
    
    def predict(self, text: str) -> Dict:
        """
        Dự đoán intent, command và entities từ text
        
        Args:
            text: Input text
            
        Returns:
            Dict chứa kết quả dự đoán
        """
        # Encode text
        encoding = self.data_processor.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=model_config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            intent_logits, entity_logits, command_logits = self.model(input_ids, attention_mask)
            
            # Intent prediction
            intent_pred = torch.argmax(intent_logits, dim=1)
            intent = self.data_processor.intent_id2label[intent_pred.item()]
            
            # Command prediction
            command_pred = torch.argmax(command_logits, dim=1)
            command = self.data_processor.command_id2label[command_pred.item()]
            
            # Entity prediction
            entity_preds = torch.argmax(entity_logits, dim=2)
            tokens = self.data_processor.tokenizer.tokenize(text)
            predicted_labels = entity_preds[0][:len(tokens)].cpu().numpy()
            
            # Extract entities
            entities = extract_entities_from_predictions(tokens, predicted_labels, self.data_processor.entity_id2label)
        
        # Format result với cấu trúc List Dict
        result = {
            "input": text,
            "intent": intent,
            "command": command,
            "entities": convert_entities_to_dict_list(entities),  # Chuyển sang List Dict format
            "confidence": {
                "intent": torch.softmax(intent_logits, dim=1).max().item(),
                "command": torch.softmax(command_logits, dim=1).max().item()
            }
        }
        
        return result
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Dự đoán cho nhiều text cùng lúc
        
        Args:
            texts: List các input text
            
        Returns:
            List các kết quả dự đoán
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def get_command_execution_params(self, prediction: Dict) -> Dict:
        """
        Trích xuất tham số để thực thi command
        
        Args:
            prediction: Kết quả dự đoán từ predict()
            
        Returns:
            Dict chứa tham số để thực thi command
        """
        command = prediction["command"]
        entities = prediction["entities"]  # Đã là List Dict format
        
        params = {
            "command": command,
            "intent": prediction["intent"],
            "confidence": prediction["confidence"]
        }
        
        # Trích xuất tham số dựa trên loại command
        if command == "set_alarm":
            # Tìm thời gian
            time_entities = [e for e in entities if e.get("label") == "TIME"]
            if time_entities:
                params["time"] = time_entities[0]["text"]
        
        elif command == "make_call":
            # Tìm người nhận
            receiver_entities = [e for e in entities if e.get("label") == "RECEIVER"]
            if receiver_entities:
                params["receiver"] = receiver_entities[0]["text"]
        
        elif command == "send_message":
            # Tìm người nhận và nội dung
            receiver_entities = [e for e in entities if e.get("label") == "RECEIVER"]
            message_entities = [e for e in entities if e.get("label") == "MESSAGE"]
            
            if receiver_entities:
                params["receiver"] = receiver_entities[0]["text"]
            if message_entities:
                params["message"] = message_entities[0]["text"]
        
        return params
    
    def format_output(self, prediction: Dict) -> str:
        """
        Format kết quả dự đoán thành string đẹp
        
        Args:
            prediction: Kết quả dự đoán
            
        Returns:
            String đã format
        """
        output = f"Input: {prediction['input']}\n"
        output += f"Intent: {prediction['intent']}\n"
        output += f"Command: {prediction['command']}\n"
        output += f"Confidence: Intent={prediction['confidence']['intent']:.3f}, Command={prediction['confidence']['command']:.3f}\n"
        output += "Entities:\n"
        
        if prediction['entities']:
            for entity in prediction['entities']:
                output += f"  - {entity['label']}: '{entity['text']}'\n"
        else:
            output += "  - Không có entities được tìm thấy\n"
        
        return output

def main():
    """Demo sử dụng inference engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PhoBERT_SAM Inference")
    parser.add_argument("--text", type=str, required=True, help="Text để dự đoán")
    parser.add_argument("--model_path", type=str, default=None, help="Đường dẫn đến model")
    parser.add_argument("--output_format", choices=["json", "text"], default="text", help="Format output")
    
    args = parser.parse_args()
    
    # Khởi tạo inference engine
    inference = PhoBERTSAMInference(args.model_path)
    
    # Dự đoán
    result = inference.predict(args.text)
    
    # Format output
    if args.output_format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(inference.format_output(result))
    
    # Hiển thị tham số thực thi command
    params = inference.get_command_execution_params(result)
    print("\nCommand Execution Parameters:")
    print(json.dumps(params, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 