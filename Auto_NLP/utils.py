"""
Utilities cho dự án PhoBERT_SAM
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_training_history(train_losses: List[float], val_losses: List[float], 
                             train_metrics: List[float], val_metrics: List[float],
                             metric_name: str = "Accuracy"):
    """Vẽ biểu đồ training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training và Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metric plot
    ax2.plot(train_metrics, label=f'Train {metric_name}')
    ax2.plot(val_metrics, label=f'Validation {metric_name}')
    ax2.set_title(f'Training và Validation {metric_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_report(y_true: List[int], y_pred: List[int], 
                              labels: List[str], title: str = "Classification Report"):
    """In classification report"""
    print(f"\n{title}")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_entities_from_predictions(tokens: List[str], predictions: List[int], 
                                    entity_id2label: Dict[int, str]) -> List[Dict]:
    """Trích xuất entities từ token-level predictions"""
    entities = []
    current_entity = None
    
    for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
        label = entity_id2label[pred_id]
        
        if label.startswith('B-'):
            # Bắt đầu entity mới
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': label[2:],  # Bỏ 'B-' prefix
                'text': token,
                'start': i,
                'label': label  # Thêm label gốc
            }
        elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
            # Tiếp tục entity hiện tại
            current_entity['text'] += ' ' + token
        else:
            # Kết thúc entity hiện tại
            if current_entity:
                current_entity['end'] = i - 1
                entities.append(current_entity)
                current_entity = None
    
    # Thêm entity cuối cùng nếu có
    if current_entity:
        current_entity['end'] = len(tokens) - 1
        entities.append(current_entity)
    
    return entities

def convert_entities_to_dict_list(entities: List[Dict]) -> List[Dict]:
    """Chuyển đổi entities sang format List Dict với text và label"""
    dict_list = []
    for entity in entities:
        dict_list.append({
            "text": entity["text"],
            "label": entity["type"].upper()  # Chuyển type thành label
        })
    return dict_list

def format_prediction_output(text: str, intent: str, entities: List[Dict]) -> str:
    """Format kết quả prediction thành string đẹp"""
    output = f"Input: {text}\n"
    output += f"Intent: {intent}\n"
    output += "Entities:\n"
    
    if entities:
        for entity in entities:
            output += f"  - {entity['type']}: '{entity['text']}' (tokens {entity['start']}-{entity['end']})\n"
    else:
        output += "  - Không có entities được tìm thấy\n"
    
    return output

def calculate_metrics_for_entity_extraction(y_true: List[List[int]], y_pred: List[List[int]], 
                                          entity_id2label: Dict[int, str]) -> Dict[str, float]:
    """Tính metrics cho entity extraction"""
    all_true = []
    all_pred = []
    
    for true_seq, pred_seq in zip(y_true, y_pred):
        # Chỉ tính cho non-padding tokens
        for true_label, pred_label in zip(true_seq, pred_seq):
            if true_label != -100:  # -100 là padding label
                all_true.append(true_label)
                all_pred.append(pred_label)
    
    # Tính các metrics
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_pred, average='weighted', zero_division=0
    )
    
    accuracy = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true) if all_true else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model_info(model_path: str, config: Dict, metrics: Dict):
    """Lưu thông tin model"""
    import json
    import os
    
    info = {
        'model_path': model_path,
        'config': config,
        'metrics': metrics,
        'timestamp': str(torch.datetime.now())
    }
    
    info_path = model_path.replace('.pth', '_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def load_model_info(model_path: str) -> Dict:
    """Tải thông tin model"""
    import json
    
    info_path = model_path.replace('.pth', '_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def set_random_seed(seed: int = 42):
    """Set random seed cho reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Tính kích thước model tính bằng MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb 