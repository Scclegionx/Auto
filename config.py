import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Cấu hình cho mô hình PhoBERT - Tối ưu cho CPU training"""
    # Chọn model size: "base" hoặc "large"
    model_size: str = "base"  # "base" hoặc "large"
    
    @property
    def model_name(self) -> str:
        """Lấy tên model dựa trên size"""
        if self.model_size == "large":
            return "vinai/phobert-large"
        else:
            return "vinai/phobert-base"
    
    @property
    def hidden_size(self) -> int:
        """Lấy hidden size dựa trên model size"""
        if self.model_size == "large":
            return 1024
        else:
            return 768
    
    @property
    def num_layers(self) -> int:
        """Lấy số layers dựa trên model size"""
        if self.model_size == "large":
            return 24
        else:
            return 12
    
    @property
    def num_attention_heads(self) -> int:
        """Lấy số attention heads dựa trên model size"""
        if self.model_size == "large":
            return 16
        else:
            return 12
    
    # Các tham số tối ưu cho CPU training (16GB RAM, i5-11400F)
    max_length: int = 128
    batch_size: int = 8  # Giảm từ 16 xuống 8 cho CPU
    learning_rate: float = 1e-5  # Giảm từ 2e-5 xuống 1e-5 cho ổn định
    num_epochs: int = 15  # Tăng từ 10 lên 15 để bù đắp batch size nhỏ
    warmup_steps: int = 50  # Giảm warmup steps
    weight_decay: float = 0.01
    dropout: float = 0.1
    
    # CPU-specific settings
    num_workers: int = 2  # Số worker cho DataLoader
    pin_memory: bool = False  # Tắt pin_memory cho CPU
    gradient_accumulation_steps: int = 2  # Gradient accumulation để mô phỏng batch size lớn hơn
    
    def __post_init__(self):
        """Validate model size"""
        if self.model_size not in ["base", "large"]:
            raise ValueError("model_size phải là 'base' hoặc 'large'")

@dataclass
class IntentConfig:
    """Cấu hình cho Intent Recognition nâng cao"""
    num_intents: int = 17  # Tất cả các loại command trong dataset
    intent_labels: List[str] = None
    
    # Confidence threshold cho intent recognition
    confidence_threshold: float = 0.7
    unknown_intent_label: str = "unknown"
    
    # Attention mechanism settings
    use_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Multi-layer classifier settings
    use_multi_layer: bool = True
    hidden_layer_sizes: List[int] = None  # Sẽ được set động dựa trên model size
    
    # CRF settings
    use_crf: bool = False
    
    # Ensemble settings
    use_ensemble: bool = False
    ensemble_method: str = "weighted"  # "weighted", "voting", "stacking"
    
    # Advanced features
    use_confidence_scoring: bool = True
    use_batch_norm: bool = True
    use_residual_connections: bool = True
    
    def __post_init__(self):
        if self.intent_labels is None:
            self.intent_labels = [
                "call", "check-health-status", "check-weather", "express-emotion",
                "express-fatigue", "find-information", "general-conversation", 
                "general-request", "play-media", "read-news", "report-symptom",
                "request-comfort", "request-entertainment", "request-instruction",
                "send-mess", "set-alarm", "set-reminder"
            ]
        
        # Set hidden layer sizes dựa trên model size
        if self.hidden_layer_sizes is None:
            # Sử dụng giá trị mặc định thay vì relative import
            base_size = 768  # PhoBERT-base hidden size
            self.hidden_layer_sizes = [base_size, base_size // 2, base_size // 4]

@dataclass
class EntityConfig:
    """Cấu hình cho Entity Extraction"""
    entity_labels: List[str] = None
    
    def __post_init__(self):
        if self.entity_labels is None:
            # IOB2 format cho entity extraction
            self.entity_labels = [
                "O",  # Outside
                "B-RECEIVER", "I-RECEIVER",  # Người nhận
                "B-TIME", "I-TIME",  # Thời gian
                "B-MESSAGE", "I-MESSAGE"  # Nội dung tin nhắn
            ]
    
    @property
    def num_entities(self) -> int:
        return len(self.entity_labels)

@dataclass
class CommandConfig:
    """Cấu hình cho Command Processing"""
    command_labels: List[str] = None
    
    def __post_init__(self):
        if self.command_labels is None:
            # Các command có thể thực thi - mapping từ intent sang command
            self.command_labels = [
                "make_call",              # call
                "check_health_status",    # check-health-status
                "check_weather",          # check-weather
                "express_emotion",        # express-emotion
                "express_fatigue",        # express-fatigue
                "find_information",       # find-information
                "general_conversation",   # general-conversation
                "general_request",        # general-request
                "play_media",             # play-media
                "read_news",              # read-news
                "report_symptom",         # report-symptom
                "request_comfort",        # request-comfort
                "request_entertainment",  # request-entertainment
                "request_instruction",    # request-instruction
                "send_message",           # send-mess
                "set_alarm",             # set-alarm
                "set_reminder",          # set-reminder
                "unknown"                # Không xác định
            ]
    
    @property
    def num_commands(self) -> int:
        return len(self.command_labels)

@dataclass
class TrainingConfig:
    """Cấu hình cho training - Tối ưu cho CPU"""
    output_dir: str = "./models"
    data_dir: str = "./data"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "cpu"  # Force CPU training
    save_steps: int = 200  # Giảm save frequency cho CPU
    eval_steps: int = 200  # Giảm eval frequency
    logging_steps: int = 50  # Tăng logging frequency để theo dõi
    
    # CPU optimization settings
    use_mixed_precision: bool = False  # Tắt mixed precision cho CPU
    max_grad_norm: float = 1.0  # Gradient clipping
    early_stopping_patience: int = 5  # Early stopping
    save_best_only: bool = True  # Chỉ lưu model tốt nhất
    
    def __post_init__(self):
        # Tạo các thư mục cần thiết
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# Khởi tạo các config
model_config = ModelConfig()
intent_config = IntentConfig()
entity_config = EntityConfig()
command_config = CommandConfig()
training_config = TrainingConfig() 