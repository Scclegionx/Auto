import os
from dataclasses import dataclass
from typing import List

class ModelConfig:

    def __init__(self):
        self.model_name = "vinai/phobert-large"
        self.model_size = "large"
        
        # Tham số cơ bản - tối ưu cho GPU 6GB
        # Điều chỉnh dựa trên environment variables nếu có
        self.max_length = int(os.environ.get('MAX_LENGTH', 64))   # Tối ưu cho GPU 6GB
        self.batch_size = int(os.environ.get('BATCH_SIZE', 2))   # Tối ưu cho GPU 6GB
        self.num_epochs = 10   # Tăng epochs cho GPU
        self.learning_rate = 3e-5  
        self.freeze_layers = 4 
        
        # Device và precision được quản lý bởi TrainingConfig
        
        self.gradient_accumulation_steps = 4  # Tương đương batch 8 logic  
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 1000  
        
        self.num_workers = 0  # Windows - tránh multiprocessing issues
        self.pin_memory = True  # CUDA cần pin_memory
        self.persistent_workers = False  # Không cần khi num_workers=0
        self.dropout = 0.1  
        self.optimizer = "adamw"
    
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
    
    def __post_init__(self):
        """Validate model size"""
        if self.model_size not in ["base", "large"]:
            raise ValueError("model_size phải là 'base' hoặc 'large'")

@dataclass
class IntentConfig:
    """Cấu hình cho Intent Recognition nâng cao"""
    num_intents: int = 26  
    intent_labels: List[str] = None
    
    confidence_threshold: float = 0.7
    unknown_intent_label: str = "unknown"
    
    use_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    use_multi_layer: bool = True
    hidden_layer_sizes: List[int] = None  
    
    use_crf: bool = False
    
    use_ensemble: bool = False
    ensemble_method: str = "weighted"  # "weighted", "voting", "stacking"
    
    use_confidence_scoring: bool = True
    use_batch_norm: bool = True
    use_residual_connections: bool = True
    
    def __post_init__(self):
        if self.intent_labels is None:
            self.intent_labels = [
                "adjust-settings", "app-tutorial", "browse-social-media", "call", 
                "check-device-status", "check-health-status", "check-messages", 
                "check-weather", "control-device", "general-conversation", 
                "make-video-call", "navigation-help", "open-app", 
                "open-app-action", "play-audio", "play-content", "play-media", 
                "provide-instructions", "read-content", "read-news", "search-content", 
                "search-internet", "send-mess", "set-alarm", 
                "set-reminder", "view-content"
            ]
        
        if self.hidden_layer_sizes is None:
            # Sử dụng giá trị mặc định cho PhoBERT-large
            large_size = 1024  # PhoBERT-large hidden size
            self.hidden_layer_sizes = [large_size, large_size // 2, large_size // 4]

@dataclass
class EntityConfig:
    """Cấu hình cho Entity Extraction - Cập nhật theo dataset mới"""
    entity_labels: List[str] = None
    
    def __post_init__(self):
        if self.entity_labels is None:
            self.entity_labels = [
                "O",  # Outside
                "B-PERSON", "I-PERSON",      # Người
                "B-TIME", "I-TIME",          # Thời gian
                "B-LOCATION", "I-LOCATION",  # Địa điểm
                "B-MESSAGE", "I-MESSAGE",    # Tin nhắn
                "B-PLATFORM", "I-PLATFORM",  # Nền tảng
                "B-CONTENT", "I-CONTENT",    # Nội dung
                "B-QUERY", "I-QUERY",        # Truy vấn
                "B-REMINDER", "I-REMINDER",  # Nhắc nhở
                "B-APP", "I-APP",           # Ứng dụng
                "B-DEVICE", "I-DEVICE",      # Thiết bị
                "B-ACTION", "I-ACTION",      # Hành động
                "B-RECEIVER", "I-RECEIVER"   # Người nhận (đặc biệt)
            ]
    
    @property
    def num_entity_labels(self) -> int:
        return len(self.entity_labels)

@dataclass
class ValueConfig:
    """Cấu hình cho Value Extraction - Mới thêm"""
    value_labels: List[str] = None
    
    def __post_init__(self):
        if self.value_labels is None:
            self.value_labels = [
                "O",  # Outside
                "B-VALUE", "I-VALUE",        # Giá trị chung
                "B-TEXT", "I-TEXT",          # Văn bản
                "B-NUMBER", "I-NUMBER",      # Số
                "B-DATE", "I-DATE",          # Ngày
                "B-TIME", "I-TIME",          # Thời gian
                "B-URL", "I-URL",            # URL
                "B-EMAIL", "I-EMAIL",        # Email
                "B-PHONE", "I-PHONE",        # Số điện thoại
                "B-ADDRESS", "I-ADDRESS",    # Địa chỉ
                "B-MESSAGE", "I-MESSAGE"     # Nội dung tin nhắn (đặc biệt)
            ]
    
    @property
    def num_value_labels(self) -> int:
        return len(self.value_labels)

@dataclass
class CommandConfig:
    command_labels: List[str] = None
    
    def __post_init__(self):
        if self.command_labels is None:
            self.command_labels = [
                "adjust-settings", "app-tutorial", "browse-social-media", "call", 
                "check-device-status", "check-health-status", "check-messages", 
                "check-weather", "control-device", "general-conversation", 
                "make-video-call", "navigation-help", "open-app", "open-app-action", 
                "play-audio", "play-content", "play-media", "provide-instructions", 
                "read-content", "read-news", "search-content", "search-internet", 
                "send-mess", "set-alarm", "set-reminder", "view-content", "unknown"
            ]
    
    @property
    def num_command_labels(self) -> int:
        return len(self.command_labels)

@dataclass
class TrainingConfig:
    """Cấu hình cho training - Nguồn chân lý cho device & precision"""
    output_dir: str = "./models"
    data_dir: str = "./data"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "auto"
    
    # Tham số training
    save_steps: int = 25      
    eval_steps: int = 25     
    logging_steps: int = 5   
    early_stopping_patience: int = 10  
    save_total_limit: int = 5  
    
    # Precision settings (gộp thành một cờ)
    use_amp: bool = False  # Tắt AMP để tránh dtype mismatch
    max_grad_norm: float = 1.0
    save_best_only: bool = True 
    use_gradient_checkpointing: bool = False  # Tắt để tránh lỗi
    max_memory_usage: str = "auto"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Khởi tạo các config
model_config = ModelConfig()
intent_config = IntentConfig()
entity_config = EntityConfig()
value_config = ValueConfig()
command_config = CommandConfig()
training_config = TrainingConfig()

# Kiểm định kích thước nhãn ngay khi load
assert intent_config.num_intents == len(intent_config.intent_labels), f"Intent mismatch: {intent_config.num_intents} vs {len(intent_config.intent_labels)}"
assert entity_config.num_entity_labels > 0, f"Entity labels empty: {entity_config.num_entity_labels}"
assert value_config.num_value_labels > 0, f"Value labels empty: {value_config.num_value_labels}"
assert command_config.num_command_labels > 0, f"Command labels empty: {command_config.num_command_labels}"

# Removed print statements to avoid Unicode errors 
