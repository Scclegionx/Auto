import os
from dataclasses import dataclass
from typing import List

class ModelConfig:

    def __init__(self):
        self.model_name = "vinai/phobert-large"
        self.model_size = "large"
        
        # Tham số cơ bản
        self.max_length = 512  
        self.batch_size = 32     
        self.num_epochs = 20    
        self.learning_rate = 3e-5  
        self.freeze_layers = 4 
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True 
        self.use_amp = True
        self.gradient_checkpointing = True  
        self.use_mixed_precision = True
        
        self.gradient_accumulation_steps = 2  
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 1000  
        
        self.num_workers = 8 
        self.pin_memory = True
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
        """Validate model size và setup device"""
        if self.model_size not in ["base", "large"]:
            raise ValueError("model_size phải là 'base' hoặc 'large'")
        
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ Auto-detected device: {self.device}")
            
            if self.device == "cuda":
                print(f"🎮 GPU: {torch.cuda.get_device_name()}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

@dataclass
class IntentConfig:
    """Cấu hình cho Intent Recognition nâng cao"""
    num_intents: int = 28  
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
                "make-call", "make-video-call", "navigation-help", "open-app", 
                "open-app-action", "play-audio", "play-content", "play-media", 
                "provide-instructions", "read-content", "read-news", "search-content", 
                "search-internet", "send-message", "send-mess", "set-alarm", 
                "set-reminder", "view-content"
            ]
        
        if self.hidden_layer_sizes is None:
            # Sử dụng giá trị mặc định thay vì relative import
            base_size = 768  # PhoBERT-base hidden size
            self.hidden_layer_sizes = [base_size, base_size // 2, base_size // 4]

@dataclass
class EntityConfig:
    """Cấu hình cho Entity Extraction - Cập nhật theo dataset mới"""
    entity_labels: List[str] = None
    
    def __post_init__(self):
        if self.entity_labels is None:
            self.entity_labels = [
                "O",  # Outside
                "FAMILY_RELATIONSHIP",  # Mối quan hệ gia đình
                "DATE_EXPRESSION",      # Biểu thức ngày tháng
                "CONTACT_PERSON",       # Người liên hệ
                "LOCATION",             # Địa điểm
                "TIME_EXPRESSION",      # Biểu thức thời gian
                "ARTIST_NAME"           # Tên nghệ sĩ
            ]
    
    @property
    def num_entities(self) -> int:
        return len(self.entity_labels)

@dataclass
class ValueConfig:
    """Cấu hình cho Value Extraction - Mới thêm"""
    value_labels: List[str] = None
    
    def __post_init__(self):
        if self.value_labels is None:
            self.value_labels = [
                "TIME_EXPRESSION",      # Biểu thức thời gian
                "MESSAGE_CONTENT",      # Nội dung tin nhắn
                "REMINDER_CONTENT",     # Nội dung nhắc nhở
                "HEALTH_METRIC",        # Chỉ số sức khỏe
                "MEDIA_TYPE",           # Loại media
                "NEWS_CATEGORY",        # Danh mục tin tức
                "DATE_EXPRESSION",      # Biểu thức ngày tháng
                "WEATHER_CONDITION",    # Điều kiện thời tiết
                "MEDIA_CONTENT",        # Nội dung media
                "CALL_TYPE",            # Loại cuộc gọi
                "FREQUENCY",            # Tần suất
                "SYMPTOM",              # Triệu chứng
                "TOPIC_NEWS",           # Chủ đề tin tức
                "NEWS_SOURCE"           # Nguồn tin tức
            ]
    
    @property
    def num_values(self) -> int:
        return len(self.value_labels)

@dataclass
class CommandConfig:
    """Cấu hình cho Command Processing - Cập nhật theo dataset thực tế"""
    command_labels: List[str] = None
    
    def __post_init__(self):
        if self.command_labels is None:
            self.command_labels = [
                "adjust-settings", "app-tutorial", "browse-social-media", "call", 
                "check-device-status", "check-health-status", "check-messages", 
                "check-weather", "control-device", "general-conversation", 
                "make-call", "make-video-call", "navigation-help", "open-app", 
                "open-app-action", "play-audio", "play-content", "play-media", 
                "provide-instructions", "read-content", "read-news", "search-content", 
                "search-internet", "send-mess", "send-message", "set-alarm", 
                "set-reminder", "view-content", "unknown"
            ]
    
    @property
    def num_commands(self) -> int:
        return len(self.command_labels)

@dataclass
class TrainingConfig:
    """Cấu hình cho training - Thay đổi các tham số dưới đây"""
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
    
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    save_best_only: bool = True 
    use_gradient_checkpointing: bool = False  
    use_fp16: bool = True
    use_amp: bool = True
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

print("🔧 Model Configuration:")
print(f"  Model: {model_config.model_name}")
print(f"  Size: {model_config.model_size}")
print(f"  Max Length: {model_config.max_length}")
print(f"  Batch Size: {model_config.batch_size}")
print(f"  Learning Rate: {model_config.learning_rate}")
print(f"  Epochs: {model_config.num_epochs}")
print(f"  Device: {model_config.device}")
print(f"  FP16: {model_config.use_fp16}")
print(f"  Gradient Checkpointing: {model_config.gradient_checkpointing}")
print(f"  Freeze Layers: {model_config.freeze_layers}")
print(f"  Optimizer: {model_config.optimizer}") 
