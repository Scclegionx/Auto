import os
from dataclasses import dataclass
from typing import List

class ModelConfig:
    """Cấu hình Model - Thay đổi các tham số dưới đây để tối ưu"""
    
    def __init__(self):
        self.model_name = "vinai/phobert-large"
        self.model_size = "large"
        
        # Tham số cơ bản
        self.max_length = 512  # 192 (tiết kiệm) hoặc 512 (tối ưu)
        self.batch_size = 32     # 8 (tiết kiệm) hoặc 32 (tối ưu)
        self.num_epochs = 20     # 5 (tiết kiệm) hoặc 20 (tối ưu)
        self.learning_rate = 3e-5  # 1e-5 (tiết kiệm) hoặc 3e-5 (tối ưu)
        self.freeze_layers = 4  # 8 (tiết kiệm) hoặc 4 (tối ưu)
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True  # False (tiết kiệm) hoặc True (tối ưu)
        self.use_amp = True
        self.gradient_checkpointing = False  # True (tiết kiệm) hoặc False (tối ưu)
        self.use_mixed_precision = True
        
        self.gradient_accumulation_steps = 1  # 4 (tiết kiệm) hoặc 1 (tối ưu)
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 1000  # 300 (tiết kiệm) hoặc 1000 (tối ưu)
        
        self.num_workers = 8  # 4 (tiết kiệm) hoặc 8 (tối ưu)
        self.pin_memory = True
        self.dropout = 0.1  # 0.1 (tiết kiệm) hoặc 0.1 (tối ưu)
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
    hidden_layer_sizes: List[int] = None  # Sẽ được set động dựa trên model size
    
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
    save_steps: int = 25      # 100 (tiết kiệm) hoặc 25 (tối ưu)
    eval_steps: int = 25      # 100 (tiết kiệm) hoặc 25 (tối ưu)
    logging_steps: int = 5    # 20 (tiết kiệm) hoặc 5 (tối ưu)
    early_stopping_patience: int = 10  # 15 (tiết kiệm) hoặc 10 (tối ưu)
    save_total_limit: int = 5  # 3 (tiết kiệm) hoặc 5 (tối ưu)
    
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    save_best_only: bool = True  # False (tiết kiệm) hoặc True (tối ưu)
    use_gradient_checkpointing: bool = False  # True (tiết kiệm) hoặc False (tối ưu)
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
