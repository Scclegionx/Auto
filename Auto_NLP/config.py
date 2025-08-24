import os
from dataclasses import dataclass
from typing import List

class ModelConfig:
    """Cấu hình tối ưu cho Model với cải tiến cho PhoBERT-Large"""
    
    def __init__(self):
        # Cài đặt cơ bản
        self.model_name = "vinai/phobert-large"
        self.model_size = "large"  # "base" hoặc "large"
        self.max_length = 256  # Giảm từ 512 để tiết kiệm bộ nhớ
        
        # GPU settings
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True  # Sử dụng mixed precision
        self.use_amp = True  # Automatic Mixed Precision
        self.gradient_checkpointing = True  # Tiết kiệm memory cho large model
        self.use_mixed_precision = True  # Sử dụng mixed precision để tiết kiệm memory
        
        # Optimizer settings
        self.batch_size = 16  # Giảm batch size cho large model
        self.gradient_accumulation_steps = 2  # Tăng lên 2-4 để mô phỏng batch size lớn hơn
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_steps = 500
        
        # Training settings
        self.num_epochs = 10
        self.num_workers = 4
        self.pin_memory = True
        
        # Model architecture
        self.dropout = 0.2
        self.freeze_layers = 6  # Đóng băng 6 layer đầu tiên
        
        # Optimizer type
        self.optimizer = "adamw"  # "adamw" hoặc "adafactor"
        
        # Điều chỉnh cấu hình dựa trên kích thước model
        if self.model_size == "large":
            self.batch_size = 8  # Giảm batch size cho large model
            self.gradient_accumulation_steps = 4
            self.max_length = 192  # Tiết kiệm bộ nhớ hơn
            self.learning_rate = 1e-5  # Giảm learning rate
            self.num_epochs = 15  # Tăng epochs để tận dụng large model
            self.warmup_steps = 300  # Tăng warmup steps cho large model
            self.dropout = 0.1  # Giảm dropout để tận dụng capacity của large model
            self.freeze_layers = 8  # Đóng băng nhiều layer hơn cho large model
            self.use_fp16 = False  # Tắt FP16 để tránh lỗi gradients
    
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
        
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ Auto-detected device: {self.device}")
            
            if self.device == "cuda":
                print(f"🎮 GPU: {torch.cuda.get_device_name()}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def __post_init__(self):
        """Validate model size và setup device"""
        if self.model_size not in ["base", "large"]:
            raise ValueError("model_size phải là 'base' hoặc 'large'")
        
        # Auto-detect device
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
    num_intents: int = 28  # Cập nhật theo dataset thực tế: 27 commands
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
            # Cập nhật theo dataset thực tế: 27 commands
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
        
        # Set hidden layer sizes dựa trên model size
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
            # Cập nhật theo dataset mới: 6 entity labels chính
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
            # Cập nhật theo dataset mới: 14 value labels chính
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
            # Cập nhật theo dataset thực tế: 28 commands
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
    """Cấu hình cho training - Tối ưu cho GPU và Large model"""
    output_dir: str = "./models"
    data_dir: str = "./data"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "auto"  # Auto-detect GPU/CPU
    save_steps: int = 50  # Tăng save frequency cho large model
    eval_steps: int = 50  # Tăng eval frequency cho large model
    logging_steps: int = 10  # Tăng logging frequency để theo dõi large model
    
    # GPU optimization settings
    use_mixed_precision: bool = True  # Bật mixed precision cho GPU
    max_grad_norm: float = 1.0  # Gradient clipping
    early_stopping_patience: int = 15  # Tăng patience cho large model
    save_best_only: bool = False  # Lưu tất cả checkpoints để có thể resume
    save_total_limit: int = 10  # Giữ 10 checkpoints gần nhất
    
    # Large model specific settings
    use_gradient_checkpointing: bool = True  # Tiết kiệm memory
    use_fp16: bool = True  # Sử dụng FP16
    use_amp: bool = True  # Automatic Mixed Precision
    max_memory_usage: str = "auto"  # Auto-detect memory usage
    
    def __post_init__(self):
        # Tạo các thư mục cần thiết
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

# In thông tin cấu hình
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