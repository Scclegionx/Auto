import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Cấu hình cho mô hình PhoBERT"""
    model_name: str = "vinai/phobert-base"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1
    
@dataclass
class IntentConfig:
    """Cấu hình cho Intent Recognition"""
    num_intents: int = 17  # Tất cả các loại command trong dataset
    intent_labels: List[str] = None
    
    def __post_init__(self):
        if self.intent_labels is None:
            self.intent_labels = [
                "call", "check-health-status", "check-weather", "express-emotion",
                "express-fatigue", "find-information", "general-conversation", 
                "general-request", "play-media", "read-news", "report-symptom",
                "request-comfort", "request-entertainment", "request-instruction",
                "send-mess", "set-alarm", "set-reminder"
            ]

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
    """Cấu hình cho training"""
    output_dir: str = "./models"
    data_dir: str = "./data"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "cuda" if os.path.exists("/dev/cuda") else "cpu"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
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