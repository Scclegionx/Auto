import os
import sys
from pathlib import Path

import torch

_this_file = Path(__file__).resolve()
_src_root = _this_file.parents[2]  # .../src
_src_root_str = str(_src_root)
if _src_root_str not in sys.path:
    sys.path.insert(0, _src_root_str)

from data.entity_schema import ENTITY_BASE_NAMES, generate_entity_labels


class ModelConfig:

    # Model settings - tự điều chỉnh theo VRAM
    model_name = os.environ.get("MODEL_NAME", "vinai/phobert-large")
    model_size = "base" if "base" in model_name else "large"
    max_length = int(os.environ.get("MAX_LENGTH", 128))
    use_safetensors = True

    # Training settings
    num_epochs = int(os.environ.get("EPOCHS", 4))
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    learning_rate = float(os.environ.get("LEARNING_RATE", 2e-5))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 300))

    # Loss weights
    LAMBDA_INTENT = float(os.environ.get("LAMBDA_INTENT", 0.45))
    LAMBDA_ENTITY = float(os.environ.get("LAMBDA_ENTITY", 0.25))
    LAMBDA_COMMAND = float(os.environ.get("LAMBDA_COMMAND", 0.2))
    LAMBDA_ENTITY_INITIAL = float(os.environ.get("LAMBDA_ENTITY_INITIAL", 0.05))
    LAMBDA_ENTITY_TARGET = float(os.environ.get("LAMBDA_ENTITY_TARGET", 0.25))
    LAMBDA_ENTITY_WARMUP_EPOCHS = int(os.environ.get("LAMBDA_ENTITY_WARMUP_EPOCHS", 3))

    # Label smoothing
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))

    # Encoder freezing
    freeze_encoder_epochs = int(os.environ.get("FREEZE_ENCODER_EPOCHS", 0))
    freeze_layers = int(os.environ.get("FREEZE_LAYERS", 0))

    # Gradient handling
    max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", 1.0))
    gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", 2))

    # Mixed precision
    use_mixed_precision = os.environ.get("USE_MIXED_PRECISION", "1") != "0"

    # Early stopping
    patience = int(os.environ.get("PATIENCE", 3))
    min_delta = float(os.environ.get("MIN_DELTA", 1e-3))

    # Optimizer
    optimizer = os.environ.get("OPTIMIZER", "adamw")
    adam_epsilon = float(os.environ.get("ADAM_EPSILON", 1e-8))
    adam_betas = (0.9, 0.999)
    clip_threshold = float(os.environ.get("CLIP_THRESHOLD", 1.0))
    entity_head_lr_factor = float(os.environ.get("ENTITY_HEAD_LR_FACTOR", 0.5))

    # Dropout
    dropout = float(os.environ.get("DROPOUT", 0.1))
    entity_dropout = float(os.environ.get("ENTITY_DROPOUT", 0.2))

    # Data paths - có thể override qua biến môi trường
    dataset_path = os.environ.get("DATASET_PATH", "src/data/raw/elderly_commands_master.json")
    train_data_path = os.environ.get("TRAIN_DATA_PATH", "src/data/processed/train.json")
    val_data_path = os.environ.get("VAL_DATA_PATH", "src/data/processed/val.json")
    test_data_path = os.environ.get("TEST_DATA_PATH", "src/data/processed/test.json")

    # Output paths
    output_dir = os.environ.get("OUTPUT_DIR", "models/phobert_multitask")
    log_dir = os.environ.get("LOG_DIR", "logs/multitask_training")

    # Metrics
    metrics = ["accuracy", "f1", "precision", "recall"]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Training stability
    seed = int(os.environ.get("SEED", 42))
    deterministic = os.environ.get("DETERMINISTIC", "1") != "0"

class IntentConfig:
    """Intent classification configuration - UPDATED cho bộ lệnh rút gọn"""
    num_intents = 10
    intent_labels = [
        "add-contacts", "call", "control-device", "get-info", "make-video-call",
        "open-cam", "search-internet", "search-youtube", "send-mess", "set-alarm"
    ]

class EntityConfig:
    """Entity recognition configuration - chuẩn hóa theo schema cuối."""
    entity_base_names = ENTITY_BASE_NAMES
    entity_labels = generate_entity_labels()
    num_entity_labels = len(entity_labels)

class CommandConfig:
    """Command classification configuration - UPDATED cho bộ lệnh rút gọn"""
    num_command_labels = 10
    command_labels = [
        "add-contacts", "call", "control-device", "get-info", "make-video-call",
        "open-cam", "search-internet", "search-youtube", "send-mess", "set-alarm"
    ]

class TrainingConfig:
    """Training configuration - OPTIMIZED for VITEXT dataset"""
    device = "cuda"
    seed = 42
    deterministic = True
    
    # GPU memory management
    empty_cache_frequency = 20  # Clear GPU cache every 20 batches
    max_memory_usage = 0.85  # Use max 85% of GPU memory for stability
    
    # Training monitoring
    log_interval = 100  # Log every 100 batches
    save_interval = 1  # Save every epoch
    eval_interval = 1  # Evaluate every epoch
    
    # Validation settings
    val_batch_size = 32  # Larger batch for validation
    val_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    
    # Model saving
    save_best_model = True
    save_last_model = True
    model_save_format = "pytorch"  # Save as .pth files