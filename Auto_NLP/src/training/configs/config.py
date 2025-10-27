#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - OPTIMIZED for RTX 2060 6GB GPU with VITEXT dataset
Updated for normalized Vietnamese text and processed data
"""

import os

class ModelConfig:
    # Model settings - OPTIMIZED for Vietnamese text
    model_name = "vinai/phobert-large"
    model_size = "large"
    max_length = 128  # Reduced for Vietnamese text efficiency
    use_safetensors = True
    
    # Training settings - OPTIMIZED for balanced dataset (VITEXT normalized)
    num_epochs = int(os.environ.get('EPOCHS', 4))  # Reduced epochs to prevent overfitting
    batch_size = 16  # Optimal for RTX 2060 6GB
    learning_rate = 2e-05  # Standard LR for PhoBERT fine-tuning
    weight_decay = 0.01
    warmup_steps = 500  # Reduced for faster convergence
    
    # Loss weights - OPTIMIZED for multi-task learning
    LAMBDA_INTENT = 0.7  # Primary focus on intent
    LAMBDA_ENTITY = 0.2  # Increased entity weight for NER
    LAMBDA_COMMAND = 0.1  # Command classification
    LAMBDA_VALUE = 0.0   # Disabled value extraction
    
    # Label smoothing
    label_smoothing = 0.1
    
    # Freeze encoder - OPTIMIZED for PhoBERT-large fine-tuning
    freeze_encoder_epochs = 0  # No freezing for better learning
    freeze_layers = 0  # No layer freezing
    
    # Gradient clipping - OPTIMIZED for RTX 2060 6GB VRAM
    max_grad_norm = 1.0
    gradient_accumulation_steps = 2  # Effective batch size = 32
    
    # Mixed precision training - ENABLED for GPU
    use_mixed_precision = True  # Enable AMP for memory efficiency
    fp16 = True  # Enable FP16 training
    
    # Early stopping - OPTIMIZED for fewer epochs
    patience = 3  # Reduced patience for 4 epochs
    min_delta = 0.001
    
    # Optimizer - OPTIMIZED for GPU
    optimizer = "adamw"
    adam_epsilon = 1e-8
    adam_betas = (0.9, 0.999)
    clip_threshold = 1.0
    
    # Dropout - OPTIMIZED
    dropout = 0.1
    
    # Data paths - UPDATED for processed VITEXT data
    dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    train_data_path = "src/data/processed/train.json"
    val_data_path = "src/data/processed/val.json"
    test_data_path = "src/data/processed/test.json"
    
    # Output paths
    output_dir = "models/phobert_large_intent_model"
    log_dir = "logs/vitext_training"
    
    # Metrics
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'confusion_matrix']
    
    # GPU-specific settings
    device = "cuda"
    use_cuda = True
    cuda_visible_devices = "0"  # Use first GPU
    
    # Memory optimization
    gradient_checkpointing = False  # Disabled for stability
    use_gradient_accumulation = True  # Simulate larger batch
    
    # Training stability
    seed = 42
    deterministic = True

class IntentConfig:
    """Intent classification configuration - UPDATED for VITEXT dataset"""
    num_intents = 13
    intent_labels = [
        "add-contacts", "call", "control-device", "get-info", "make-video-call", 
        "open-cam", "play-media", "search-internet", "search-youtube", 
        "send-mess", "set-alarm", "set-event-calendar", "view-content"
    ]

class EntityConfig:
    """Entity recognition configuration - UPDATED for VITEXT dataset"""
    num_entity_labels = 30  # Updated to match actual dataset
    entity_labels = [
        "O", "B-ACTION", "B-CONTACT_NAME", "B-CONTENT_TYPE", "B-DATE", "B-DEVICE", 
        "B-LEVEL", "B-LOCATION", "B-MEDIA_TYPE", "B-MESSAGE", "B-PHONE", "B-PLATFORM", 
        "B-QUERY", "B-RECEIVER", "B-TIME", "B-TITLE", "I-CONTACT_NAME", "I-CONTENT_TYPE", 
        "I-DATE", "I-DEVICE", "I-LEVEL", "I-LOCATION", "I-MEDIA_TYPE", "I-MESSAGE", 
        "I-PHONE", "I-PLATFORM", "I-QUERY", "I-RECEIVER", "I-TIME", "I-TITLE"
    ]

class ValueConfig:
    """Value extraction configuration - UPDATED for VITEXT dataset"""
    num_value_labels = 33  # Updated based on actual value labels
    value_labels = [
        "O", "B-action", "I-action", "B-contact_name", "I-contact_name", 
        "B-content_type", "I-content_type", "B-date", "I-date", "B-device", 
        "I-device", "B-level", "I-level", "B-location", "I-location", 
        "B-media_type", "I-media_type", "B-phone", "I-phone", "B-platform", 
        "I-platform", "B-query", "I-query", "B-time", "I-time", "B-title", 
        "I-title", "B-value", "I-value", "B-number", "I-number", "B-percentage", "I-percentage"
    ]

class CommandConfig:
    """Command classification configuration - UPDATED for VITEXT dataset"""
    num_command_labels = 13
    command_labels = [
        "add-contacts", "call", "control-device", "get-info", "make-video-call", 
        "open-cam", "play-media", "search-internet", "search-youtube", 
        "send-mess", "set-alarm", "set-event-calendar", "view-content"
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