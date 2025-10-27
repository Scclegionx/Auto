#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Configuration
Cấu hình chính của hệ thống
"""

import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # System
    SYSTEM_NAME = "Auto_NLP_Hybrid_System"
    VERSION = "1.0.0"
    DEBUG = True
    
    # Paths
    BASE_DIR = Path(__file__).parent
    CORE_DIR = BASE_DIR / "core"
    API_DIR = BASE_DIR / "api"
    DATA_DIR = BASE_DIR / "src" / "data"
    MODELS_DIR = BASE_DIR / "models"
    TRAINING_DIR = BASE_DIR / "src" / "training"
    
    # Model
    MODEL_PATH = MODELS_DIR / "phobert_large_intent_model"
    MODEL_DEVICE = "auto"  # "cuda", "cpu", or "auto"
    MAX_LENGTH = 128
    
    # Hybrid System
    PRIMARY_MODEL = "trained_model"
    SECONDARY_ENGINE = "reasoning_engine"
    CONFIDENCE_THRESHOLD = 0.7
    FALLBACK_ENABLED = True
    
    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 1
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Data
    MAIN_DATASET = DATA_DIR / "raw" / "elderly_command_dataset_MERGED_13C_VITEXT.json"
    TRAIN_DATA = DATA_DIR / "processed" / "train.json"
    VAL_DATA = DATA_DIR / "processed" / "val.json"
    TEST_DATA = DATA_DIR / "processed" / "test.json"
    
    @classmethod
    def get_model_path(cls):
        """Get model path"""
        return str(cls.MODEL_PATH)
    
    @classmethod
    def get_data_path(cls, data_type="main"):
        """Get data path"""
        paths = {
            "main": cls.MAIN_DATASET,
            "train": cls.TRAIN_DATA,
            "val": cls.VAL_DATA,
            "test": cls.TEST_DATA
        }
        return str(paths.get(data_type, cls.MAIN_DATASET))
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print(f"🔧 {cls.SYSTEM_NAME} v{cls.VERSION}")
        print(f"   Debug: {cls.DEBUG}")
        print(f"   Model Path: {cls.MODEL_PATH}")
        print(f"   Device: {cls.MODEL_DEVICE}")
        print(f"   API: {cls.API_HOST}:{cls.API_PORT}")
        print(f"   Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")

# Global config instance
config = Config()
