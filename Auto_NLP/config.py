#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Auto NLP Hybrid System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "src" / "data"
CACHE_DIR = PROJECT_ROOT / "model_cache"

# Model configuration
MODEL_NAME = "vinai/phobert-large"
MODEL_CACHE_DIR = str(CACHE_DIR)

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True

# Hybrid system configuration
CONFIDENCE_THRESHOLD = 0.7
REASONING_ENABLED = True
SPECIALIZED_ENTITY_EXTRACTION = True

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "hybrid_system.log"

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration object
config = {
    "project_root": str(PROJECT_ROOT),
    "model_dir": str(MODEL_DIR),
    "data_dir": str(DATA_DIR),
    "cache_dir": str(CACHE_DIR),
    "model_name": MODEL_NAME,
    "model_cache_dir": MODEL_CACHE_DIR,
    "api_host": API_HOST,
    "api_port": API_PORT,
    "api_debug": API_DEBUG,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "reasoning_enabled": REASONING_ENABLED,
    "specialized_entity_extraction": SPECIALIZED_ENTITY_EXTRACTION,
    "log_level": LOG_LEVEL,
    "log_file": LOG_FILE
}
