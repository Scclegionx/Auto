"""
Models package cho PhoBERT_SAM
"""

from .intent_model import IntentRecognitionModel
from .entity_model import EntityExtractionModel
from .command_model import CommandProcessingModel
from .unified_model import UnifiedModel, JointModel

def create_model(model_type: str = "unified"):
    """Tạo mô hình theo loại"""
    if model_type == "intent":
        return IntentRecognitionModel()
    elif model_type == "entity":
        return EntityExtractionModel()
    elif model_type == "command":
        return CommandProcessingModel()
    elif model_type == "joint":
        return JointModel()
    elif model_type == "unified":
        return UnifiedModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

__all__ = [
    "IntentRecognitionModel",
    "EntityExtractionModel", 
    "CommandProcessingModel",
    "UnifiedModel",
    "JointModel",
    "create_model"
] 