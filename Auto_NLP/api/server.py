#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated API Server sử dụng Model-First Hybrid System
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any
import uvicorn
from datetime import datetime
import time
import logging

# Add project root to path (api/ -> project root)
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import hybrid system
try:
    from core.hybrid_system import ModelFirstHybridSystem
    print("Imported ModelFirstHybridSystem")
except ImportError as e:
    print(f"Failed to import hybrid system: {e}")
    sys.exit(1)

# Import config
try:
    from config import config
    print("Imported config")
except ImportError as e:
    print(f"Failed to import config: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class IntentRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = None

class IntentResponse(BaseModel):
    input_text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]  # Changed to Any to handle lists
    command: str
    method: str
    decision_reason: Optional[str] = None
    primary_source: Optional[str] = None
    processing_time: float
    timestamp: str
    model_result: Optional[Dict[str, Any]] = None
    reasoning_result: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    hybrid_system_status: Dict[str, Any]
    system_stats: Dict[str, Any]

class StatsResponse(BaseModel):
    total_predictions: int
    model_predictions: int
    reasoning_predictions: int
    hybrid_predictions: int
    fallback_predictions: int
    avg_processing_time: float
    avg_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    success_rate: float

# Initialize FastAPI app
app = FastAPI(
    title="Auto NLP Hybrid System API",
    description="API cho hệ thống NLP Hybrid kết hợp trained model với reasoning engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hybrid system instance
hybrid_system: Optional[ModelFirstHybridSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize hybrid system on startup"""
    global hybrid_system
    
    try:
        logger.info("🚀 Initializing Hybrid System...")
        hybrid_system = ModelFirstHybridSystem()
        logger.info("✅ Hybrid System initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize hybrid system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global hybrid_system
    hybrid_system = None
    logger.info("🛑 Hybrid System shutdown")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Auto NLP Hybrid System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global hybrid_system
    
    try:
        # Get hybrid system status
        hybrid_status = {
            "model_loaded": hybrid_system.model_loaded if hybrid_system else False,
            "reasoning_loaded": hybrid_system.reasoning_loaded if hybrid_system else False,
            "device": str(hybrid_system.device) if hybrid_system else "unknown"
        }
        
        # Get system stats
        system_stats = hybrid_system.get_stats() if hybrid_system else {}
        
        return HealthResponse(
            status="healthy" if hybrid_system else "unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            hybrid_system_status=hybrid_status,
            system_stats=system_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict-simple")
async def predict_intent_simple(request: IntentRequest):
    """Predict intent using hybrid system - simplified response"""
    global hybrid_system
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        start_time = time.time()
        
        # Use hybrid system to predict
        result = hybrid_system.predict(request.text, request.context)
        
        processing_time = time.time() - start_time
        
        # Clean entities - convert empty arrays to empty strings for Pydantic
        entities = result.get("entities", {})
        cleaned_entities = {}
        for key, value in entities.items():
            if isinstance(value, list):
                if len(value) == 0:
                    cleaned_entities[key] = ""
                else:
                    cleaned_entities[key] = ", ".join(str(v) for v in value)
            else:
                cleaned_entities[key] = str(value) if value is not None else ""
        
        # Create simplified response
        simplified_response = {
            "input_text": request.text,
            "intent": result.get("intent", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "entities": cleaned_entities,
            "command": result.get("command", "unknown"),
            "method": result.get("method", "unknown"),
            "suggestions": result.get("reasoning_result", {}).get("reasoning_details", {}).get("suggestions", [])
        }
        
        return simplified_response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=IntentResponse)
async def predict_intent(request: IntentRequest):
    """Predict intent using hybrid system"""
    global hybrid_system
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        start_time = time.time()
        
        # Use hybrid system to predict
        result = hybrid_system.predict(request.text, request.context)
        
        processing_time = time.time() - start_time
        
        # Clean entities - convert empty arrays to empty strings for Pydantic
        entities = result.get("entities", {})
        cleaned_entities = {}
        for key, value in entities.items():
            if isinstance(value, list):
                if len(value) == 0:
                    cleaned_entities[key] = ""
                else:
                    cleaned_entities[key] = ", ".join(str(v) for v in value)
            else:
                cleaned_entities[key] = str(value) if value is not None else ""
        
        return IntentResponse(
            input_text=request.text,
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            entities=cleaned_entities,
            command=result.get("command", "unknown"),
            method=result.get("method", "unknown"),
            decision_reason=result.get("decision_reason"),
            primary_source=result.get("primary_source"),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            model_result=result.get("model_result"),
            reasoning_result=result.get("reasoning_result")
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    global hybrid_system
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        stats = hybrid_system.get_stats()
        
        return StatsResponse(
            total_predictions=stats.get("total_predictions", 0),
            model_predictions=stats.get("model_predictions", 0),
            reasoning_predictions=stats.get("reasoning_predictions", 0),
            hybrid_predictions=stats.get("hybrid_predictions", 0),
            fallback_predictions=stats.get("fallback_predictions", 0),
            avg_processing_time=stats.get("avg_processing_time", 0.0),
            avg_confidence=stats.get("avg_confidence"),
            min_confidence=stats.get("min_confidence"),
            max_confidence=stats.get("max_confidence"),
            success_rate=stats.get("success_rate", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/test")
async def test_system():
    """Test hybrid system with sample cases"""
    global hybrid_system
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        # Test cases
        test_cases = [
            "gọi điện cho mẹ",
            "bật đèn phòng khách",
            "phát nhạc",
            "tìm kiếm nhạc trên youtube",
            "đặt báo thức 7 giờ sáng"
        ]
        
        results = []
        
        for test_case in test_cases:
            result = hybrid_system.predict(test_case)
            results.append({
                "input": test_case,
                "intent": result.get("intent", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "method": result.get("method", "unknown"),
                "command": result.get("command", "unknown")
            })
        
        return {
            "message": "Test completed successfully",
            "test_cases": len(test_cases),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get system configuration"""
    try:
        return {
            "system_name": config.SYSTEM_NAME,
            "version": config.VERSION,
            "debug": config.DEBUG,
            "model_path": str(config.MODEL_PATH),
            "device": config.MODEL_DEVICE,
            "api_host": config.API_HOST,
            "api_port": config.API_PORT,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "fallback_enabled": config.FALLBACK_ENABLED
        }
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@app.post("/reload")
async def reload_system():
    """Reload hybrid system"""
    global hybrid_system
    
    try:
        logger.info("🔄 Reloading hybrid system...")
        
        # Reinitialize hybrid system
        hybrid_system = ModelFirstHybridSystem()
        
        logger.info("✅ Hybrid system reloaded successfully")
        
        return {
            "message": "Hybrid system reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reload system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload system: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    # Print startup info
    print("Starting Auto NLP Hybrid System API...")
    print(f"   Host: {config['api_host']}:{config['api_port']}")
    print(f"   Debug: {config['api_debug']}")
    print(f"   Model: {config['model_name']}")
    print(f"   Cache: {config['model_cache_dir']}")
    
    # Run server
    uvicorn.run(
        app,
        host=config['api_host'],
        port=config['api_port'],
        log_level="info" if not config['api_debug'] else "debug"
    )