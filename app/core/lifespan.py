from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import logger, ml_resources

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Simplified lifespan without ONNX model.
    We now rely on HuggingFace model inside new_model.py
    """

    try:
        logger.info("🚀 Starting app without ONNX model (using HuggingFace)...")
        
        # مفيش تحميل لأي موديل هنا
        yield

    except Exception as e:
        logger.error(f"Startup error: {e}")
        yield

    finally:
        logger.info("🛑 Shutting down app...")
        ml_resources.clear()
