import pickle
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import logger, ml_resources

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI app.
    Loads models and tokenizers into the global ml_resources dictionary.
    """
    try:
        logger.info("Starting up: Loading ML models and tokenizers...")
        
        with open("weights/emotion_bilstm/tokenizer.pkl", "rb") as f:
            ml_resources["tokenizer"] = pickle.load(f)

        with open("weights/emotion_bilstm/labels.pkl", "rb") as f:
            ml_resources["labels"] = pickle.load(f)

        session = ort.InferenceSession("weights/emotion_bilstm/model.onnx")
        ml_resources["session"] = session
        ml_resources["input_name"] = session.get_inputs()[0].name
        
        logger.info("All ML resources successfully loaded.")
        yield
    except Exception as e:
        logger.error(f"Failed to load ML resources: {e}")
        yield
    finally:
        logger.info("Shutting down: Clearing ML resources.")
        ml_resources.clear()
