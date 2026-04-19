from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import logger, ml_resources
from app.services.memory_service import init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting app (HF model + SQLite memory)...")
        init_database()
        yield

    except Exception as e:
        logger.error("Startup error: %s", e)
        yield

    finally:
        logger.info("Shutting down app...")
        ml_resources.clear()
