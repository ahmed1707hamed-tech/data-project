import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("CRITICAL: GEMINI_API_KEY is missing from environment variables!")

CHAT_MEMORY_DB = os.getenv("CHAT_MEMORY_DB", str(_ROOT / "data" / "chat_memory.db"))

# Global Settings
MAX_LEN = 50
LOW_CONFIDENCE_THRESHOLD = 0.45

# Shared resource dictionary
ml_resources = {}
