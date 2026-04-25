import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# root path
_ROOT = Path(__file__).resolve().parents[2]

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# load .env
load_dotenv()

# =========================
# 🔑 API KEYS
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("CRITICAL: GROQ_API_KEY is missing!")

# ❌ حذفنا الاعتماد الإجباري على Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # اختياري بس

# =========================
# 📁 Paths
# =========================

CHAT_MEMORY_DB = os.getenv(
    "CHAT_MEMORY_DB",
    str(_ROOT / "data" / "chat_memory.db")
)

# =========================
# ⚙️ Settings
# =========================

MAX_LEN = 50
LOW_CONFIDENCE_THRESHOLD = 0.45

# shared resources
ml_resources = {}