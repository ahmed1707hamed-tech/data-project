from typing import List
from app.core.config import logger
from app.schemas.chat_schema import Message
from app.services.gemini_service import generate_gemini_response
from app.utils.heuristics import get_fallback

from app.ml.new_model import NextGenModel

# 🔥 تحميل الموديل مرة واحدة
model = NextGenModel()
model.load_model()

# 🔥 Memory لكل users
user_memory = {}


def classify_emotion(text: str) -> dict:
    try:
        print("🔥 classify_emotion called")

        result = model.predict(text)

        print("🔥 MODEL RESULT:", result)

        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "language": "en",
        }

    except Exception as e:
        import traceback
        print("🔥 CLASSIFICATION ERROR:")
        traceback.print_exc()

        return {
            "emotion": "error",
            "confidence": 0.0,
            "language": "en",
        }


def generate_chat_response(
    text: str,
    emotion: str,
    user_id: str = "default"
) -> dict:

    try:
        # 🔥 لو user جديد
        if user_id not in user_memory:
            user_memory[user_id] = []

        memory = user_memory[user_id]

        # 🔥 أضف رسالة اليوزر
        memory.append({
            "role": "user",
            "text": text
        })

        # 🔥 ناخد آخر 10 رسائل بس
        recent_memory = memory[-10:]

        # 🔥 نكلم Gemini بالـ memory
        gemini_msg = generate_gemini_response(
            text,
            emotion,
            "en",
            recent_memory
        )

        # 🔥 أضف رد البوت
        memory.append({
            "role": "ai",
            "text": gemini_msg
        })

        return {
            "message": gemini_msg,
            "emotion": emotion,
            "ai": "gemini",
        }

    except Exception as e:
        logger.error(f"Chat generation error: {e}")

        return {
            "message": get_fallback("en", emotion),
            "emotion": emotion,
            "ai": "fallback",
        }