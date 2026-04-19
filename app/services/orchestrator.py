from app.core.config import logger
from app.services.gemini_service import generate_gemini_response
from app.services.memory_service import get_last_messages, save_message
from app.utils.heuristics import get_fallback

from app.ml.new_model import NextGenModel

model = NextGenModel()
model.load_model()


def classify_emotion(text: str) -> dict:
    try:
        result = model.predict(text)

        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "language": "en",
        }

    except Exception as e:
        logger.error("Classification error: %s", e)

        return {
            "emotion": "error",
            "confidence": 0.0,
            "language": "en",
        }


def generate_chat_response(
    text: str,
    emotion: str,
    user_id: str = "default",
) -> dict:

    uid = (user_id or "default").strip() or "default"

    try:
        # 🔥 1. هات التاريخ الأول (الأهم)
        history = get_last_messages(uid, limit=6)

        # 🔥 2. خزّن رسالة المستخدم
        save_message(uid, "user", text)

        # 🔥 3. ابعت التاريخ + الرسالة الجديدة
        reply = generate_gemini_response(
            text,
            emotion,
            "en",
            history,
        )

        # 🔥 4. خزّن رد AI
        save_message(uid, "ai", reply)

        return {
            "message": reply,
            "emotion": emotion,
            "ai": "gemini",
        }

    except Exception as e:
        logger.error("Chat generation error: %s", e)

        return {
            "message": get_fallback("en", emotion),
            "emotion": emotion,
            "ai": "fallback",
        }