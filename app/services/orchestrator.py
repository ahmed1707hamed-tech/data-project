from app.core.config import logger
from app.services.gemini_service import generate_gemini_response
from app.services.memory_service import get_last_messages, save_message
from app.utils.heuristics import get_fallback

from app.ml.new_model import NextGenModel

model = NextGenModel()
model.load_model()


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
    user_id: str = "default",
) -> dict:
    uid = (user_id or "default").strip() or "default"

    try:
        save_message(uid, "user", text)
        recent = get_last_messages(uid, limit=6)

        gemini_msg = generate_gemini_response(
            text,
            emotion,
            "en",
            recent,
        )

        save_message(uid, "ai", gemini_msg)

        return {
            "message": gemini_msg,
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
