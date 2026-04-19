from app.core.config import logger
from app.services.groq_service import generate_groq_response
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
        # ✅ خزّن رسالة المستخدم
        save_message(uid, "user", text)

        # ✅ هات history بعد التحديث
        history = get_last_messages(uid, limit=6)

        # ✅ نمنع التكرار (مهم)
        if len(history) > 1 and history[-1].text == history[-2].text:
            history = history[:-1]

        # ✅ Groq response
        reply = generate_groq_response(
            text,
            emotion,
            "en",
            history,
        )

        # fallback
        if not reply:
            reply = get_fallback("en", emotion)
            ai_type = "fallback"
        else:
            ai_type = "gemini"

        # خزّن الرد
        save_message(uid, "ai", reply)

        return {
            "message": reply,
            "emotion": emotion,
            "ai": ai_type,
        }

    except Exception as e:
        logger.error("Chat generation error: %s", e)

        return {
            "message": get_fallback("en", emotion),
            "emotion": emotion,
            "ai": "fallback",
        }