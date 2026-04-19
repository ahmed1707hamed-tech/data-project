import re
from typing import List
import google.generativeai as genai

from app.core.config import GEMINI_API_KEY, logger
from app.utils.heuristics import get_fallback
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _classify_intent(text: str, lang: str) -> str:
    t = text.lower()

    q_en = ["?", "how", "what", "why", "when", "where", "can you", "help", "advice"]
    q_ar = ["؟", "كيف", "ماذا", "لماذا", "متى", "أين", "هل", "مساعدة", "نصيحة", "ساعدني"]

    v_en = ["hate", "give up", "tired of", "sick of", "always", "never"]
    v_ar = ["تعبت", "بكره", "مش طايق", "دايما", "ابدا"]

    if any(m in t for m in (q_ar if lang == "ar" else q_en)):
        return "question"
    if any(m in t for m in (v_ar if lang == "ar" else v_en)):
        return "venting"

    return "statement"


def build_system_prompt(detected_emotion: str, language: str, intent: str) -> str:
    return f"""
You are a natural human-like conversational assistant.

- Respond ONLY in {language}
- Be relevant to the user's last message
- Continue the conversation naturally
- Do NOT repeat yourself
"""


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Message]
) -> str:

    if not GEMINI_API_KEY:
        return get_fallback(lang, emotion)

    intent = _classify_intent(text, lang)
    language_full = "Arabic" if lang == "ar" else "English"
    system_instruction = build_system_prompt(emotion, language_full, intent)

    try:
        model = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            system_instruction=system_instruction
        )

        # 🔥 هنا السحر الحقيقي (تحويل history لنص واحد)
        conversation = ""

        for msg in history:
            if msg.role.lower() in ["ai", "assistant", "bot", "model"]:
                conversation += f"Assistant: {msg.text}\n"
            else:
                conversation += f"User: {msg.text}\n"

        # 🔥 آخر رسالة
        conversation += f"User: {text}\nAssistant:"

        # 🔥 إرسال كل الحوار مرة واحدة
        response = model.generate_content(conversation)

        reply = response.text.strip()

        # تنظيف الرد
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        # fallback لو اللغة غلط
        if detect_language(reply) != lang:
            return get_fallback(lang, emotion)

        return reply

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return get_fallback(lang, emotion)
