import re
from typing import List
import google.generativeai as genai

from app.core.config import GEMINI_API_KEY, logger
from app.utils.heuristics import get_fallback
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Message]
) -> str:

    if not GEMINI_API_KEY:
        return get_fallback(lang, emotion)

    try:
        # ❌ شيلنا system_instruction خالص
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        # 🔥 نبني conversation قوية
        conversation = ""

        for msg in history:
            if msg["role"] == "ai":
                conversation += f"Assistant: {msg['text']}\n"
            else:
                conversation += f"User: {msg['text']}\n"

        # 🔥 Prompt ذكي يخليه يرد صح
        full_prompt = f"""
You are a helpful emotional support assistant.

Conversation so far:
{conversation}

User just said:
{text}

Your job:
- Continue the conversation naturally
- Give helpful advice if user asks
- Be specific (not generic)
- Do NOT repeat phrases like "tell me more"
- Respond like a real human

Answer:
"""

        # 🔥 إرسال
        response = model.generate_content(full_prompt)

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