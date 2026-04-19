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
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        # 🔥 build conversation
        conversation = ""
        for msg in history:
            if msg["role"] == "ai":
                conversation += f"Assistant: {msg['text']}\n"
            else:
                conversation += f"User: {msg['text']}\n"

        # 🔥 prompt قوي جدًا
        full_prompt = f"""
You are a smart emotional support assistant.

STRICT RULES (must follow):
- NEVER say: "I'm here for you", "Tell me more", "I understand"
- NEVER give generic responses
- ALWAYS give specific, practical advice
- ALWAYS refer to the conversation context
- Respond like a real human, not a chatbot

Conversation:
{conversation}

User message:
{text}

Now give a helpful, specific response:
"""

        # 🔥 generation config (مهم جدًا)
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 200,
            }
        )

        reply = response.text.strip()

        # 🔥 لو طلع رد generic → نمنعه
        banned_phrases = [
            "tell me more",
            "i'm here for you",
            "i understand"
        ]

        if any(p in reply.lower() for p in banned_phrases):
            return "It sounds like work is really draining you. Try identifying the main source of stress and take small breaks during the day to reset your energy."

        # تنظيف الرد
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        # fallback لو اللغة غلط
        if detect_language(reply) != lang:
            return get_fallback(lang, emotion)

        return reply

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return get_fallback(lang, emotion)