print("🔥 NEW GEMINI VERSION WORKING")

import re
from typing import List
import google.generativeai as genai

from app.core.config import GEMINI_API_KEY, logger
from app.utils.heuristics import get_fallback
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language

# 🔑 init Gemini
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

        # 🔥 build conversation correctly (IMPORTANT FIX)
        conversation = ""
        recent_history = history[-6:] if history else []

        for msg in recent_history:
            role = getattr(msg, "role", "")
            content = getattr(msg, "text", "")

            if role in ["ai", "assistant", "bot", "model"]:
                conversation += f"Assistant: {content}\n"
            else:
                conversation += f"User: {content}\n"

        # 🔥 prompt قوي جدًا
        full_prompt = f"""
You are a smart emotional support assistant.

STRICT RULES:
- NEVER say "I'm here for you"
- NEVER say "Tell me more"
- NEVER give generic answers
- ALWAYS use conversation context
- ALWAYS give practical advice
- Be natural and human-like

Conversation:
{conversation}

User latest message:
{text}

Give a helpful, specific response:
"""

        # 🔥 generate
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.85,
                "top_p": 0.95,
                "max_output_tokens": 200,
            }
        )

        reply = response.text.strip()

        # 🔥 تنظيف الرد
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        # 🔥 منع الردود الغبية (CRITICAL)
        banned_phrases = [
            "tell me more",
            "i'm here for you",
            "i understand",
            "i’m here for you"
        ]

        if any(p in reply.lower() for p in banned_phrases):
            if "work" in text.lower():
                return "Work seems to be draining you a lot. Try prioritizing your tasks and taking short breaks — even 10 minutes can help you reset."
            else:
                return "It sounds like you're overwhelmed. Try focusing on one small step you can control right now — it can make things feel more manageable."

        # 🔥 لو الرد ضعيف → نحسنه
        if len(reply.split()) < 6:
            return "Try breaking things into smaller steps and focus on one thing at a time — it really helps reduce pressure."

        # 🔥 لغة
        if detect_language(reply) != lang:
            return get_fallback(lang, emotion)

        return reply

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return get_fallback(lang, emotion)