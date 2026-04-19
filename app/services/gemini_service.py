print("🔥 GEMINI FINAL VERSION RUNNING")

import re
from typing import List, Union
import google.generativeai as genai

from app.core.config import GEMINI_API_KEY, logger
from app.utils.heuristics import get_fallback
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language

# 🔑 Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
) -> str:

    if not GEMINI_API_KEY:
        return get_fallback(lang, emotion)

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        # 🔥 Build conversation (supports dict + object)
        conversation = ""
        recent_history = history[-6:] if history else []

        for msg in recent_history:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("text", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "text", "")

            if role.lower() in ["ai", "assistant", "bot", "model"]:
                conversation += f"Assistant: {content}\n"
            else:
                conversation += f"User: {content}\n"

        # 🔥 Strong prompt
        prompt = f"""
You are a smart emotional support assistant.

STRICT RULES:
- NEVER say "I'm here for you"
- NEVER say "Tell me more"
- NEVER give generic responses
- ALWAYS give practical advice
- ALWAYS use the conversation context
- Be natural and human-like

Conversation:
{conversation}

User message:
{text}

Give a helpful and specific response:
"""

        # 🔥 Generate
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "max_output_tokens": 200
            }
        )

        reply = response.text.strip()

        # 🔥 Clean reply
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        # 💣 Block bad responses
        banned = [
            "tell me more",
            "i'm here for you",
            "i understand",
            "i’m here for you"
        ]

        if any(b in reply.lower() for b in banned):

            # 🔥 Smart fallback based on context
            if "work" in text.lower():
                return "Work stress can really build up. Try prioritizing your tasks and taking short breaks during the day to recharge."

            if "tired" in text.lower():
                return "Feeling drained is a sign you need rest. Try stepping away for a bit and giving yourself a mental break."

            return "Try focusing on one small step you can take right now. Small progress can make things feel more manageable."

        # 🔥 Weak response fix
        if len(reply.split()) < 6:
            return "Try breaking things into smaller steps and focus on one task at a time — it helps reduce stress."

        # 🔥 Language check
        if detect_language(reply) != lang:
            return get_fallback(lang, emotion)

        return reply

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return get_fallback(lang, emotion)