from groq import Groq
import re
from typing import List, Union

from app.core.config import logger
from app.schemas.chat_schema import Message
from app.utils.heuristics import get_fallback
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_groq_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
) -> str:

    try:
        messages = []

        # 🔥 history
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("text")
            else:
                role = msg.role
                content = msg.text

            if role == "ai":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})

        # 🔥 current message
        messages.append({"role": "user", "content": text})

        # 🔥 request
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()

        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply).strip()

        return reply

    except Exception as e:
        logger.error(f"Groq error: {e}")
        return None