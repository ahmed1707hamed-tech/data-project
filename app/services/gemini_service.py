from google import genai
import re
from typing import List, Union

from app.core.config import GEMINI_API_KEY, logger
from app.schemas.chat_schema import Message
from app.utils.heuristics import get_fallback


# 🔥 init client
client = genai.Client(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
) -> str:

    try:
        # 🔥 build conversation
        conversation = ""

        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("text")
            else:
                role = msg.role
                content = msg.text

            if role == "ai":
                conversation += f"Assistant: {content}\n"
            else:
                conversation += f"User: {content}\n"

        conversation += f"User: {text}\nAssistant:"

        print("🔥 PROMPT:\n", conversation)

        # 🔥 call Gemini (NEW SDK)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=conversation
        )

        # 🔥 مهم جدًا
        if not response or not response.text:
            print("❌ EMPTY GEMINI RESPONSE")
            return None

        reply = response.text.strip()

        # تنظيف
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        return reply

    except Exception as e:
        print("❌ GEMINI ERROR:", str(e))
        logger.error(f"Gemini error: {e}")
        return None
