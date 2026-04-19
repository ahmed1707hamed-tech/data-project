import google.generativeai as genai
import re
from typing import List, Union

from app.core.config import GEMINI_API_KEY
from app.schemas.chat_schema import Message


if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
) -> str:

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")

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

        response = model.generate_content(conversation)

        print("🔥 RAW RESPONSE:", response)

        # 🔥 أهم check
        if not response or not hasattr(response, "text") or not response.text:
            print("❌ Gemini returned empty response")
            return None

        reply = response.text.strip()

        # تنظيف
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

        return reply

    except Exception as e:
        print("❌ GEMINI ERROR:", str(e))
        return None
