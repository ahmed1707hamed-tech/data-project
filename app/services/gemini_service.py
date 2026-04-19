import google.generativeai as genai
import re
from typing import List, Union

from app.core.config import GEMINI_API_KEY
from app.schemas.chat_schema import Message

genai.configure(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
):

    try:
        model = genai.GenerativeModel("gemini-pro")

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

        if not response or not response.text:
            print("❌ EMPTY RESPONSE")
            return None

        reply = response.text.strip()

        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply).strip()

        return reply

    except Exception as e:
        print("❌ GEMINI ERROR:", e)
        return None