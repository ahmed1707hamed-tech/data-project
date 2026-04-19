from google import genai
import re
from typing import List, Union

from app.core.config import GEMINI_API_KEY
from app.schemas.chat_schema import Message


print("🔥 GEMINI API KEY:", GEMINI_API_KEY)

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
):

    try:
        if not client:
            print("❌ NO GEMINI CLIENT")
            return None

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

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=conversation
        )

        print("🔥 RAW RESPONSE:", response)

        if not response:
            print("❌ EMPTY RESPONSE OBJECT")
            return None

        if not hasattr(response, "text") or not response.text:
            print("❌ NO TEXT IN RESPONSE")
            return None

        return response.text.strip()

    except Exception as e:
        print("❌ GEMINI ERROR:", str(e))
        return None
