import requests
import re
from typing import List, Union

from app.core.config import GEMINI_API_KEY
from app.schemas.chat_schema import Message


def generate_gemini_response(
    text: str,
    emotion: str,
    lang: str,
    history: List[Union[Message, dict]]
):

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

        # 🔥 REST API (مهم جدًا)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": conversation}
                    ]
                }
            ]
        }

        response = requests.post(url, json=payload)

        print("🔥 RAW:", response.text)

        if response.status_code != 200:
            print("❌ API ERROR:", response.text)
            return None

        data = response.json()

        reply = data["candidates"][0]["content"]["parts"][0]["text"]

        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply).strip()

        return reply

    except Exception as e:
        print("❌ GEMINI ERROR:", e)
        return None