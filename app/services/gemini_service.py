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
        # 🔥 بناء المحادثة من الذاكرة
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

        # 🔥 آخر رسالة
        conversation += f"User: {text}\nAssistant:"

        print("🔥 PROMPT:\n", conversation)

        # 🔥 Gemini API (النسخة الصح)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": conversation}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 200
            }
        }

        response = requests.post(url, json=payload)

        print("🔥 RAW:", response.text)

        # ❌ لو API ضرب
        if response.status_code != 200:
            print("❌ API ERROR:", response.text)
            return None

        data = response.json()

        # 🔥 استخراج الرد
        reply = data["candidates"][0]["content"]["parts"][0]["text"]

        # تنظيف
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply).strip()

        return reply

    except Exception as e:
        print("❌ GEMINI ERROR:", e)
        return None