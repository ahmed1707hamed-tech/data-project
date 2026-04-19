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

    model = genai.GenerativeModel("models/gemini-1.5-flash")

    # 🔥 نبني conversation حقيقي
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

    # 🔥 أهم سطر
    conversation += f"User: {text}\nAssistant:"

    print("🔥 FULL PROMPT:\n", conversation)

    # 🔥 بدون system prompt نهائي
    response = model.generate_content(
        conversation,
        generation_config={
            "temperature": 0.9,
            "top_p": 0.95,
            "max_output_tokens": 200,
        }
    )

    reply = response.text.strip()

    # 🔥 فلترة الرد الغبي
    if "tell me more" in reply.lower():
        return "It sounds like something is draining you. Try identifying what exactly is causing the stress and take one small step to reduce it."

    reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()

    return reply
