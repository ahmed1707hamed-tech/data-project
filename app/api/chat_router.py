from fastapi import APIRouter
from pydantic import BaseModel

from app.schemas.chat_schema import AnalyzeInput
from app.services.orchestrator import classify_emotion, generate_chat_response

router = APIRouter()


# 🔥 Chat request الجديد (بدون history)
class ChatInput(BaseModel):
    text: str
    emotion: str
    user_id: str  # 🔥 مهم


@router.post("/analyze")
def analyze_endpoint(data: AnalyzeInput):
    """
    Step 1 — Emotion Classification Only.
    """
    if not data.text or not data.text.strip():
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "language": "en",
        }

    return classify_emotion(data.text.strip())


@router.post("/chat")
def chat_endpoint(data: ChatInput):
    """
    Step 2 — Chatbot with memory per user.
    """

    if not data.text or not data.text.strip():
        return {
            "message": "Please send a valid message.",
            "emotion": data.emotion,
            "ai": "system",
        }

    return generate_chat_response(
        text=data.text.strip(),
        emotion=data.emotion,
        user_id=data.user_id  # 🔥 هنا السر
    )