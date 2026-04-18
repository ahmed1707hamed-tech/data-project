from fastapi import APIRouter
from app.schemas.chat_schema import AnalyzeInput, ChatInput
from app.services.orchestrator import classify_emotion, generate_chat_response

router = APIRouter()


@router.post("/analyze")
def analyze_endpoint(data: AnalyzeInput):
    """
    Step 1 — Emotion Classification Only.
    Send the user's raw text → get back the detected emotion & confidence.
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
    Step 2 — Gemini Chatbot Response.
    Send the user's text + the emotion from /analyze + conversation history
    → get back an empathetic AI-generated reply.
    """
    if not data.text or not data.text.strip():
        return {
            "message": "Please send a valid message.",
            "emotion": data.emotion,
            "ai": "system",
        }

    return generate_chat_response(data.text.strip(), data.emotion, data.history)
