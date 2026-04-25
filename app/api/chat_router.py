from fastapi import APIRouter, UploadFile, Form
import os

from app.schemas.chat_schema import AnalyzeInput
from app.services.orchestrator import classify_emotion, generate_chat_response
from app.services.speech_service import speech_to_text
from app.services.emotion_service import detect_emotion_from_audio

router = APIRouter()


@router.post("/analyze")
def analyze_endpoint(data: AnalyzeInput):
    if not data.text or not data.text.strip():
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "language": "en",
        }

    return classify_emotion(data.text.strip())


@router.post("/chat")
async def chat_endpoint(
    input_type: str = Form(...),
    text: str = Form(None),
    emotion: str = Form(None),
    user_id: str = Form(None),
    file: UploadFile = None
):
    """
    Unified chat endpoint:
    - audio → يحسب text + emotion تلقائي
    - text → يستخدم النص مباشرة
    """

    # =========================
    # 🎤 AUDIO MODE
    # =========================
    if input_type == "audio":
        if file is None:
            return {"message": "Please upload an audio file."}

        if not file.content_type.startswith("audio"):
            return {"message": "Invalid file type"}

        os.makedirs("data/audio/temp", exist_ok=True)

        file_path = f"data/audio/temp/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        try:
            # ❗ تجاهل أي text/emotion جايين من المستخدم
            text = speech_to_text(file_path)
            emotion = detect_emotion_from_audio(file_path)

        finally:
            # تنظيف الملف مهما حصل
            if os.path.exists(file_path):
                os.remove(file_path)

    # =========================
    # 📝 TEXT MODE
    # =========================
    elif input_type == "text":
        if not text or not text.strip():
            return {
                "message": "Please send a valid message.",
                "emotion": "neutral",
                "ai": "system",
            }

        # لو المستخدم مبعتش emotion نحسبه
        if not emotion:
            result = classify_emotion(text.strip())
            emotion = result.get("emotion", "neutral")

    else:
        return {"message": "Invalid input_type. Use 'audio' or 'text'."}

    # =========================
    # 🔒 SAFETY
    # =========================
    if not emotion:
        emotion = "neutral"

    if not text or not text.strip():
        return {
            "message": "Could not process input.",
            "emotion": emotion,
            "ai": "system",
        }

    # =========================
    # 🤖 RESPONSE
    # =========================
    return generate_chat_response(
        text=text.strip(),
        emotion=emotion,
        user_id=user_id,
    )