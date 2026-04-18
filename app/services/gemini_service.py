import re
from typing import List
import google.generativeai as genai
from app.core.config import GEMINI_API_KEY, logger
from app.utils.heuristics import get_fallback
from app.schemas.chat_schema import Message
from app.services.translator_service import detect_language

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def _classify_intent(text: str, lang: str) -> str:
    t = text.lower()
    q_en = ["?", "how", "what", "why", "when", "where", "can you", "help", "advice"]
    q_ar = ["؟", "كيف", "ماذا", "لماذا", "متى", "أين", "هل", "مساعدة", "نصيحة", "ساعدني"]
    v_en = ["hate", "give up", "tired of", "sick of", "always", "never"]
    v_ar = ["تعبت", "بكره", "مش طايق", "دايما", "ابدا"]
    
    if any(m in t for m in (q_ar if lang == "ar" else q_en)): return "question"
    if any(m in t for m in (v_ar if lang == "ar" else v_en)): return "venting"
    return "statement"

def build_system_prompt(detected_emotion: str, language: str, intent: str) -> str:
    return f"""You are a highly empathetic, conversational human-like companion. 

CURRENT CONTEXT:
- Detected Emotion: {detected_emotion}
- User Intent: {intent}
- Response Language: {language}

STRICT INSTRUCTIONS:
1. You MUST respond ONLY in {language}.
2. Calibrate your tone to match and validate the '{detected_emotion}' emotion.
3. NEVER use phrases like "As an AI", "I understand how you feel", "I'm here for you", or "I am a virtual assistant".
4. Do not state the detected emotion explicitly (e.g., do not say "I see you are sad"). 
5. Keep the response natural, concise, and devoid of robotic formatting like bullet points unless directly asked for a list.
6. Address the user's intent ({intent}) directly without lecturing.
"""

def generate_gemini_response(text: str, emotion: str, lang: str, history: List[Message]) -> str:
    """Generate dynamic contextual AI responses via Gemini API."""
    if not GEMINI_API_KEY:
        return get_fallback(lang, emotion)

    intent = _classify_intent(text, lang)
    language_full = "Arabic" if lang == "ar" else "English"
    system_instruction = build_system_prompt(emotion, language_full, intent)

    try:
        model = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            system_instruction=system_instruction
        )
        
        formatted_history = []
        for msg in history:
            role = "model" if msg.role.lower() in ["model", "ai", "bot", "assistant"] else "user"
            formatted_history.append({"role": role, "parts": [msg.text]})

        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(text)
        reply = response.text.strip()
        
        reply = re.sub(r"^\[?reply\]?\s*:?\s*", "", reply, flags=re.IGNORECASE).strip()
        
        if detect_language(reply) != lang:
            return get_fallback(lang, emotion)
            
        return reply

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return get_fallback(lang, emotion)
