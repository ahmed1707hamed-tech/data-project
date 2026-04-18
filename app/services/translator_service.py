import re
from deep_translator import GoogleTranslator
from app.core.config import logger

ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

def detect_language(text: str) -> str:
    """Returns 'ar' if the text contains Arabic characters, 'en' otherwise."""
    return "ar" if ARABIC_PATTERN.search(text.strip()) else "en"

def translate_to_english(text: str) -> str:
    """Safely translate text to English for the ML Model."""
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text
