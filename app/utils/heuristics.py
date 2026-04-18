ARABIC_EMOTION_KEYWORDS = {
    "sadness": ["حزين", "زعلان", "مكتئب", "تعبان", "وحيد"],
    "joy":     ["مبسوط", "فرحان", "سعيد", "مسرور"],
    "fear":    ["خايف", "قلقان", "مرعوب", "توتر"],
    "anger":   ["غاضب", "عصبي", "متضايق"],
}

FALLBACK_RESPONSES = {
    "ar": {
        "sadness":  "أنا هنا معاك 💙 حدثني أكثر.",
        "joy":      "جميل جدًا، يسعدني سماع ذلك 😊",
        "fear":     "لا تقلق، كل شيء هيتحسن 💪",
        "anger":    "خذ نفساً عميقاً، أنا معاك 🤍",
        "surprise": "يبدو ذلك مثيراً! أخبرني أكثر 😮",
        "default":  "أنا هنا للاستماع إليك 🤍",
    },
    "en": {
        "sadness":  "I'm here for you 💙 Tell me more.",
        "joy":      "That's wonderful, so glad to hear it 😊",
        "fear":     "Don't worry, things will get better 💪",
        "anger":    "Take a deep breath — I'm with you 🤍",
        "surprise": "That sounds exciting! Tell me more 😮",
        "default":  "I'm here and listening 🤍",
    },
}

def get_fallback(lang: str, emotion: str) -> str:
    """Fetch a safe fallback response based on language and emotion."""
    lang_map = FALLBACK_RESPONSES.get(lang, FALLBACK_RESPONSES["en"])
    return lang_map.get(emotion, lang_map["default"])
