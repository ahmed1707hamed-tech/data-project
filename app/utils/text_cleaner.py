import re

def clean_text(text: str) -> str:
    """Normalize and clean text before tokenization."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()
