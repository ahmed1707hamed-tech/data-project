import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.core.config import ml_resources, MAX_LEN
from app.utils.text_cleaner import clean_text

def preprocess_text(text: str) -> np.ndarray:
    """Tokenize and pad text using the loaded tokenizer."""
    tokenizer = ml_resources.get("tokenizer")
    if not tokenizer:
        raise RuntimeError("Tokenizer is not loaded properly.")
        
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    return padded.astype(np.int32)

def run_onnx_inference(text_en: str) -> tuple[str, float]:
    """Execute ONNX model prediction."""
    session = ml_resources.get("session")
    labels = ml_resources.get("labels")
    input_name = ml_resources.get("input_name")
    
    if session is None or labels is None or input_name is None:
        raise RuntimeError("ONNX session or labels are not loaded.")

    input_data = preprocess_text(text_en)
    outputs = session.run(None, {input_name: input_data})
    probs = outputs[0][0]

    index = int(np.argmax(probs))
    emotion = str(labels[index]).strip().lower()
    confidence = round(float(probs[index]), 4)
    
    return emotion, confidence
