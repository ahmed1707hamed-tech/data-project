import whisper

model = whisper.load_model("base")

def speech_to_text(file_path: str) -> str:
    result = model.transcribe(file_path)
    return result["text"]