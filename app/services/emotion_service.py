import librosa
import numpy as np
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

# 🎧 تحويل الصوت
def convert_audio(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path,
            "-y"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# 🎯 emotion detection بسيط وسريع
def detect_emotion_from_audio(file_path: str) -> str:
    try:
        os.makedirs("data/audio/temp", exist_ok=True)
        clean_path = "data/audio/temp/clean.wav"

        convert_audio(file_path, clean_path)

        # قراءة الصوت
        y, sr = librosa.load(clean_path, sr=16000)

        # features بسيطة
        energy = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # rules
        if energy > 0.1 and zcr > 0.1:
            return "angry"
        elif energy < 0.03:
            return "sad"
        elif energy > 0.07:
            return "happy"
        else:
            return "neutral"

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return "neutral"