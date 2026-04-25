def get_base_response(emotion: str) -> str:
    responses = {
        "sad": "أنا فاهم إنك حاسس بحزن أو تعب نفسي.",
        "angry": "واضح إنك متضايق أو غاضب من حاجة.",
        "happy": "جميل إنك حاسس بالسعادة!",
        "neutral": "تمام، خلينا نفهم أكتر."
    }

    return responses.get(emotion, "أنا سامعك.")