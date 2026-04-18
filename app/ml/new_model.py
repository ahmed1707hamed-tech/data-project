from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NextGenModel:

    def load_model(self):
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

        print(f"🚀 Loading model from HuggingFace: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

        print("✅ Model loaded successfully")

    def predict(self, text: str):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        label = labels[pred]

        return {
            "emotion": label,
            "confidence": float(confidence)
        }