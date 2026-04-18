from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import joblib


class NextGenModel:

    def load_model(self):
        model_path = "weights/my_model"

        print(f"🚀 Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise Exception(f"❌ Model path not found: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.eval()

        label_path = os.path.join(model_path, "label_encoder.pkl")
        if os.path.exists(label_path):
            self.label_encoder = joblib.load(label_path)
        else:
            self.label_encoder = None

        print("✅ Model loaded successfully")

    def predict(self, text: str):
        try:
            # 🔥 tokenize
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # 🔥 ناخد بس الحاجات اللي الموديل بيدعمها
            inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }

            # 🔥 inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

            # 🔥 label mapping
            if self.label_encoder:
                label = self.label_encoder.inverse_transform([pred])[0]
            else:
                labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
                label = labels[pred] if pred < len(labels) else str(pred)

            return {
                "emotion": str(label),
                "confidence": float(confidence)
            }

        except Exception as e:
            import traceback
            print("🔥 MODEL ERROR:")
            traceback.print_exc()
            raise e