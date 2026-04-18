from app.ml.new_model import NextGenModel

model = NextGenModel()
model.load_model()

print("🔥 Testing model...")

result = model.predict("I feel sad today")

print("✅ RESULT:", result)