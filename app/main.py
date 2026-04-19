import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.lifespan import lifespan
from app.api.chat_router import router as chat_router

# 🔥 إنشاء التطبيق
app = FastAPI(
    title="Emotional AI Chatbot",
    description="Production-ready bilingual emotion-aware chatbot powered by HuggingFace + Gemini API.",
    version="3.0.0",
    lifespan=lifespan
)

# 🔥 CORS (عشان frontend بعدين)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 Routes
app.include_router(chat_router)


# 🔥 Health Check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Emotional AI Chatbot running 🚀"
    }


# 🔥 تشغيل السيرفر (Lightning compatible)
if __name__ == "__main__":
    port = 8080  # ✅ مهم جدًا

    print(f"🚀 Starting server on http://0.0.0.0:{port}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
