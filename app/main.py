import socket
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.lifespan import lifespan
from app.api.chat_router import router as chat_router

app = FastAPI(
    title="Emotional AI Chatbot",
    description="Production-ready bilingual emotion-aware chatbot powered by ONNX + Gemini API.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Emotional AI Chatbot running 🚀"}


def find_free_port() -> int:
    """Bind to port 0 to let the OS assign any available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    port = find_free_port()
    print(f"🚀 Starting server on http://localhost:{port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
