# Emotional AI Chatbot 🤖

A production-ready bilingual (Arabic & English) backend system powered by **FastAPI**. It combines deep learning for emotion classification (via ONNX BiLSTM) with Google's Generative AI (Gemini 1.5 Flash) to generate empathetic, context-aware chatbot responses.

## 📂 Project Structure

This project follows a highly modular domain-driven structure, allowing seamless future expansions:

```text
emotional_chatbot/
├── app/                  # Main Application Source Code
│   ├── api/              # FastAPI Routers & Endpoints (`chat_router.py`)
│   ├── core/             # App lifecycles and Configuration keys (`config.py`, `lifespan.py`)
│   ├── schemas/          # Pydantic typing and models (`chat_schema.py`)
│   ├── services/         # Business Logic! (Gemini API, Orchestrators, Language mapping)
│   ├── ml/               # Machine Learning integrations (`emotion_bilstm.py` and stubs)
│   └── utils/            # Shared utilities (Regex text cleaners, Fallback Logic)
├── weights/              # ML Models Storage
│   ├── emotion_bilstm/   # Current active models (Contains `.onnx`, `.pkl`)
│   └── next_model/       # Place directory for future models
├── Dockerfile            # Container configuration
└── pyproject.toml / uv.lock # Python environment management
```

## 🛠️ Tech Stack & Dependencies

- **FastAPI** + **Uvicorn/Gunicorn**: Lightning-fast web framework and server routing.
- **ONNXRuntime**: High-performance optimization engine for trained Neural Networks.
- **Google Generative AI**: LLM backend (Gemini 1.5 Flash).
- **Deep-Translator**: On-the-fly cross-language bridging.
- **UV Package Manager**: Next-generation, hyper-fast pip replacement.

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.11+
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`)

### 2. Environment Variables
Ensure you have a `.env` file at the root containing:
```ini
GEMINI_API_KEY="your-gemini-key-here"
```

### 3. Start the Server
Start the FastAPI server via `uv`:
```bash
uv run python -m app.main
```
The application will automatically detect and bind to an available free port on your system.

---

## 🔌 API Endpoints

### 1. Health Check
`GET /`
Returns the status of the server.

### 2. Emotion Analysis (Step 1 — ML Model)
`POST /analyze`

Send the user's raw text. The ML model classifies the emotion and returns the result.

**Request**:
```json
{
  "text": "I feel so lonely today 😔"
}
```

**Response**:
```json
{
  "emotion": "sadness",
  "confidence": 0.98,
  "language": "en"
}
```

### 3. Chat Response (Step 2 — Gemini Chatbot)
`POST /chat`

Send the user's text **along with the emotion detected from `/analyze`** and the conversation history. The Gemini LLM generates a contextual, empathetic reply.

**Request**:
```json
{
  "text": "I feel so lonely today 😔",
  "emotion": "sadness",
  "history": [
    {"role": "user", "text": "Are you there?"},
    {"role": "ai", "text": "I'm always here for you. How was your day?"}
  ]
}
```

**Response**:
```json
{
  "message": "That sounds really tough. Do you want to talk about what's going on?",
  "emotion": "sadness",
  "ai": "gemini"
}
```

---

## 📈 How to Add a New ML Model

To scale the engine or test a completely new architecture (e.g., a Transformer-based sentiment model):

1. **Place Model Weights**:
   Put your newly exported weights (like `pytorch_model.bin` or `.onnx`) into `weights/next_model/`.
2. **Implement Model Logic**:
   Open `app/ml/new_model.py`. You will notice it inherits from an abstract class `BaseModelConfig` located in `app/ml/base.py`. Use it to instantiate your `load_model` and `predict` interfaces natively.
3. **Plug into Orchestrator**:
   Go to `app/services/orchestrator.py` and import/plug-in your new classifier logic beneath the original translation mapping.
