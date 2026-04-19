from pydantic import BaseModel


class Message(BaseModel):
    role: str
    text: str


class AnalyzeInput(BaseModel):
    """Input for the /analyze endpoint — text only."""
    text: str


class ChatInput(BaseModel):
    """Input for the /chat endpoint — requires the detected emotion from /analyze."""
    text: str
    emotion: str
    user_id: str = "default"
