from pydantic import BaseModel, Field, field_validator


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
    user_id: str = Field(..., min_length=1)

    @field_validator("user_id")
    @classmethod
    def normalize_user_id(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("user_id must be a non-empty string")
        return s
