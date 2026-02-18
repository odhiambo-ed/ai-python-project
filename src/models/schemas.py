from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)
    tokens: Optional[int] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-4o"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=128000)


class DataRecord(BaseModel):
    id: str
    text: str
    label: str
    quality_score: float = Field(ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
