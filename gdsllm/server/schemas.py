"""
GdsLLM — Pydantic schemas for API request/response models.

Covers both Ollama-style native API and OpenAI-compatible endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ─── Shared types ─────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str = ""


class Options(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


# ─── Ollama-style: /api/generate ──────────────────────────────────────────────


class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = True
    options: Optional[Options] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str  # ISO 8601 timestamp
    response: str  # token text (streaming) or full text (non-streaming)
    done: bool
    done_reason: Optional[str] = None
    # Timing — populated when done=True
    total_duration: Optional[int] = None  # nanoseconds
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# ─── Ollama-style: /api/chat ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message] = Field(default_factory=list)
    stream: bool = True
    options: Optional[Options] = None


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# ─── Ollama-style: /api/tags ──────────────────────────────────────────────────


class ModelDetails(BaseModel):
    family: str = "llama"
    parameter_size: str = ""
    quantization_level: str = ""


class ModelInfo(BaseModel):
    name: str
    modified_at: str = ""
    size: int = 0
    details: ModelDetails = Field(default_factory=ModelDetails)


class TagsResponse(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)


# ─── Ollama-style: /api/show ─────────────────────────────────────────────────


class ShowRequest(BaseModel):
    name: str


class ShowResponse(BaseModel):
    name: str
    details: ModelDetails = Field(default_factory=ModelDetails)
    parameters: dict = Field(default_factory=dict)


# ─── OpenAI-compatible: /v1/chat/completions ──────────────────────────────────


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = None  # "stop", "length"


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int  # unix timestamp
    model: str
    choices: list[Choice]
    usage: Usage


# ─── OpenAI-compatible: streaming chunks ──────────────────────────────────────


class ChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[str] = None


class OpenAIChatChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
