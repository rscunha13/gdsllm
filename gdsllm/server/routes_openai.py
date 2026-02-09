"""
GdsLLM — OpenAI-Compatible API Routes

Endpoints:
    POST /v1/chat/completions  — chat completion (SSE stream or JSON)
    GET  /v1/models            — list available models
"""

import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gdsllm.server import schemas
from gdsllm.server.app import get_engine, inference_lock

router = APIRouter()


def _gen_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


# ─── POST /v1/chat/completions ────────────────────────────────────────────────


@router.post("/chat/completions")
async def openai_chat_completions(req: schemas.OpenAIChatRequest):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    sp = {
        "max_tokens": req.max_tokens or 128,
        "temperature": req.temperature if req.temperature is not None else 0.0,
        "top_p": req.top_p if req.top_p is not None else 1.0,
        "top_k": req.top_k if req.top_k is not None else 0,
        "repeat_penalty": req.repeat_penalty if req.repeat_penalty is not None else 1.0,
    }

    if req.stream:
        return StreamingResponse(
            _stream_openai_chat(messages, **sp),
            media_type="text/event-stream",
        )
    else:
        return await _openai_chat_full(messages, **sp)


async def _openai_chat_full(messages: list[dict], **sp):
    """Non-streaming: collect all tokens, return OpenAI-format JSON."""
    engine = get_engine()
    async with inference_lock:
        full_text = []
        final_event = None
        for event in engine.generate_chat(messages, **sp):
            full_text.append(event.text)
            if event.done:
                final_event = event

        content = "".join(full_text)
        prompt_tokens = final_event.prompt_tokens if final_event else 0
        completion_tokens = final_event.completion_tokens if final_event else 0

        return schemas.OpenAIChatResponse(
            id=_gen_id(),
            created=int(time.time()),
            model=engine.model_name,
            choices=[
                schemas.Choice(
                    index=0,
                    message=schemas.ChoiceMessage(
                        role="assistant", content=content,
                    ),
                    finish_reason=final_event.done_reason if final_event else "stop",
                )
            ],
            usage=schemas.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


async def _stream_openai_chat(messages: list[dict], **sp):
    """Streaming: yield SSE events in OpenAI format."""
    engine = get_engine()
    chat_id = _gen_id()
    created = int(time.time())

    async with inference_lock:
        # First chunk: role
        first_chunk = schemas.OpenAIChatChunk(
            id=chat_id,
            created=created,
            model=engine.model_name,
            choices=[
                schemas.ChunkChoice(
                    index=0,
                    delta=schemas.ChunkDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Content chunks
        for event in engine.generate_chat(messages, **sp):
            if event.done:
                chunk = schemas.OpenAIChatChunk(
                    id=chat_id,
                    created=created,
                    model=engine.model_name,
                    choices=[
                        schemas.ChunkChoice(
                            index=0,
                            delta=schemas.ChunkDelta(),
                            finish_reason=event.done_reason or "stop",
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                chunk = schemas.OpenAIChatChunk(
                    id=chat_id,
                    created=created,
                    model=engine.model_name,
                    choices=[
                        schemas.ChunkChoice(
                            index=0,
                            delta=schemas.ChunkDelta(content=event.text),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"


# ─── GET /v1/models ───────────────────────────────────────────────────────────


@router.get("/models")
async def openai_list_models():
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    info = engine.model_info
    return {
        "object": "list",
        "data": [
            {
                "id": info["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gdsllm",
            }
        ],
    }
