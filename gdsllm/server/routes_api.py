"""
GdsLLM — Ollama-style API Routes

Endpoints:
    POST /api/generate  — text completion (streaming NDJSON or JSON)
    POST /api/chat      — chat completion (streaming NDJSON or JSON)
    GET  /api/tags      — list local models
    POST /api/show      — model info & config
"""

import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gdsllm.server import schemas
from gdsllm.server.app import get_engine, inference_lock

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scan_models(model_root: str) -> list[schemas.ModelInfo]:
    """Scan a directory for GdsLLM model subdirectories (containing metadata.json)."""
    models = []
    if not model_root or not os.path.isdir(model_root):
        return models

    for entry in sorted(os.listdir(model_root)):
        meta_path = os.path.join(model_root, entry, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Calculate total size
            total_size = 0
            model_dir = os.path.join(model_root, entry)
            for root, _, files in os.walk(model_dir):
                for fname in files:
                    total_size += os.path.getsize(os.path.join(root, fname))

            stat = os.stat(meta_path)
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            quant = meta.get("quantization", "none")
            hidden = meta.get("hidden_size", 0)
            layers = meta.get("num_layers", 0)
            params = 12 * hidden * hidden * layers
            if params > 50e9:
                param_str = f"{params / 1e9:.0f}B"
            elif params > 1e9:
                param_str = f"{params / 1e9:.1f}B"
            else:
                param_str = f"{params / 1e6:.0f}M"

            models.append(schemas.ModelInfo(
                name=entry,
                modified_at=modified,
                size=total_size,
                details=schemas.ModelDetails(
                    family="llama",
                    parameter_size=param_str,
                    quantization_level=quant.upper() if quant != "none" else "FP16",
                ),
            ))
        except (json.JSONDecodeError, OSError):
            continue
    return models


# ─── POST /api/generate ───────────────────────────────────────────────────────


@router.post("/generate")
async def api_generate(req: schemas.GenerateRequest):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    temperature = 0.0
    max_tokens = 128
    if req.options:
        if req.options.temperature is not None:
            temperature = req.options.temperature
        if req.options.max_tokens is not None:
            max_tokens = req.options.max_tokens

    if req.stream:
        return StreamingResponse(
            _stream_generate(req.prompt, max_tokens, temperature),
            media_type="application/x-ndjson",
        )
    else:
        return await _generate_full(req.prompt, max_tokens, temperature)


async def _generate_full(prompt: str, max_tokens: int, temperature: float):
    """Non-streaming: collect all tokens, return single JSON response."""
    engine = get_engine()
    async with inference_lock:
        full_text = []
        final_event = None
        for event in engine.generate(prompt, max_tokens, temperature, chat=False):
            full_text.append(event.text)
            if event.done:
                final_event = event

        return schemas.GenerateResponse(
            model=engine.model_name,
            created_at=_now_iso(),
            response="".join(full_text),
            done=True,
            done_reason=final_event.done_reason if final_event else None,
            total_duration=final_event.total_duration_ns if final_event else None,
            prompt_eval_count=final_event.prompt_tokens if final_event else None,
            prompt_eval_duration=final_event.prefill_duration_ns if final_event else None,
            eval_count=final_event.completion_tokens if final_event else None,
            eval_duration=final_event.eval_duration_ns if final_event else None,
        )


async def _stream_generate(prompt: str, max_tokens: int, temperature: float):
    """Streaming: yield NDJSON lines, one per token."""
    engine = get_engine()
    async with inference_lock:
        for event in engine.generate(prompt, max_tokens, temperature, chat=False):
            resp = schemas.GenerateResponse(
                model=engine.model_name,
                created_at=_now_iso(),
                response=event.text,
                done=event.done,
                done_reason=event.done_reason if event.done else None,
                total_duration=event.total_duration_ns if event.done else None,
                prompt_eval_count=event.prompt_tokens if event.done else None,
                prompt_eval_duration=event.prefill_duration_ns if event.done else None,
                eval_count=event.completion_tokens if event.done else None,
                eval_duration=event.eval_duration_ns if event.done else None,
            )
            yield resp.model_dump_json() + "\n"


# ─── POST /api/chat ───────────────────────────────────────────────────────────


@router.post("/chat")
async def api_chat(req: schemas.ChatRequest):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    temperature = 0.0
    max_tokens = 128
    if req.options:
        if req.options.temperature is not None:
            temperature = req.options.temperature
        if req.options.max_tokens is not None:
            max_tokens = req.options.max_tokens

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_chat(messages, max_tokens, temperature),
            media_type="application/x-ndjson",
        )
    else:
        return await _chat_full(messages, max_tokens, temperature)


async def _chat_full(messages: list[dict], max_tokens: int, temperature: float):
    """Non-streaming chat: collect all tokens, return single JSON."""
    engine = get_engine()
    async with inference_lock:
        full_text = []
        final_event = None
        for event in engine.generate_chat(messages, max_tokens, temperature):
            full_text.append(event.text)
            if event.done:
                final_event = event

        content = "".join(full_text)
        return schemas.ChatResponse(
            model=engine.model_name,
            created_at=_now_iso(),
            message=schemas.Message(role="assistant", content=content),
            done=True,
            done_reason=final_event.done_reason if final_event else None,
            total_duration=final_event.total_duration_ns if final_event else None,
            prompt_eval_count=final_event.prompt_tokens if final_event else None,
            prompt_eval_duration=final_event.prefill_duration_ns if final_event else None,
            eval_count=final_event.completion_tokens if final_event else None,
            eval_duration=final_event.eval_duration_ns if final_event else None,
        )


async def _stream_chat(messages: list[dict], max_tokens: int, temperature: float):
    """Streaming chat: yield NDJSON lines, one per token."""
    engine = get_engine()
    async with inference_lock:
        for event in engine.generate_chat(messages, max_tokens, temperature):
            resp = schemas.ChatResponse(
                model=engine.model_name,
                created_at=_now_iso(),
                message=schemas.Message(role="assistant", content=event.text),
                done=event.done,
                done_reason=event.done_reason if event.done else None,
                total_duration=event.total_duration_ns if event.done else None,
                prompt_eval_count=event.prompt_tokens if event.done else None,
                prompt_eval_duration=event.prefill_duration_ns if event.done else None,
                eval_count=event.completion_tokens if event.done else None,
                eval_duration=event.eval_duration_ns if event.done else None,
            )
            yield resp.model_dump_json() + "\n"


# ─── GET /api/tags ────────────────────────────────────────────────────────────


@router.get("/tags")
async def api_tags():
    engine = get_engine()
    models = []

    # Include currently loaded model
    if engine is not None:
        info = engine.model_info
        models.append(schemas.ModelInfo(
            name=info["name"],
            modified_at=_now_iso(),
            size=info.get("size_bytes", 0),
            details=schemas.ModelDetails(
                family=info.get("family", "llama"),
                parameter_size=info.get("parameter_size", ""),
                quantization_level=info.get("quantization", "none").upper(),
            ),
        ))

    # Scan model root for additional models
    if engine is not None:
        model_root = os.path.dirname(engine.model_dir)
        scanned = _scan_models(model_root)
        loaded_names = {m.name for m in models}
        for m in scanned:
            if m.name not in loaded_names:
                models.append(m)

    return schemas.TagsResponse(models=models)


# ─── POST /api/show ───────────────────────────────────────────────────────────


@router.post("/show")
async def api_show(req: schemas.ShowRequest):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    info = engine.model_info
    if req.name and req.name != info["name"]:
        raise HTTPException(status_code=404, detail=f"Model '{req.name}' not loaded")

    return schemas.ShowResponse(
        name=info["name"],
        details=schemas.ModelDetails(
            family=info.get("family", "llama"),
            parameter_size=info.get("parameter_size", ""),
            quantization_level=info.get("quantization", "none").upper(),
        ),
        parameters=engine.config,
    )
