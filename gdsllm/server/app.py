"""
GdsLLM — FastAPI Application

Provides the HTTP server with Ollama-style and OpenAI-compatible endpoints.
The inference engine is loaded on startup and shared across all requests.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gdsllm.engine import InferenceEngine

logger = logging.getLogger("gdsllm.server")

# Global engine instance — set during lifespan
engine: InferenceEngine | None = None

# Concurrency lock — single request at a time (GPU constraint)
inference_lock = asyncio.Lock()


def get_engine() -> InferenceEngine | None:
    """Access the engine via function call to avoid stale import references."""
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, shut down on exit."""
    global engine

    model_dir = app.state.model_dir
    preload = app.state.preload

    logger.info(f"Loading model from {model_dir} (preload={preload})...")
    engine = InferenceEngine(model_dir, preload=preload)
    info = engine.model_info
    logger.info(
        f"Model loaded: {info['name']} "
        f"({info['parameter_size']}, {info['quantization']}, "
        f"{info['num_layers']} layers, {info['size_bytes'] / 1e9:.1f} GB)"
    )

    yield

    logger.info("Shutting down engine...")
    if engine is not None:
        engine.shutdown()
        engine = None


def create_app(
    model_dir: str | None = None,
    preload: bool = True,
    model_root: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GdsLLM",
        version="0.1.0",
        description="LLM inference with NVMe-to-VRAM weight streaming via GPUDirect Storage",
        lifespan=lifespan,
    )

    # Store config in app state for lifespan access
    app.state.model_dir = model_dir or os.environ.get("GDSLLM_MODEL_DIR", "")
    app.state.preload = preload
    app.state.model_root = model_root or os.environ.get("GDSLLM_MODEL_ROOT", "")

    # CORS — allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from gdsllm.server.routes_api import router as api_router
    from gdsllm.server.routes_openai import router as openai_router

    app.include_router(api_router, prefix="/api")
    app.include_router(openai_router, prefix="/v1")

    @app.get("/")
    async def root():
        return {"status": "GdsLLM is running"}

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

    return app
