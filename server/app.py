"""FastAPI app for the live SR-KV demo. Run with:

    uvicorn server.app:app --reload --port 8000

(from the repo root, with the project venv active).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from . import model_singleton
from .config import CORS_ORIGINS
from .ws_handler import handle_connection

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Cache Lens backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _load_model_eagerly() -> None:
    # Loading Qwen2.5-0.5B can take a while (download + CPU init); do it once
    # at startup, off the event loop thread, so the first browser run isn't
    # the one paying that cost.
    await asyncio.to_thread(model_singleton.get_model)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": model_singleton.is_loaded()}


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket) -> None:
    await handle_connection(websocket)
