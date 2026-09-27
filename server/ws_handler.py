"""The one WebSocket endpoint: `/ws/generate`.

One inbound loop (receives `start_run`/`cancel_run`) and, per run, one
background task that drives the decode loop in a worker thread and drains its
events onto the same socket. Only one run may hold `model_singleton.RUN_LOCK`
at a time - a second `start_run` while one is in flight gets an immediate
`error`, never a queued or interleaved run (see `model_singleton.py`).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from . import model_singleton
from .decode_loop import run as run_decode_loop
from .protocol import CancelRun, StartRun, parse_client_message

logger = logging.getLogger(__name__)

_SENTINEL = object()


async def handle_connection(websocket: WebSocket) -> None:
    await websocket.accept()
    cancel_event: asyncio.Event | None = None

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                msg = parse_client_message(raw)
            except Exception as exc:  # bad payload - tell the client, keep the socket open
                await websocket.send_json({"type": "error", "message": f"bad message: {exc}"})
                continue

            if isinstance(msg, CancelRun):
                if cancel_event is not None:
                    cancel_event.set()
                continue

            if model_singleton.RUN_LOCK.locked():
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "A generation run is already in progress on this server "
                        "(one shared model, one run at a time). Wait for it to finish or cancel it.",
                    }
                )
                continue

            cancel_event = asyncio.Event()
            asyncio.create_task(_run_and_stream(websocket, msg, cancel_event))
    except WebSocketDisconnect:
        if cancel_event is not None:
            cancel_event.set()


async def _run_and_stream(websocket: WebSocket, msg: StartRun, cancel_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def worker() -> None:
        async with model_singleton.RUN_LOCK:
            try:
                model, tokenizer = await asyncio.to_thread(model_singleton.get_model)
                await asyncio.to_thread(
                    run_decode_loop,
                    model,
                    tokenizer,
                    msg.prompt,
                    budget=msg.budget,
                    max_new_tokens=msg.max_new_tokens,
                    overrides=None,
                    emit=emit,
                    should_cancel=cancel_event.is_set,
                )
            except Exception:
                logger.exception("decode loop failed")
                emit({"type": "error", "message": "generation failed - see server logs for details"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    worker_task = asyncio.create_task(worker())
    try:
        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            try:
                await websocket.send_json(event)
            except Exception:
                break  # client is gone; let cancel_event (set by the disconnect handler) stop the worker
    finally:
        await worker_task
