"""One loaded model, reused sequentially across every lane and every run.

`src/attn_patch.py`'s `_ACTIVE_CACHE` is a module-level global (its own
docstring: "generation is single-threaded and batch-1"). Two generation runs
must never be in flight at once against this model, so `RUN_LOCK` is acquired
for the *entire* duration of a run (prefill through the last decode step) -
not just around individual forward calls - and a second `start_run` while one
is held fails fast with a clear error instead of queueing or interleaving.
"""

from __future__ import annotations

import asyncio
import logging

from src.models import load_model

from .config import MODEL_ALIAS

logger = logging.getLogger(__name__)

RUN_LOCK = asyncio.Lock()

_model = None
_tokenizer = None


def get_model():
    """Load the model on first use (blocking - call this from a worker thread
    or during startup, never on the asyncio event loop thread)."""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading %s on CPU (allow_cpu=True: explicit local-demo choice, no GPU present)", MODEL_ALIAS)
        _model, _tokenizer = load_model(MODEL_ALIAS, allow_cpu=True)
    return _model, _tokenizer


def is_loaded() -> bool:
    return _model is not None
