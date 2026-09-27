"""Demo constants. Not part of the research config surface (`configs/*.yaml`)."""

from __future__ import annotations

MODEL_ALIAS = "qwen2.5-0.5b"

BUDGET_MIN = 0.1
BUDGET_MAX = 0.5
BUDGET_DEFAULT = 0.3

#: passage length (server/gist_source.py's `context_len`). Real Phase 8 runs
#: use 2048-16384; short for a live CPU demo (5 lanes, sequential, plus the
#: question pass and decode) to stay in the tens-of-seconds range.
GIST_CONTEXT_LEN_MIN = 200
GIST_CONTEXT_LEN_MAX = 800
GIST_CONTEXT_LEN_DEFAULT = 400

CORS_ORIGINS = ["http://localhost:3000"]
