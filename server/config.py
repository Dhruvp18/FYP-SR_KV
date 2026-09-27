"""Demo constants. Not part of the research config surface (`configs/*.yaml`)."""

from __future__ import annotations

MODEL_ALIAS = "qwen2.5-0.5b"

BUDGET_MIN = 0.1
BUDGET_MAX = 0.5
BUDGET_DEFAULT = 0.3

#: raised from 40 - at ~0.15s/token-round on CPU (5 lanes, sequential), 80
#: tokens is roughly a minute, and longer completions give divergence between
#: methods more room to actually show up.
MAX_NEW_TOKENS_CAP = 80
MAX_NEW_TOKENS_DEFAULT = 24

#: generous on purpose - showing methods genuinely diverge (rather than all
#: keeping ~everything) needs a prompt long enough to force real compression,
#: e.g. a fact stated early + filler + a question about it. ~2000 chars is
#: still a few hundred tokens, a few seconds of CPU prefill at most.
PROMPT_MAX_CHARS = 2000

CORS_ORIGINS = ["http://localhost:3000"]
