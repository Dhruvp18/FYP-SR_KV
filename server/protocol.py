"""WebSocket wire format. Hand-mirrored in `webapp/lib/types.ts` — this file
is the single source of truth for the shape; keep both in sync by hand.

Client -> server messages are validated with pydantic (free error messages on
a bad payload). Server -> client messages are emitted as plain dicts straight
out of `server/decode_loop.py`'s `emit()` callback — their shape is documented
here, not enforced, since we control every value we put in them.

Server -> client message shapes (all include "type"):
    run_started   {lanes: [str], prompt_tokens: int, prompt_token_texts: [str],
                   budget: float, max_new_tokens: int}
    step          {lane, step, phase: "prefill"|"decode", position: int|null, token_text,
                    narration: str (plain-English summary of this step),
                    stats: {n_tokens_cached, n_tokens_evicted, n_centroids, budget_used_pct},
                    conservation_ok: bool,
                    slot_snapshot: {positions: [float], weights: [int], is_centroid: [bool]},
                    diff: {merged: [{position, weight, member_positions: [int]}],
                           evicted_positions: [int], folded_positions: [int],
                           evicted_positions_cumulative: [int]}}
    lane_complete {lane, generated_text, n_tokens_generated}
    run_complete  {summary: [{lane, generated_text, generated_ids, final_stats, conservation_ok}]}
    error         {message: str}
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import BUDGET_MAX, BUDGET_MIN, MAX_NEW_TOKENS_CAP, PROMPT_MAX_CHARS


class StartRun(BaseModel):
    type: str = Field(default="start_run", frozen=True)
    prompt: str = Field(min_length=1, max_length=PROMPT_MAX_CHARS)
    budget: float = Field(ge=BUDGET_MIN, le=BUDGET_MAX)
    max_new_tokens: int = Field(ge=1, le=MAX_NEW_TOKENS_CAP)


class CancelRun(BaseModel):
    type: str = Field(default="cancel_run", frozen=True)


def parse_client_message(raw: dict) -> StartRun | CancelRun:
    msg_type = raw.get("type")
    if msg_type == "start_run":
        return StartRun(**raw)
    if msg_type == "cancel_run":
        return CancelRun(**raw)
    raise ValueError(f"unknown message type {msg_type!r}")
