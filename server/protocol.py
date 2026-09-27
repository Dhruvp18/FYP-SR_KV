"""WebSocket wire format. Hand-mirrored in `webapp/lib/types.ts` — this file
is the single source of truth for the shape; keep both in sync by hand.

Client -> server messages are validated with pydantic (free error messages on
a bad payload). Server -> client messages are emitted as plain dicts straight
out of `server/decode_loop.py`'s `emit()` callback — their shape is documented
here, not enforced, since we control every value we put in them.

Server -> client message shapes (all include "type"):
    run_started   {lanes: [str], variant: "attribution"|"aggregation",
                   question: str, options: [str, str, str, str],
                   passage_tokens: int, passage_token_texts: [str],
                   question_token_texts: [str], budget: float, max_new_tokens: int}
                   (answer_letter is deliberately absent - revealed only in run_complete)
    step          {lane, step, phase: "passage"|"question"|"decode",
                   position: int|null, token_text,
                    narration: str (plain-English summary of this step),
                    stats: {n_tokens_cached, n_tokens_evicted, n_centroids, budget_used_pct},
                    conservation_ok: bool,
                    slot_snapshot: {positions: [float], weights: [int], is_centroid: [bool]},
                    diff: {evicted_positions: [int], folded_positions: [int],
                           evicted_positions_cumulative: [int]},
                   batch_cell?: {variant, budget, sample_idx, n_per_cell} (present only during a batch run)}
    lane_complete {lane, generated_text, n_tokens_generated, chosen_letter: str|null, correct: bool}
    run_complete  {summary: [{lane, generated_text, generated_ids, final_stats,
                              conservation_ok, chosen_letter, correct}],
                   answer_letter: str}
                   (single-question mode only - a batch run's per-sample completion
                   is `batch_sample_complete` instead, see below)
    batch_sample_complete {variant, budget, sample_idx, n_per_cell,
                           summary: [... same shape as run_complete's summary rows ...]}
                           (budget is always server/batch.py's fixed BATCH_BUDGET)
    batch_progress {done: int, total: int, table: {variant: {lane: {correct, total}}}}
    batch_complete {table: {...same shape as batch_progress's table...}, cancelled: bool}
    error         {message: str}
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import BUDGET_MAX, BUDGET_MIN, GIST_CONTEXT_LEN_DEFAULT, GIST_CONTEXT_LEN_MAX, GIST_CONTEXT_LEN_MIN
from .gist_source import VARIANTS


class StartGistRun(BaseModel):
    type: str = Field(default="start_gist_run", frozen=True)
    budget: float = Field(ge=BUDGET_MIN, le=BUDGET_MAX)
    variant: str | None = Field(default=None)
    context_len: int = Field(default=GIST_CONTEXT_LEN_DEFAULT, ge=GIST_CONTEXT_LEN_MIN, le=GIST_CONTEXT_LEN_MAX)

    def model_post_init(self, __context) -> None:
        if self.variant is not None and self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS} or null, got {self.variant!r}")


class StartBatchRun(BaseModel):
    """No budget/variant/n fields on purpose - those are the fixed constants
    in `server/batch.py` (a single demo budget, not the pre-registered
    0.1/0.15) that keep every consumer's framing consistent. Only the
    passage length is adjustable, same range as the single-question mode."""

    type: str = Field(default="start_batch_run", frozen=True)
    context_len: int = Field(default=GIST_CONTEXT_LEN_DEFAULT, ge=GIST_CONTEXT_LEN_MIN, le=GIST_CONTEXT_LEN_MAX)


class CancelRun(BaseModel):
    type: str = Field(default="cancel_run", frozen=True)


def parse_client_message(raw: dict) -> StartGistRun | StartBatchRun | CancelRun:
    msg_type = raw.get("type")
    if msg_type == "start_gist_run":
        return StartGistRun(**raw)
    if msg_type == "start_batch_run":
        return StartBatchRun(**raw)
    if msg_type == "cancel_run":
        return CancelRun(**raw)
    raise ValueError(f"unknown message type {msg_type!r}")
