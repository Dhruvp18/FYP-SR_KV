"""The manual, interleaved, round-robin multi-lane decode loop.

Why this exists instead of `model.generate()`: the repo has no streaming
primitive (no `TextIteratorStreamer`/`StoppingCriteria`/`BaseStreamer`
anywhere), and `model.generate()` runs one lane to completion before you'd see
the next lane start - the demo wants all 5 lanes visibly advancing together,
one token per lane per round. This module is the only place that calls
`model(...)` directly instead of going through `generate()`.

It reuses everything else as-is: `make_cache` builds the caches, `attach_cache`
routes attention through them, `eval.gist_mcq` supplies the question and
scores the answer, and every reported number
(`get_stats()`/`check_conservation()`) is read straight off the cache, never
recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import RepetitionPenaltyLogitsProcessor

from eval.gist_mcq import (
    GistSample,
    _AFTER_ANSWER_RE,
    _LEADING_LETTER_RE,
    _LETTER_PAREN_RE,
    _two_pass_texts,
)
from src.attn_patch import attach_cache

from .lanes import LANE_METHODS, USES_CLUSTERING, build_lanes
from .telemetry import Diff, LaneTelemetryState, snapshot

EmitFn = Callable[[dict], None]


@dataclass
class LaneState:
    name: str
    cache: object
    next_pos: int = 0
    last_token: torch.Tensor | None = None
    all_ids: torch.Tensor | None = None  # everything seen so far, for repetition_penalty
    done: bool = False
    generated_ids: list[int] = field(default_factory=list)
    telemetry: LaneTelemetryState = None  # set in run_gist() - needs this lane's use_clustering flag


def _forward_step(model, cache, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """One forward pass through `model` with `cache` attached. Returns last-token logits.

    `position_ids` is always passed explicitly - never left to default - because
    `Qwen2Model.forward` derives a default from `past_key_values.get_seq_length()`,
    which is the *compressed* slot count once a lane has evicted anything, not
    the true absolute position. The cache's own bookkeeping (positions/weights/
    get_stats) does not depend on this either way - `SRKVCacheBase._pos_counter`
    tracks it internally per `update()` call - but the rotary embedding angle
    used in the actual attention math does, so getting this wrong would silently
    degrade generation quality on every compressed lane without ever showing up
    as a wrong stat. Attention masking needs no equivalent care: batch-1,
    unpadded, so `attention_mask=None` is exactly right (CLAUDE.md A6).
    """
    with attach_cache(model, cache):
        out = model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :]


def _narrate(lane: LaneState, diff: Diff, phase: str) -> str:
    """One plain-English sentence describing what this step just did to the
    cache - so the UI never depends on a viewer correctly reading a diagram.
    """
    stats = lane.cache.get_stats()
    if diff.newly_folded_positions:
        return (
            f"Folded {len(diff.newly_folded_positions)} more token(s) into the centroid pool "
            f"(now {stats['n_centroids']} slot(s), {len(diff.folded_positions)} tokens folded so far)."
        )
    if diff.evicted_positions:
        return f"Forgot {len(diff.evicted_positions)} old token(s) to stay within budget."
    if phase in ("passage", "question"):
        return f"Read the {phase} — {stats['n_tokens_cached']} tokens cached."
    return "Added the new token — still within budget, nothing removed."


def _step_event(
    lane: LaneState, *, step: int, phase: str, position: int | None, diff: Diff, snap, tokenizer
) -> dict:
    token_text = tokenizer.decode(lane.generated_ids[-1:], skip_special_tokens=True) if lane.generated_ids else ""
    return {
        "type": "step",
        "lane": lane.name,
        "step": step,
        "phase": phase,
        "position": position,
        "token_text": token_text,
        "narration": _narrate(lane, diff, phase),
        "stats": lane.cache.get_stats(),
        "conservation_ok": bool(lane.cache.check_conservation()),
        "slot_snapshot": {
            "positions": snap.positions,
            "weights": snap.weights,
            "is_centroid": snap.is_centroid,
        },
        "diff": {
            "evicted_positions": diff.evicted_positions,
            "folded_positions": diff.folded_positions,
            "evicted_positions_cumulative": diff.evicted_positions_cumulative,
        },
    }


def _eos_ids(model, tokenizer) -> set[int]:
    """Every token id that should end generation - `generate()` accepts
    `eos_token_id` as a scalar or a list; Qwen2.5-Instruct ships a list
    (`<|im_end|>` alongside the tokenizer's own eos)."""
    ids: set[int] = set()
    cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if cfg_eos is not None:
        ids.update(cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def _argmax_next(logits: torch.Tensor, all_ids: torch.Tensor, rep_penalty) -> torch.Tensor:
    """Greedy next-token pick, replicating `generate(do_sample=False)`'s own
    logits processing - not just a raw argmax. Confirmed necessary
    empirically: Qwen2.5-0.5B-Instruct's `generation_config` ships
    `repetition_penalty=1.1`, which `model.generate()` applies even in greedy
    mode (it is a `LogitsProcessor`, not a sampling-only knob) and which
    changes the argmax on any prompt long enough to repeat itself.
    """
    if rep_penalty is not None:
        logits = rep_penalty(all_ids, logits)
    return logits.argmax(dim=-1, keepdim=True)


def _extract_choice(generated_text: str) -> str | None:
    """The MCQ letter the generation chose, or None. Reuses the exact
    compiled patterns and precedence `eval.gist_mcq.score()` uses, so scoring
    here can never quietly drift from the real eval's own definition of
    "answered A/B/C/D"."""
    text = generated_text.strip().upper()
    for pattern in (_LETTER_PAREN_RE, _AFTER_ANSWER_RE, _LEADING_LETTER_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


@torch.no_grad()
def run_gist(
    model,
    tokenizer,
    sample: GistSample,
    *,
    budget: float,
    overrides: dict | None,
    emit: EmitFn,
    should_cancel: Callable[[], bool] = lambda: False,
) -> dict:
    """Run all 5 lanes on the same real eval question (`eval.gist_mcq`),
    two-pass, interleaved one token at a time.

    Mirrors `eval.gist_mcq.measure()`'s own two-pass discipline exactly: the
    passage prefills and compresses with zero knowledge the question exists,
    then the question is prefilled as a second forward pass, continuing
    position from `cache.t_now + 1` - the same fix documented in `measure()`'s
    own docstring (without it, HF derives new `position_ids` from the
    *compressed* `get_seq_length()`, which turned every compressed method's
    output to repetitive garbage on real hardware). A single combined pass
    would let each cache's compression "see" the question before it
    compresses, which this task exists specifically to prevent - see
    `eval/gist_mcq.py`'s module docstring.

    Batch-1, unpadded, per CLAUDE.md A6.
    """
    device = next(model.parameters()).device
    passage_text, question_text = _two_pass_texts(tokenizer, sample.context, sample.question)

    passage_ids = tokenizer(passage_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    question_ids = tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if passage_ids.shape[0] != 1:
        raise ValueError("batch-1 only (CLAUDE.md A6)")
    passage_len = int(passage_ids.shape[1])

    eos_ids = _eos_ids(model, tokenizer)
    penalty = getattr(getattr(model, "generation_config", None), "repetition_penalty", 1.0)
    rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty) if penalty and penalty != 1.0 else None

    caches = build_lanes(model, budget, overrides)
    lanes = {
        name: LaneState(name=name, cache=cache, telemetry=LaneTelemetryState(USES_CLUSTERING[name]))
        for name, cache in caches.items()
    }

    passage_token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in passage_ids[0].tolist()]
    question_token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in question_ids[0].tolist()]

    emit(
        {
            "type": "run_started",
            "lanes": LANE_METHODS,
            "variant": sample.variant,
            "question": sample.question,
            "options": sample.options,
            "passage_tokens": passage_len,
            "passage_token_texts": passage_token_texts,
            "question_token_texts": question_token_texts,
            "budget": budget,
            "max_new_tokens": sample.max_new_tokens,
        }
    )

    def prefill_segment(lane: LaneState, ids: torch.Tensor, start_pos: int, phase: str, step: int) -> None:
        n_new = ids.shape[1]
        new_positions = range(start_pos, start_pos + n_new)

        pos_ids = torch.arange(start_pos, start_pos + n_new, device=device).unsqueeze(0)
        logits = _forward_step(model, lane.cache, ids, pos_ids)
        lane.all_ids = ids.clone() if lane.all_ids is None else torch.cat([lane.all_ids, ids], dim=-1)
        lane.last_token = _argmax_next(logits, lane.all_ids, rep_penalty)
        lane.next_pos = start_pos + n_new

        snap = snapshot(lane.cache)
        diff = lane.telemetry.observe(new_positions, snap)
        emit(_step_event(lane, step=step, phase=phase, position=None, diff=diff, snap=snap, tokenizer=tokenizer))

    # ---- pass 1: the passage, with zero knowledge the question exists ----
    for lane in lanes.values():
        prefill_segment(lane, passage_ids, 0, "passage", step=0)

    # ---- pass 2: the question, continuing from each lane's own t_now ----
    for lane in lanes.values():
        start_pos = int(lane.cache.t_now) + 1
        prefill_segment(lane, question_ids, start_pos, "question", step=1)

    # ---- decode: round-robin, one new token per lane per round ----
    max_new_tokens = sample.max_new_tokens
    for decode_step in range(1, max_new_tokens + 1):
        if should_cancel():
            break
        step = decode_step + 1  # 0=passage, 1=question, so decode starts at 2
        any_active = False
        for lane in lanes.values():
            if lane.done:
                continue
            any_active = True

            pos_added = lane.next_pos
            pos_ids = torch.tensor([[pos_added]], device=device)
            logits = _forward_step(model, lane.cache, lane.last_token, pos_ids)

            token_id = int(lane.last_token[0, 0].item())
            lane.generated_ids.append(token_id)
            lane.all_ids = torch.cat([lane.all_ids, lane.last_token], dim=-1)
            lane.next_pos += 1
            lane.last_token = _argmax_next(logits, lane.all_ids, rep_penalty)
            lane.done = token_id in eos_ids or decode_step == max_new_tokens

            snap = snapshot(lane.cache)
            diff = lane.telemetry.observe(range(pos_added, pos_added + 1), snap)
            emit(
                _step_event(
                    lane, step=step, phase="decode", position=pos_added, diff=diff, snap=snap, tokenizer=tokenizer
                )
            )

            if lane.done:
                text = tokenizer.decode(lane.generated_ids, skip_special_tokens=True)
                chosen = _extract_choice(text)
                emit(
                    {
                        "type": "lane_complete",
                        "lane": lane.name,
                        "generated_text": text,
                        "n_tokens_generated": len(lane.generated_ids),
                        "chosen_letter": chosen,
                        "correct": chosen == sample.answer_letter,
                    }
                )
        if not any_active:
            break

    summary = []
    for lane in lanes.values():
        text = tokenizer.decode(lane.generated_ids, skip_special_tokens=True)
        chosen = _extract_choice(text)
        summary.append(
            {
                "lane": lane.name,
                "generated_text": text,
                "generated_ids": list(lane.generated_ids),
                "final_stats": lane.cache.get_stats(),
                "conservation_ok": bool(lane.cache.check_conservation()),
                "chosen_letter": chosen,
                "correct": chosen == sample.answer_letter,
            }
        )
    emit({"type": "run_complete", "summary": summary, "answer_letter": sample.answer_letter})
    return {"lanes": lanes, "summary": summary}
