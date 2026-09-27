"""The manual, interleaved, round-robin multi-lane decode loop.

Why this exists instead of `model.generate()`: the repo has no streaming
primitive (no `TextIteratorStreamer`/`StoppingCriteria`/`BaseStreamer`
anywhere), and `model.generate()` runs one lane to completion before you'd see
the next lane start - the demo wants all 5 lanes visibly advancing together,
one token per lane per round. This module is the only place that calls
`model(...)` directly instead of going through `generate()`.

It reuses everything else as-is: `make_cache` builds the caches, `attach_cache`
routes attention through them, and every reported number
(`get_stats()`/`check_conservation()`) is read straight off the cache, never
recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import RepetitionPenaltyLogitsProcessor

from src.attn_patch import attach_cache

from .lanes import LANE_METHODS, build_lanes
from .telemetry import LaneTelemetryState, SlotSnapshot, snapshot

EmitFn = Callable[[dict], None]


@dataclass
class LaneState:
    name: str
    cache: object
    next_pos: int = 0
    last_token: torch.Tensor | None = None
    all_ids: torch.Tensor | None = None  # prompt + generated so far, for repetition_penalty
    done: bool = False
    generated_ids: list[int] = field(default_factory=list)
    telemetry: LaneTelemetryState = field(default_factory=LaneTelemetryState)
    prev_snapshot: SlotSnapshot = field(default_factory=lambda: SlotSnapshot([], [], []))


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


def _narrate(lane: LaneState, diff, phase: str) -> str:
    """One plain-English sentence describing what this step just did to the
    cache - so the UI never depends on a viewer correctly reading a diagram.
    """
    stats = lane.cache.get_stats()
    if diff.merged:
        biggest = max(diff.merged, key=lambda m: m.weight)
        return (
            f"Merged {len(biggest.member_positions)} older tokens into one summary slot "
            f"(now stands for {biggest.weight} tokens)."
        )
    if diff.evicted_positions:
        return f"Forgot {len(diff.evicted_positions)} old token(s) to stay within budget."
    if phase == "prefill":
        return f"Read the prompt — {stats['n_tokens_cached']} tokens cached."
    return "Added the new token — still within budget, nothing removed."


def _step_event(
    lane: LaneState, *, step: int, phase: str, position: int | None, snap: SlotSnapshot, diff, tokenizer
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
            "merged": [vars(m) for m in diff.merged],
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
    changes the argmax on any prompt long enough to repeat itself - skipping
    it made this loop's "full" lane silently diverge from the real
    `eval.memory.generate_and_measure` path it's supposed to match.
    """
    if rep_penalty is not None:
        logits = rep_penalty(all_ids, logits)
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def run(
    model,
    tokenizer,
    prompt: str,
    *,
    budget: float,
    max_new_tokens: int,
    overrides: dict | None,
    emit: EmitFn,
    should_cancel: Callable[[], bool] = lambda: False,
) -> dict:
    """Run all 5 lanes on the same prompt, interleaved one token at a time.

    Batch-1, unpadded, per CLAUDE.md A6 - raises if the tokenizer ever
    produces more than one row for a single prompt string.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device)
    if prompt_ids.shape[0] != 1:
        raise ValueError("batch-1 only (CLAUDE.md A6)")
    prompt_len = int(prompt_ids.shape[1])

    eos_ids = _eos_ids(model, tokenizer)
    penalty = getattr(getattr(model, "generation_config", None), "repetition_penalty", 1.0)
    rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty) if penalty and penalty != 1.0 else None

    caches = build_lanes(model, budget, overrides)
    lanes = {name: LaneState(name=name, cache=cache) for name, cache in caches.items()}

    prompt_token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in prompt_ids[0].tolist()]

    emit(
        {
            "type": "run_started",
            "lanes": LANE_METHODS,
            "prompt_tokens": prompt_len,
            "prompt_token_texts": prompt_token_texts,
            "budget": budget,
            "max_new_tokens": max_new_tokens,
        }
    )

    # ---- prefill: one full forward per lane over the shared prompt ----
    for lane in lanes.values():
        # Seed the "before" state as the full, uncompressed prompt - not
        # truly empty. If `prompt_len > budget_tokens`, prefill's own
        # `_compress()` call runs *inside* the single forward pass below, so
        # without this seed the diff would compare an empty prior against an
        # already-compressed result and silently miss every eviction/merge
        # that happened during prefill itself (they'd look like positions
        # that were simply never seen, not ones that got dropped).
        for p in range(prompt_len):
            lane.telemetry.membership.setdefault(p, {p})
        lane.prev_snapshot = SlotSnapshot(
            positions=list(range(prompt_len)), weights=[1] * prompt_len, is_centroid=[False] * prompt_len
        )

        pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits = _forward_step(model, lane.cache, prompt_ids, pos_ids)
        lane.all_ids = prompt_ids.clone()
        lane.last_token = _argmax_next(logits, lane.all_ids, rep_penalty)
        lane.next_pos = prompt_len
        snap = snapshot(lane.cache)
        diff = lane.telemetry.diff(lane.prev_snapshot, snap)
        lane.prev_snapshot = snap
        emit(_step_event(lane, step=0, phase="prefill", position=None, snap=snap, diff=diff, tokenizer=tokenizer))

    # ---- decode: round-robin, one new token per lane per round ----
    for step in range(1, max_new_tokens + 1):
        if should_cancel():
            break
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
            lane.done = token_id in eos_ids or step == max_new_tokens

            snap = snapshot(lane.cache)
            diff = lane.telemetry.diff(lane.prev_snapshot, snap)
            lane.prev_snapshot = snap
            emit(
                _step_event(
                    lane, step=step, phase="decode", position=pos_added, snap=snap, diff=diff, tokenizer=tokenizer
                )
            )

            if lane.done:
                emit(
                    {
                        "type": "lane_complete",
                        "lane": lane.name,
                        "generated_text": tokenizer.decode(lane.generated_ids, skip_special_tokens=True),
                        "n_tokens_generated": len(lane.generated_ids),
                    }
                )
        if not any_active:
            break

    summary = [
        {
            "lane": lane.name,
            "generated_text": tokenizer.decode(lane.generated_ids, skip_special_tokens=True),
            "generated_ids": list(lane.generated_ids),
            "final_stats": lane.cache.get_stats(),
            "conservation_ok": bool(lane.cache.check_conservation()),
        }
        for lane in lanes.values()
    ]
    emit({"type": "run_complete", "summary": summary})
    return {"lanes": lanes, "summary": summary}
