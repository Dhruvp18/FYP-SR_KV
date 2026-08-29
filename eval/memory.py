"""Generation + measurement harness.

This module owns the single place where `model.generate()` is called. It
attaches a cache, measures peak CUDA memory and throughput around the call,
and reads `get_stats()` afterwards. It contains no eviction logic of any kind -
that would violate the contract - and it must stay that way.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from src.attn_patch import attach_cache


@contextmanager
def peak_memory():
    """Yields a dict that gets `max_memory_allocated` filled in on exit."""
    result: dict = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.max_memory_allocated()
    else:
        start = 0
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
            result["memory_allocated_delta"] = int(torch.cuda.max_memory_allocated() - start)
        else:
            result["max_memory_allocated"] = 0
            result["memory_allocated_delta"] = 0


def build_prompt(tokenizer, context: str, question: str, use_chat_template: bool = True) -> str:
    """Format a (context, question) pair for an instruct model."""
    user = f"{context}\n\n{question}"
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
    return user


@torch.no_grad()
def generate_and_measure(
    model,
    tokenizer,
    prompt: str,
    cache,
    *,
    max_new_tokens: int = 32,
    device: torch.device | None = None,
) -> dict:
    """Run one generation with `cache` attached and report what it cost.

    Batch size is 1 and nothing is padded, per CLAUDE.md A6.
    """
    device = device or next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[0] != 1:
        raise ValueError("SR-KV eval is batch-1 only (CLAUDE.md A6)")

    prompt_len = int(input_ids.shape[1])

    with peak_memory() as mem:
        started = time.perf_counter()
        with attach_cache(model, cache):
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started

    generated_ids = output[0, prompt_len:]
    n_new = int(generated_ids.shape[0])
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = cache.get_stats()
    history = list(getattr(cache, "budget_history", []))
    return {
        "generated_text": text,
        "prompt_tokens": prompt_len,
        "generated_tokens": n_new,
        "seconds": elapsed,
        "tokens_per_sec": (n_new / elapsed) if elapsed > 0 else 0.0,
        "max_memory_allocated": mem["max_memory_allocated"],
        "memory_allocated_delta": mem["memory_allocated_delta"],
        "cache_stats": stats,
        "budget_used_pct_max": max(history) if history else 100.0,
        "budget_used_pct_final": history[-1] if history else 100.0,
        "conservation_ok": bool(cache.check_conservation()),
        "cache_config": cache.config_dict(),
    }
